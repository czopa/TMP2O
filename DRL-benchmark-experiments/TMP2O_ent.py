import imageio
from tqdm import tqdm

import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool

import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import math


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename('').rstrip(".py"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    # PPO args
    parser.add_argument("--env-id", type=str, default="entombed_cooperative_v3")
    parser.add_argument("--total-timesteps", type=int, default=600000)  # small for testing
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=None)

    # Multiplier-pref & dynamics args
    parser.add_argument("--theta-mp", type=float, default=1.0, help="theta penalty for multiplier preferences")
    parser.add_argument("--K-model-samps", type=int, default=8, help="samples per (z,a) to approximate T")
    parser.add_argument("--dyn-lr", type=float, default=1e-3)
    parser.add_argument("--dyn-epochs", type=int, default=2)

    args = parser.parse_args('')
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def to_torch_obs(x, device):
    """
    Robust conversion for env.reset() and env.step() outputs (Gymnasium style).
    Returns a torch tensor on device with shape (num_envs, H, W, C), dtype float32.
    """
    if isinstance(x, tuple) and len(x) >= 1:
        obs = x[0]
    else:
        obs = x
    arr = np.asarray(obs)
    if arr.ndim == 4 and arr.shape[1] in (1, 3, 4):
        arr = arr.transpose(0, 2, 3, 1)
    if arr.ndim == 3:
        arr = arr[None]
    return torch.tensor(arr, dtype=torch.float32, device=device)


def logmeanexp(x, dim=-1):
    # stable log(mean(exp(x))) = logsumexp(x) - log(n)
    lse = torch.logsumexp(x, dim=dim)
    n = x.shape[dim]
    return lse - math.log(n)


class Agent(nn.Module):
    """
    Actor-Critic that returns hidden embedding too.
    Input obs expected channels-last (B,H,W,C); Agent converts and normalizes inside.
    """

    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(6, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    def get_hidden(self, x):
        # x: tensor (B, H, W, C)
        x = x.clone()
        # normalize the first 4 channels (frame stack)
        x[:, :, :, [0, 1, 2, 3]] /= 255.0
        h = self.network(x.permute((0, 3, 1, 2)))
        return h

    def get_value_from_hidden(self, hidden):
        return self.critic(hidden).squeeze(-1)

    def get_value(self, x):
        hidden = self.get_hidden(x)
        return self.get_value_from_hidden(hidden)

    def get_action_and_value(self, x, action=None):
        # returns action, logprob, entropy, value, hidden
        hidden = self.get_hidden(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden).squeeze(-1), hidden


class DynamicsModel(nn.Module):
    """
    Predict next-hidden mean & logvar given (hidden, action one-hot).
    Diagonal Gaussian output.
    """

    def __init__(self, hidden_dim=512, n_actions=6, hidden_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + n_actions, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_size, hidden_dim)
        self.logvar_head = nn.Linear(hidden_size, hidden_dim)

    def forward(self, z, a_onehot):
        x = torch.cat([z, a_onehot], dim=-1)
        h = self.net(x)
        mean = self.mean_head(h)
        logvar = torch.clamp(self.logvar_head(h), -10.0, 1.0)
        return mean, logvar

    def sample(self, z, a_onehot, K):
        # z: (B, D), a_onehot: (B, A)
        mean, logvar = self.forward(z, a_onehot)
        std = (0.5 * logvar).exp()
        B, D = mean.shape
        mean_k = mean.unsqueeze(1).expand(-1, K, -1)
        std_k = std.unsqueeze(1).expand(-1, K, -1)
        eps = torch.randn_like(std_k)
        samples = mean_k + std_k * eps  # (B, K, D)
        return samples


args = parse_args()


seeds = [1,3,5,10,20,25]
for seed_ in seeds:
    nllll = []
    agentlll = []
    rnn = []
    rne = []
    pgllll = []
    elll = []
    print('==========================================================================')
    print('seed: ', seed_)
    run_name = f"TMP2O_50_{seed_}"
    writer = SummaryWriter(f"runs/TMP3O")
    # seeding
    random.seed(seed_)
    np.random.seed(seed_)
    torch.manual_seed(seed_)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    # ----- env setup (parallel PettingZoo Atari Pong) -----
    env = importlib.import_module(f"pettingzoo.atari.{args.env_id}").parallel_env()
    env = ss.max_observation_v0(env, 2)
    env = ss.frame_skip_v0(env, 4)
    env = ss.clip_reward_v0(env, lower_bound=-1, upper_bound=1)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4)
    env = ss.agent_indicator_v0(env, type_only=False)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env, args.num_envs // 2, num_cpus=0, base_class="gymnasium")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True

    # ----- agent & dynamics -----
    agent = Agent(envs).to(device)
    dyn = DynamicsModel(hidden_dim=512, n_actions=envs.single_action_space.n).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer_dyn = optim.Adam(dyn.parameters(), lr=args.dyn_lr, eps=1e-5)
    # storage (time-major): shapes (num_steps, num_envs, ...)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float32,
                      device=device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=torch.int64,
                          device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    terminations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    truncations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    # hidden buffers for dynamics training
    hiddens = torch.zeros((args.num_steps, args.num_envs, 512), dtype=torch.float32, device=device)
    next_hiddens = torch.zeros((args.num_steps, args.num_envs, 512), dtype=torch.float32, device=device)
    # initial reset (Gymnasium reset returns (obs, info))
    reset_out = envs.reset(seed=seed_)
    next_obs = to_torch_obs(reset_out, device)  # shape (num_envs, H, W, C)
    # initial termination/truncation
    next_termination = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    next_truncation = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    global_step = 0
    start_time = time.time()
    num_updates = args.total_timesteps // args.batch_size
    for update in tqdm(range(1, int(num_updates) + 1)):
        # anneal lr
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / max(1, num_updates)
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        # ---------- Collect rollout ----------
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation
            # action selection
            with torch.no_grad():
                action, logprob, _, value, hidden = agent.get_action_and_value(next_obs)
                values[step] = value.reshape(-1)
                hiddens[step] = hidden  # store hidden for dynamics training
            actions[step] = action
            logprobs[step] = logprob
            # step envs (action must be numpy array)
            step_out = envs.step(action.cpu().numpy())  # Gymnasium parallel vec returns (obs, reward, term, trunc, infos)
            # Unpack according to Gymnasium Vector API
            next_obs_raw, reward_raw, term_raw, trunc_raw, info = step_out
            # convert to tensors

            next_obs = to_torch_obs(next_obs_raw, device)
            rewards[step] = torch.tensor(reward_raw, dtype=torch.float32, device=device)
            next_termination = torch.tensor(term_raw, dtype=torch.float32, device=device)
            next_truncation = torch.tensor(trunc_raw, dtype=torch.float32, device=device)
            # compute and store next_hidden for dynamics training
            with torch.no_grad():
                next_hidden = agent.get_hidden(next_obs)  # (num_envs, 512)
                next_hiddens[step] = next_hidden
            # logging episodes via info dict (PettingZoo supplies episodic info per agent slot)
            # Info length == num_envs; each entry might include 'episode' key when an episode ended for that sub-env
            for idx, item in enumerate(info):
                player_idx = idx % 2
                if isinstance(item, dict) and "episode" in item:
                    writer.add_scalar(f"charts/episodic_return-player{player_idx}", item["episode"]["r"], global_step)
                    writer.add_scalar(f"charts/episodic_length-player{player_idx}", item["episode"]["l"], global_step)
        # ---------- Train dynamics model (on collected hidden pairs) ----------
        # Flatten (T * N, D)
        flat_z = hiddens.view(-1, 512)  # tensor
        flat_next_z = next_hiddens.view(-1, 512)
        flat_actions = actions.view(-1).to(torch.long)  # shape (batch_size,)
        dataset_size = flat_z.shape[0]
        if dataset_size > 0:
            a_onehot_full = torch.zeros((dataset_size, envs.single_action_space.n), device=device)
            a_onehot_full[torch.arange(dataset_size), flat_actions] = 1.0
            # dynamics training
            batch_dyn = 256
            for epoch in range(args.dyn_epochs):
                perm = torch.randperm(dataset_size, device=device)
                for start in range(0, dataset_size, batch_dyn):
                    idx = perm[start: start + batch_dyn]
                    z_b = flat_z[idx]
                    znext_b = flat_next_z[idx]
                    a_onehot_b = a_onehot_full[idx]
                    mean, logvar = dyn(z_b, a_onehot_b)
                    var = torch.exp(logvar)
                    # Gaussian NLL (mean + logvar)
                    nll = 0.5 * (((znext_b - mean) ** 2) / var + logvar).mean()
                    optimizer_dyn.zero_grad()
                    nll.backward()
                    nn.utils.clip_grad_norm_(dyn.parameters(), args.max_grad_norm)
                    optimizer_dyn.step()
        # ---------- Compute robust one-step targets T and GAE (per env) ----------
        # Flatten buffers to prepare T computation in batches
        with torch.no_grad():
            T_flat = np.zeros(args.batch_size, dtype=np.float32)  # will fill with robust targets
            batch_size = args.batch_size
            K = args.K_model_samps
            theta = float(args.theta_mp)
            # flattened buffers (numpy for ease)
            b_rewards = rewards.view(-1).cpu().numpy()  # shape (batch_size,)
            b_dones_term = terminations.view(-1).cpu().numpy()
            b_dones_trunc = truncations.view(-1).cpu().numpy()
            b_dones = np.maximum(b_dones_term, b_dones_trunc)  # treat either as termination
            b_hiddens = flat_z  # tensor (batch_size, 512)
            b_actions = flat_actions.cpu().numpy()
            # compute in minibatches
            B = 256
            for start in range(0, batch_size, B):
                end = min(batch_size, start + B)
                z_batch = b_hiddens[start:end]  # tensor (b,512)
                a_batch = b_actions[start:end]
                r_batch = b_rewards[start:end]
                done_batch = b_dones[start:end]
                bsz = z_batch.shape[0]
                if bsz == 0:
                    continue
                # create action one-hot
                a_onehot = torch.zeros((bsz, envs.single_action_space.n), device=device)
                a_tensor = torch.tensor(a_batch, dtype=torch.long, device=device)
                a_onehot[torch.arange(bsz, device=device), a_tensor] = 1.0
                # handle nonterminal and terminal separately: T = r for terminal transitions
                nonterm_idx = np.where(done_batch == 0)[0]
                if len(nonterm_idx) == 0:
                    T_flat[start:end] = r_batch
                    continue
                # select nonterminal subset
                z_nonterm = z_batch[nonterm_idx]  # (bn,512)
                a_onehot_nonterm = a_onehot[nonterm_idx]
                r_nonterm = torch.tensor(r_batch[nonterm_idx], dtype=torch.float32, device=device)
                # sample K next-hidden candidates from dynamics model
                samples = dyn.sample(z_nonterm, a_onehot_nonterm, K=K)  # (bn, K, 512)
                bn = samples.shape[0]
                samples_flat = samples.view(bn * K, 512)  # (bn*K, 512)
                # evaluate critic on samples (treat samples as hidden latents)
                V_flat = agent.get_value_from_hidden(samples_flat).view(bn, K)  # (bn, K)
                # compute f_k = r + gamma * V_k
                r_expand = r_nonterm.unsqueeze(1).expand(-1, K)  # (bn, K)
                f_k = r_expand + args.gamma * V_flat  # (bn, K)
                # compute arg = -f_k / theta, then logmeanexp over K
                arg = - f_k / float(theta)
                # logmeanexp = logsumexp(arg) - log(K)
                lme = torch.logsumexp(arg, dim=1) - math.log(K)
                T_nonterm = - float(theta) * lme  # (bn,)
                # prepare T block for this minibatch
                T_block = r_batch.copy()
                # fill nonterm locations (convert to numpy)
                T_block[nonterm_idx] = T_nonterm.cpu().numpy()
                T_flat[start:end] = T_block
            # reshape T_flat -> (num_steps, num_envs)
            T_mat = T_flat.reshape(args.num_steps, args.num_envs)
            # values tensor -> (num_steps, num_envs)
            V_mat = values.cpu().numpy()
            # compute deltas per (t,env)
            deltas = T_mat - V_mat
            # compute GAE per environment (iterate env index)
            advantages = np.zeros_like(deltas, dtype=np.float32)
            for env_i in range(args.num_envs):
                lastgaelam = 0.0
                # next_done for final step uses next_termination/next_truncation per env from last step
                next_done_env = float(max(next_termination[env_i].item(), next_truncation[env_i].item()))
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_env
                    else:
                        nextnonterminal = 1.0 - max(terminations[t + 1, env_i].item(), truncations[t + 1, env_i].item())
                    lastgaelam = deltas[t, env_i] + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t, env_i] = lastgaelam
            returns = advantages + V_mat
        # flatten for PPO update
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)  # tensor
        b_logprobs = logprobs.reshape(-1).cpu()
        b_actions = actions.reshape(-1)
        b_advantages = torch.tensor(advantages.reshape(-1), dtype=torch.float32, device=device)
        b_returns = torch.tensor(returns.reshape(-1), dtype=torch.float32, device=device)
        b_values = torch.tensor(V_mat.reshape(-1), dtype=torch.float32, device=device)
        # ---------- PPO update (standard) ----------
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                obs_mb = b_obs[mb_inds].to(device)
                actions_mb = b_actions[mb_inds].to(device)
                oldlogprob_mb = b_logprobs[mb_inds].to(device)
                adv_mb = b_advantages[mb_inds].to(device)
                ret_mb = b_returns[mb_inds].to(device)
                val_mb = b_values[mb_inds].to(device)
                rnn.append(ret_mb.mean().item())
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(obs_mb, actions_mb.long())
                logratio = newlogprob - oldlogprob_mb
                ratio = logratio.exp()
                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                mb_advantages = adv_mb
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                # value loss (clipped or not)
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - ret_mb) ** 2
                    v_clipped = val_mb + torch.clamp(newvalue - val_mb, -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - ret_mb) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - ret_mb) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                pgllll.append(pg_loss.item())
                elll.append(entropy_loss.item())

                optimizer.zero_grad()
                loss.backward()
                
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        # diagnostics
        y_pred = b_values.cpu().numpy()
        y_true = b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("losses/rewards", b_rewards.mean(), global_step)
        # print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    import pandas as pd
    print(len(rnn))
    loss_df = pd.DataFrame({
        'rewards':rnn
    })
    loss_df.to_csv(f'TMP2O_ent_{seed_}.csv')
    del agent
    del dyn
    del optimizer_dyn
    del optimizer
    envs.close()
    writer.close()
