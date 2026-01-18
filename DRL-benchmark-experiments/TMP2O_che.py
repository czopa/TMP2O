
from tqdm import tqdm
import argparse
import importlib
import matplotlib.pyplot as plt
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import math
import pandas as pd
from tqdm import tqdm

seeds_ = [0, 1, 25, 5, 3]
for see_ in tqdm(seeds_):
    nllll = []
    agentlll = []
    rnn = []
    rne = []
    pgllll= []
    elll = []
    def parse_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--exp-name", type=str, default='tmp2o_cheetah')
        parser.add_argument("--seed", type=int, default=see_)
        parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
        parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
        parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
        parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
        parser.add_argument("--wandb-entity", type=str, default=None)
        parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    
        # PPO args
        parser.add_argument("--env-id", type=str, default="HalfCheetah-v4")
        parser.add_argument("--total-timesteps", type=int, default=500000)  # small for testing
        parser.add_argument("--learning-rate", type=float, default=2.5e-4)
        parser.add_argument("--num-envs", type=int, default=1)
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
    
    def make_env(env_id, idx, capture_video, run_name, gamma):
        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env
    
        return thunk
        
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
            self.hidden_dim = 64
            super().__init__()
            obs_shape = envs.single_observation_space.shape
            obs_dim = int(np.prod(obs_shape))
            
            self.critic = layer_init(nn.Linear(self.hidden_dim, 1), std=1.0)
            
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
            
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_dim, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, self.hidden_dim)),
                nn.Tanh(),
            )
        def get_hidden(self, x):
            x = x.clone()
            h = self.network(x)
            return h
    
        def get_value_from_hidden(self, hidden):
            return self.critic(hidden).squeeze(-1)
    
        def get_value(self, x):
            hidden = self.get_hidden(x)
            return self.get_value_from_hidden(hidden)
    
        def get_action_and_value(self, x, action=None):
            hidden = self.get_hidden(x)
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(dim =-1), probs.entropy().sum(dim =-1), self.critic(hidden), hidden
    
    
    class DynamicsModel(nn.Module):
        """
        Predict next-hidden mean & logvar given (hidden, action one-hot).
        Diagonal Gaussian output.
        """
        def __init__(self, hidden_dim=64, n_actions=6, hidden_size=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(70, hidden_size),
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
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/TMP3O")
    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        )
    
    # ----- agent & dynamics -----
    agent = Agent(envs).to(device)
    action_dim = int(np.prod(envs.single_action_space.shape))
    dyn = DynamicsModel(hidden_dim=64, n_actions=envs.action_space.shape[0]).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    optimizer_dyn = optim.Adam(dyn.parameters(), lr=args.dyn_lr, eps=1e-5)
    # storage (time-major): shapes (num_steps, num_envs, ...)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape, dtype=torch.float32, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs, action_dim), dtype=torch.float32, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    terminations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    truncations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    # hidden buffers for dynamics training
    hiddens = torch.zeros((args.num_steps, args.num_envs, agent.hidden_dim), dtype=torch.float32, device=device)
    next_hiddens = torch.zeros((args.num_steps, args.num_envs, agent.hidden_dim), dtype=torch.float32, device=device)
    
    reset_out = envs.reset(seed=args.seed)
    next_obs = to_torch_obs(reset_out, device)
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
    
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            terminations[step] = next_termination
            truncations[step] = next_truncation
    
            with torch.no_grad():
                action, logprob, _, value, hidden = agent.get_action_and_value(next_obs)
                values[step] = value.reshape(-1)
                hiddens[step] = hidden  # store hidden for dynamics training
            actions[step] = action
            logprobs[step] = logprob
    
            step_out = envs.step(action.cpu().numpy())  # Gymnasium parallel vec returns (obs, reward, term, trunc, infos)
    
            next_obs_raw, reward_raw, term_raw, trunc_raw, info = step_out
            
            next_obs = to_torch_obs(next_obs_raw, device)
            rewards[step] = torch.tensor(reward_raw, dtype=torch.float32, device=device)
            next_termination = torch.tensor(term_raw, dtype=torch.float32, device=device)
            next_truncation = torch.tensor(trunc_raw, dtype=torch.float32, device=device)
    
            with torch.no_grad():
                next_hidden = agent.get_hidden(next_obs)
                next_hiddens[step] = next_hidden
            for idx, item in enumerate(info):
                player_idx = idx % 2
                if isinstance(item, dict) and "episode" in item:
                    writer.add_scalar(f"charts/episodic_return-player{player_idx}", item["episode"]["r"], global_step)
                    writer.add_scalar(f"charts/episodic_length-player{player_idx}", item["episode"]["l"], global_step)
    
        flat_z = hiddens.view(-1, 64)
        flat_next_z = next_hiddens.view(-1, 64)
        flat_actions = actions.view((-1, 6)) #here!
        dataset_size = flat_z.shape[0]
        
        if dataset_size > 0:
            batch_dyn = 256
            for epoch in range(args.dyn_epochs):
                perm = torch.randperm(dataset_size, device=device)
                for start in range(0, dataset_size, batch_dyn):
                    idx = perm[start:start+batch_dyn]
                    z_b = flat_z[idx]
                    znext_b = flat_next_z[idx]
                    a_b = flat_actions[idx]
                    mean, logvar = dyn(z_b, a_b)
                    var = torch.exp(logvar)
                    nll = 0.5 * (((znext_b - mean) ** 2) / var + logvar).mean()
                    optimizer_dyn.zero_grad()
                    nll.backward()
                    nn.utils.clip_grad_norm_(dyn.parameters(), args.max_grad_norm)
                    optimizer_dyn.step()
        with torch.no_grad():
            T_flat = np.zeros(args.batch_size, dtype=np.float32)  # will fill with robust targets
            batch_size = args.batch_size
            K = args.K_model_samps
            theta = float(args.theta_mp)
    
            b_rewards = rewards.view(-1).cpu().numpy()  # shape (batch_size,)
            b_dones_term = terminations.view(-1).cpu().numpy()
            b_dones_trunc = truncations.view(-1).cpu().numpy()
            b_dones = np.maximum(b_dones_term, b_dones_trunc)  # treat either as termination
            b_hiddens = flat_z  # tensor (batch_size, 512)
            b_actions = flat_actions#.cpu().numpy() #HERE!
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
                # terminal transitions: T = r, nonterminal: robust expectation via sampled latents
                nonterm_idx = np.where(done_batch == 0)[0]
                if len(nonterm_idx) == 0:
                    T_flat[start:end] = r_batch
                    continue
                
                z_nonterm = z_batch[nonterm_idx]           # (bn, D)
                a_nonterm = a_batch[nonterm_idx].to(device)  # (bn, action_dim)
                r_nonterm = torch.tensor(r_batch[nonterm_idx], dtype=torch.float32, device=device)
    
                samples = dyn.sample(z_nonterm, a_nonterm, K=K)
                bn = samples.shape[0]
                samples_flat = samples.view(bn * K, 64)  
                V_flat = agent.get_value_from_hidden(samples_flat).view(bn, K)  # (bn, K)
                r_expand = r_nonterm.unsqueeze(1).expand(-1, K)  # (bn, K)
                f_k = r_expand + args.gamma * V_flat  # (bn, K)
                arg = - f_k / float(theta)
                lme = torch.logsumexp(arg, dim=1) - math.log(K)
                T_nonterm = - float(theta) * lme  # (bn,)
                T_block = r_batch.copy()
                T_block[nonterm_idx] = T_nonterm.cpu().numpy()
                T_flat[start:end] = T_block
    
            T_mat = T_flat.reshape(args.num_steps, args.num_envs)
            V_mat = values.cpu().numpy()
            deltas = T_mat - V_mat
    
            advantages = np.zeros_like(deltas, dtype=np.float32)
            for env_i in range(args.num_envs):
                lastgaelam = 0.0
                next_done_env = float(max(next_termination[env_i].item(), next_truncation[env_i].item()))
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_env
                    else:
                        nextnonterminal = 1.0 - max(terminations[t + 1, env_i].item(), truncations[t + 1, env_i].item())
                    lastgaelam = deltas[t, env_i] + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t, env_i] = lastgaelam
            returns = advantages + V_mat
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)  # tensor
        b_logprobs = logprobs.reshape(-1).cpu()
        b_actions = actions.reshape(-1, action_dim)
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
                n_b = obs_mb
                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(obs_mb, actions_mb)
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
                agentlll.append(loss.item())
                loss.backward()
                rnn.append(b_returns.mean().item())
                rne.append(b_rewards.mean())
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
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        df = pd.DataFrame({'rewards':rnn})
        df.to_csv(f'tmp2o_chet_{see_}.csv')
    del agent
    del optimizer
    del optimizer_dyn
    envs.close()
    writer.close()
