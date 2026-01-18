from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from ontolearn.utils.static_funcs import concept_len, init_length_metric
from owlready2 import get_ontology
from ontolearn.metrics import F1, Accuracy
from ontolearn.quality_funcs import evaluate_concept
from ontolearn.learning_problem import PosNegLPStandard
from owlapy.owl_individual import OWLNamedIndividual
from owlapy import owl_expression_to_sparql, owl_expression_to_dl, class_expression
from ontolearn.utils.static_funcs import save_owl_class_expressions
from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from scipy.special import softmax
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from copy import deepcopy
from pettingzoo.test import parallel_api_test
import functools
import random
from copy import copy

import numpy as np
from gymnasium.spaces import Discrete, MultiDiscrete

from pettingzoo import ParallelEnv
import torch.nn.functional as F
from ontolearn.refinement_operators import ExpressRefinement, ModifiedCELOERefinement

from collections import defaultdict
import random
from rdflib import Graph
from rdflib import BNode
from rdflib import Graph, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD, Namespace
import pandas as pd
import time
import csv
import warnings

import json
from io import StringIO
from collections import defaultdict
import catboost as cb
from sklearn import svm

from catboost import CatBoostClassifier, Pool
import numpy as np
from catboost import Pool
import json
import tempfile
import os
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph, rdflib_to_networkx_graph

import networkx as nx
warnings.filterwarnings('ignore')
from tqdm import tqdm
import argparse
import importlib
import os
import random
import time
from distutils.util import strtobool
import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet, Categorical
import gymnasium as gym
import numpy as np
import supersuit as ss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal

from torch.utils.tensorboard import SummaryWriter
import math
import matplotlib.pyplot as plt
from ontolearn.incomplete_kb import make_kb_incomplete_ass 
import argparse
import importlib
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
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.running_mean_std import RunningMeanStd
from ontolearn.utils.static_funcs import concept_len
from ontolearn.metrics import F1, Accuracy
from ontolearn.quality_funcs import evaluate_concept
from ontolearn.learning_problem import PosNegLPStandard
from gymnasium import spaces
import functools
from pettingzoo import ParallelEnv
import numpy as np
from itertools import chain, combinations
from math import fsum
from typing import List, Sequence, Optional
from copy import copy
from scipy.special import logsumexp

lps = ['/upb/users/a/amgad/profiles/unix/cs/animals/',
 '/upb/users/a/amgad/profiles/unix/cs/suramin/',
 '/upb/users/a/amgad/profiles/unix/cs/nctrer/',
 '/upb/users/a/amgad/profiles/unix/cs/mammographic/',
 '/upb/users/a/amgad/profiles/unix/cs/carcinogenesis/',
 '/upb/users/a/amgad/profiles/unix/cs/hepatitis/']
kbs = ['/upb/users/a/amgad/profiles/unix/cs/35/mammographic.owl',
       '/upb/users/a/amgad/profiles/unix/cs/35/suramin.owl',
       '/upb/users/a/amgad/profiles/unix/cs/35/nctrer.owl',
       '/upb/users/a/amgad/profiles/unix/cs/35/hepatitis.owl',
       '/upb/users/a/amgad/profiles/unix/cs/35/animals.owl',
       '/upb/users/a/amgad/profiles/unix/cs/35/carcinogenesis.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/mammographic.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/suramin.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/nctrer.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/hepatitis.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/animals.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/carcinogenesis.owl',
       '/upb/users/a/amgad/profiles/unix/cs/25/mammographic.owl',
       '/upb/users/a/amgad/profiles/unix/cs/15/suramin.owl',
       '/upb/users/a/amgad/profiles/unix/cs/15/nctrer.owl',
       '/upb/users/a/amgad/profiles/unix/cs/15/hepatitis.owl',
       '/upb/users/a/amgad/profiles/unix/cs/15/animals.owl',
       '/upb/users/a/amgad/profiles/unix/cs/15/carcinogenesis.owl']
d = {}
for i in lps:
    for j in kbs:
        if i.split('/')[-2] in j.split('/')[-1].replace('.owl', ''):
             d[j] = i
            
seeds_ = [2, 5, 25, 42]
for seed_ in seeds_:        
    for original_g, lp_ in d.items():
        kb_name = original_g.split('/')[-1].split('.')[0]
        kb_prct = original_g.split('/')[-2]
        pos_file = lp_ + r'pos.txt'
        neg_file = lp_ + r'neg.txt'
        with open(pos_file, encoding='utf-8') as file:
            pos_data = file.readlines()
        
        with open(neg_file, encoding='utf-8') as file:
            neg_data = file.readlines()
        
        kb = KnowledgeBase(path=original_g)
        pos = set([OWLNamedIndividual(x) for x in [x.replace('\n','') for x in pos_data][:-1]])
        neg = set([OWLNamedIndividual(x) for x in [x.replace('\n','') for x in neg_data]])
        
        
        lp = PosNegLPStandard(pos = pos, neg=neg)
        lp.all = pos.union(neg)
        print(kb)
        print(lp_)
        
        
        class ALC_Env(ParallelEnv):
            metadata = {"name": "alc_dl_v0", "render_modes": ["human"]}
        
            def __init__(
                    self,
                    knoweldge_base,
                    pos,
                    neg,
                    reward_type,
                    max_action,
                    training=True,
                    render_mode=None,
            ):
        
                self.kb = knoweldge_base
                self.render_mode = render_mode
                self.cg = self.kb.generator
                self.pos = pos
                self.neg = neg
                self.training = training
                self.stepin_cand = self.fill_stepin_cand(self.pos)
        
                self.obs_rms = RunningMeanStd()
        
                self.lp = PosNegLPStandard(pos=pos, neg=neg)
                self.lp.all = pos.union(neg)
        
                self.reward_type = reward_type
                self.max_action = max_action
        
                self.operations = [
                    self.cg.union,
                    self.cg.intersection,
                    self.cg.negation,
                    self.cg.existential_restriction,
                    self.cg.universal_restriction
                ]
                
        
                self.concepts = list(self.kb.concepts)
                self.properties = list(self.kb.object_properties)
        
                self.mappings = {
                    "concepts": dict(
                        [(idx, val) for idx, val in enumerate(self.concepts)]
                    ),
                    "properties": dict(
                        [(idx, val) for idx, val in enumerate(self.properties)]
                    ),
                    "operations": dict(
                        [(idx, val) for idx, val in enumerate(self.operations)]
                    ),
                }
                self.possible_agents = ["a_0", "a_1"]
                self.episode = 1
        
            def reset(self, seed=None, options=None):
                self.agents = copy(self.possible_agents)
                self.f1 = 0
                self.time_stamp = 0
                self.expression = self._init_expression(
                    np.random.choice(list(self.pos))
                )
                if self.expression == None:
                    try:
                        self.expression = np.random.choice(self.stepin_cand)
                    except:
                        self.expression = np.random.choice(list(self.concepts))
                observations = {a: self.normalized_obs() for a in self.agents}
                infos = {a: {} for a in self.agents}
        
                return observations, infos
        
            def _get_f1(self, expression):
                return evaluate_concept(
                    self.kb, expression, F1(), self.lp.encode_kb(self.kb)
                ).q
        
            def normalized_obs(self, epsilon=1e-8):
                '''
                This function is inspired by:
                https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/vec_env/vec_normalize.py
                '''
                obs = np.array(self._get_observation_state())
        
                if self.training:
                    self.obs_rms.update(obs)
        
                return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + epsilon)
        
            def _get_observation_state(self):
                # [f1, Accuracy, CE_len, is trivial (T), is satisyiable (!⊥)]
        
                self.f1 = self._get_f1(self.expression)
                accuracy = evaluate_concept(
                    self.kb, self.expression, Accuracy(), self.lp.encode_kb(self.kb)
                ).q
                ce_length = concept_len(self.expression)
                is_satisyiable = 0 if self.expression == self.kb.generator.nothing else 1
                is_trivial = 1 if self.expression == self.kb.generator.thing else 0
        
                return [self.f1, accuracy, ce_length, is_satisyiable, is_trivial]
        
            def _get_props(self, actions):
                a_0 = actions['a_0']
                a_1 = actions['a_1']
                
                concept_p_1 = a_0['concept_p'] 
                properties_p_1 = a_0['properties_p'] 
                operators_p_1 = a_0['operators_p'] 
                stopping_p_1 = a_0['stopping_p'] 
        
                concept_p_2 = a_1['concept_p'] 
                properties_p_2 = a_1['properties_p'] 
                operators_p_2 = a_1['operators_p'] 
                stopping_p_2 = a_1['stopping_p'] 
                
                return concept_p_1, properties_p_1, operators_p_1, stopping_p_1, concept_p_2, properties_p_2, operators_p_2, stopping_p_2

            # def fuse_predictions(self, probs_1, probs_2):
            #     m1 = np.mean(probs_1)
            #     m2 = np.mean(probs_2)
            #     combined_m = np.mean([m1, m2])
            #     return int(np.max([0, np.round(combined_m)]))
            def fuse_predictions(self, probs_1, probs_2, deterministic=True):
                """
                Combine two agents' continuous action outputs for one action type.
                - probs_1, probs_2: numpy arrays of shape (dim,) representing agent outputs for that action type.
                - If outputs are raw means from Normal, convert to logits/probabilities with softmax.
                - Return index (int) selected for discrete choices (e.g. concept index).
                """
                # convert to logits -> probabilities (softmax). add small constant for numerical stability
                logits1 = np.array(probs_1).astype(np.float64)
                logits2 = np.array(probs_2).astype(np.float64)
                # softmax helper
                def softmax(x):
                    x = x - np.max(x)
                    ex = np.exp(x)
                    return ex / (ex.sum() + 1e-12)
                p1 = softmax(logits1)
                p2 = softmax(logits2)
            
                # Combine: mean of probabilities
                p_comb = 0.5 * (p1 + p2)
            
                if deterministic:
                    return int(np.argmax(p_comb))  
                else:
                    return int(np.random.choice(np.arange(len(p_comb)), p=p_comb))


            # def _get_best_action(self, actions):
            #     (concept_p_1, properties_p_1,
            #      operators_p_1, stopping_p_1,
            #      concept_p_2, properties_p_2, 
            #      operators_p_2, stopping_p_2) = self._get_props(actions)
                
            #     best_c = self.fuse_predictions(concept_p_1, concept_p_2)
            #     best_p = self.fuse_predictions(properties_p_1, properties_p_2)
            #     best_ob = self.fuse_predictions(operators_p_1, operators_p_2)
            #     best_s = self.fuse_predictions(stopping_p_1, stopping_p_2)
            #     return best_c, best_p, best_ob, best_s
            def _get_best_action(self, actions):
                (concept_p_1, properties_p_1,
                 operators_p_1, stopping_p_1,
                 concept_p_2, properties_p_2, 
                 operators_p_2, stopping_p_2) = self._get_props(actions)
            
                best_c = self.fuse_predictions(concept_p_1, concept_p_2, deterministic=True)
                best_p = self.fuse_predictions(properties_p_1, properties_p_2, deterministic=True)
                best_ob = self.fuse_predictions(operators_p_1, operators_p_2, deterministic=True)
                # stopping_p is usually a small vector of length 3 — treat as categorical index
                best_s = self.fuse_predictions(np.ravel(stopping_p_1), np.ravel(stopping_p_2), deterministic=True)
                return best_c, best_p, best_ob, best_s
                
            def _get_expression(self, best_concept, best_prop, best_operator, split):
                temp = self.expression
                if (split < 2):
              # union, # ((ce, ce))
                    if best_operator == 0:
                        temp = self.mappings['operations'][0](
                            [temp, self.mappings['concepts'][best_concept]]
                        )
                    # intersection, # ((ce, ce))
                    elif best_operator == 1:
                        temp = self.mappings['operations'][1](
                            [temp, self.mappings['concepts'][best_concept]]
                        )
                    # negation, # (ce)
                    elif best_operator == 2:
                        temp = self.mappings['operations'][2](temp)
                    elif best_operator == 3:
                        # existential_restriction, # (ce, op)
                        temp = self.mappings['operations'][3](
                            temp, self.mappings['properties'][best_prop]
                        )
                    # universal_restriction (ce, op)
                    elif best_operator == 4:
                        temp = self.mappings['operations'][4](
                            temp, self.mappings['properties'][best_prop]
                        )
                # # split the CE 
                else:
                    try:
                        exps = list(temp.operands())
                        if len(exps) == 2:
                            e1, e2 = exps[0], exps[1]
                            # union, # ((ce, ce))
                            if best_operator == 0:
                                temp = self.mappings['operations'][best_operator](
                                    [e1, e2]
                                    )
                            # intersection, # ((ce, ce))
                            elif best_operator == 1:
                                temp = self.mappings['operations'][1](
                                    [e1, e2]
                                )
                            # negation
                            elif best_operator == 2:
                                temp = self.cg.negation_from_iterables([e1, e2])
                                temp = next(temp)
                                
                            elif best_operator == 3:
                                # existential_restriction, # (ce, op)
                                temp = self.mappings['operations'][best_operator](
                                    temp, self.mappings['properties'][best_prop]
                                )
                            # universal_restriction (ce, op)
                            elif best_operator == 4:
                                temp = self.mappings['operations'][best_operator](
                                    temp, self.mappings['properties'][best_prop]
                                )
                    except:
                        if best_operator == 0:
                            temp = self.mappings['operations'][0](
                                [temp, self.mappings['concepts'][best_concept]]
                            )
                        # intersection, # ((ce, ce))
                        elif best_operator == 1:
                            temp = self.mappings['operations'][1](
                                [temp, self.mappings['concepts'][best_concept]]
                            )
                        # negation, # (ce)
                        elif best_operator == 2:
                            temp = self.mappings['operations'][2](temp)
                        elif best_operator == 3:
                            # existential_restriction, # (ce, op)
                            temp = self.mappings['operations'][3](
                                temp, self.mappings['properties'][best_prop]
                            )
                        # universal_restriction (ce, op)
                        elif best_operator == 4:
                            temp = self.mappings['operations'][4](
                                temp, self.mappings['properties'][best_prop]
                            )
                return temp
        
            def crra_utility_function(self, alpha, reward):
                """
                Return a CRRA utility function parametrized by alpha 
                This function is inspired from: https://python-advanced.quantecon.org/five_preferences.html
                """
                c = reward + 1e-8
                if alpha == 1.:
                    return np.log(c)
                else: 
                    return c ** (1 - alpha) / (1 - alpha)
        
        
            def step(self, actions):
                info = {}
                self.time_stamp += 1
        
                best_concept, best_prop, best_operator, stopping = (
                    self._get_best_action(actions)
                )
        
                self.expression = self._get_expression(
                    best_concept, best_prop, best_operator, stopping
                )
                
        
                self.f1 = self._get_f1(self.expression)
                cost = (1 / concept_len(self.expression))
        
        
                terminations = {a: False for a in self.agents}
                truncations = {a: False for a in self.agents}
        
                info['f1'] = self.f1
                info['CE_len'] = concept_len(self.expression)
                infos = {a: info.copy() for a in self.agents}
                infos['episode'] = self.episode
        
        
                observations = {a: (self.normalized_obs()) for a in self.agents}
                step_r = self.f1 #* (cost)
                # step_r = self.crra_utility_function(1, step_r)
                rewards = {a: step_r for a in self.agents}
                observations_ = observations
                terminations_ = terminations
                truncations_ = truncations 
                infos_ = infos
        
                if any(terminations.values()) or all(truncations.values()):
                    self.agents = []
                    self.episode += 1
                    infos = {a: {} for a in self.agents}
                if (self.time_stamp == self.max_action) or (stopping == 1):
                    terminations = {'a_0': True, 'a_1': True}
                    truncations = {'a_0': True, 'a_1': True}
                    self.reset()
                else:
                    terminations = {'a_0': False, 'a_1': False}
                    truncations = {'a_0': False, 'a_1': False}
                    # self.reset()
                rewards_ = {'a_0': step_r, 'a_1': step_r}
                observations_ = {'a_0': self.normalized_obs(), 'a_1': self.normalized_obs()}
                infos_ = infos
                return (
                    observations_,
                    rewards_,
                    terminations_,
                    truncations_,
                    infos_
                )
        
            def _init_expression(self, ind):
                try:
                    candidate_exp = []
                    direct_parent_conc = set()
                    for i in self.kb.concepts:
                        conc_list = list(self.kb.reasoner.instances(i))
                        if ind in conc_list:
                            direct_parent_conc.add(i)
                    direct_parent_obj_prop = set(
                        self.kb.get_object_properties_for_ind(ind)
                    )
            
                    if len(direct_parent_conc) > 1:
                        exp = self.kb.generator.union([x for x in direct_parent_conc])
                        for i in direct_parent_obj_prop:
                            candidate_exp.append(
                                self.kb.generator.existential_restriction(exp, i)
                            )
                            candidate_exp.append(
                                self.kb.generator.universal_restriction(exp, i)
                            )
                    else:
                        if len(direct_parent_obj_prop) >1:
                            exp = self.kb.generator.existential_restriction(
                                [x for x in direct_parent_conc][0],
                                direct_parent_obj_prop.pop(),
                            )
                        else:
                            pass
            
                    for i in direct_parent_obj_prop:
                        candidate_exp.append(
                            self.kb.generator.existential_restriction(exp, i)
                        )
                        candidate_exp.append(
                            self.kb.generator.universal_restriction(exp, i)
                        )
                    if len(self.self.stepin_cand) < 10:
                        self.stepin_cand.append([x for x in candidate_exp])
                        if len(self.stepin_cand) > 1:
                            self.stepin_cand.pop()
                    best_ce_score = 0
                    best_ce = candidate_exp[0]
                    for i in candidate_exp:
                        f1 = evaluate_concept(
                            self.kb, i, F1(), self.lp.encode_kb(self.kb)
                        ).q
                        if f1 > best_ce_score:
                            best_ce_score = f1
                            best_ce = i
                    rand_prop = np.random.random()
                    if rand_prop >= 0.5:
                        return best_ce
                    else:
                        return np.random.choice(candidate_exp)
                except:
                    return None
            def fill_stepin_cand(self, pos_list):
                stepin_cand = []
                for ind in pos_list:
                    for i in self.kb.concepts:
                        conc_list = list(self.kb.reasoner.instances(i))
                        if ind in conc_list:
                            stepin_cand.append(i)
                return stepin_cand
        
            def render(self):
                print(self.expression.str)
                return
        
            def export_expr(self, file_name, value):
                with open(f"{file_name}.txt", "a", encoding="utf-8") as myfile:
                    myfile.write(value + "\n\n")
        
            @functools.lru_cache(maxsize=None)
            def observation_space(self, agent):
                return spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32
                )
        
            @functools.lru_cache(maxsize=None)
            def action_space(self, agent):
                return spaces.Dict(
                    {                
                        "concept_p": spaces.Box(0, 1, (len(self.concepts),)),
                        "properties_p": spaces.Box(0, 1, (len(self.properties),)),
                        "operators_p": spaces.Box(0, 1, (len(self.operations),)),
                        "stopping_p": spaces.Box(0, 1, (1, 3)),
                    }
                )
        
        def get_action_logits(actions_mb, c_dim, p_dim, ob_dim):
            a1_concept_prop = actions_mb[:,0: c_dim]
            a1_properties_action_prop = actions_mb[:,c_dim:c_dim+p_dim]
            a1_operators_action_prop = actions_mb[:,c_dim+p_dim:c_dim+p_dim+ob_dim]
            a1_stopping_action_prop = actions_mb[:,c_dim+p_dim+ob_dim:c_dim+p_dim+ob_dim+3]
            a2_dim = c_dim+p_dim+ob_dim+3
            a2_concept_prop = actions_mb[:,a2_dim:a2_dim +c_dim]
            a2_properties_action_prop = actions_mb[:,a2_dim +c_dim:a2_dim +c_dim +p_dim]
            a2_operators_action_prop = actions_mb[:,a2_dim +c_dim +p_dim:a2_dim +c_dim +p_dim + ob_dim]
            a2_stopping_action_prop = actions_mb[:,a2_dim +c_dim +p_dim + ob_dim:a2_dim +c_dim +p_dim + ob_dim+3]
            
            return (a1_concept_prop
                    ,a1_properties_action_prop
                    ,a1_operators_action_prop
                    ,a1_stopping_action_prop
                    ,a2_concept_prop
                    ,a2_properties_action_prop
                    ,a2_operators_action_prop
                    ,a2_stopping_action_prop)
            
        def get_actions_props_entropy_value(agents, next_obs, actions_mb = None):
            if next_obs.dim() == 1:
                next_obs = next_obs.unsqueeze(0)
            (a1_concept, a1_properties, a1_operators, 
            a1_stopping, a2_concept, a2_properties,
            a2_operators, a2_stopping,
            a1_concept_log_prob,
            a1_properties_log_prob,
            a1_operators_log_prob,
            a1_stopping_log_prob,
            a1_concept_entropy,
            a1_properties_entropy,
            a1_operators_entropy,
            a1_stopping_entropy,
            a2_concept_log_prob,
            a2_properties_log_prob,
            a2_operators_log_prob,
            a2_stopping_log_prob,
            a2_concept_entropy,
            a2_properties_entropy,
            a2_operators_entropy,
            a2_stopping_entropy, V, hidden) = agents.get_action_and_value(next_obs, actions_mb)
            
            actions_dict = {
                'a_0': {"concept_p": a1_concept.cpu().numpy(),
                        "properties_p": a1_properties.cpu().numpy(),
                        "operators_p": a1_operators.cpu().numpy(),
                        "stopping_p": a1_stopping.cpu().numpy()},
                        
                'a_1': {"concept_p": a2_concept.cpu().numpy(),
                        "properties_p": a2_properties.cpu().numpy(),
                        "operators_p": a2_operators.cpu().numpy(),
                        "stopping_p": a2_stopping.cpu().numpy()}
          
            }
            log_prop = (a1_concept_log_prob 
                        +a1_properties_log_prob 
                        +a1_operators_log_prob 
                        +a1_stopping_log_prob
                        +a2_concept_log_prob
                        +a2_properties_log_prob
                        +a2_operators_log_prob
                        +a2_stopping_log_prob)
        
            entropy = (a1_concept_entropy 
                       +a1_properties_entropy 
                       +a1_operators_entropy 
                       +a1_stopping_entropy
                       +a2_concept_entropy
                       +a2_properties_entropy
                       +a2_operators_entropy
                       +a2_stopping_entropy)
            
            if a1_properties_log_prob.nelement() == 1:
                actions_tensor = torch.cat((a1_concept, a1_properties, a1_operators, 
                                            a1_stopping, a2_concept, a2_properties,
                                            a2_operators, a2_stopping), dim = -1)
            else:        
                actions_tensor = torch.cat((a1_concept, a1_properties, a1_operators, 
                                            a1_stopping, a2_concept, a2_properties,
                                            a2_operators, a2_stopping), dim = -1)
                
            
            return actions_dict, log_prop, entropy, V, hidden, actions_tensor
        
        nllll = []
        agentlll = []
        rnn = []
        rne = []
        pgllll= []
        elll = []
        
        # kb = KnowledgeBase(path='../Downloads/sml_new/muta/mutagenesis_miss.owl')
        
        
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument("--exp-name", type=str, default=os.path.basename('').rstrip(".py"))
            parser.add_argument("--seed", type=int, default=seed_)
            parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
            parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
            parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
            parser.add_argument("--wandb-project-name", type=str, default="cleanRL")
            parser.add_argument("--wandb-entity", type=str, default=None)
            parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
        
            # PPO args
            parser.add_argument("--total-timesteps", type=int, default=10000)  # small for testing
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
        
        
        def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        
        
        def to_torch_obs(x, device):
            return torch.tensor(x, dtype=torch.float32, device=device)
        
        
        def logmeanexp(x, dim=-1):
            # stable log(mean(exp(x))) = logsumexp(x) - log(n)
            lse = torch.logsumexp(x, dim=dim)
            n = x.shape[dim]
            return lse - math.log(n)
        
        
        class Agents(nn.Module):
            def __init__(self, env):
                super().__init__()
                self.network = nn.Sequential(
                    layer_init(nn.Linear(5, 64)),
                    nn.ReLU(),
                    layer_init(nn.Linear(64, 64)),
                    nn.ReLU(),
                    layer_init(nn.Linear(64, 64)),
                    nn.ReLU(),
                )
                self.env = env
                self.n_concepts = len(env.concepts)
                self.n_prop = len(env.properties)
                self.n_op = len(env.operations)
                
                self.critic = layer_init(nn.Linear(64, 1), std=1)
        
                
                self.a1_concept_mean, self.a1_concept_logstd  = self.continous_actor(self.n_concepts)
                self.a1_properties_mean, self.a1_properties_logstd  = self.continous_actor(self.n_prop)
                self.a1_operators_mean, self.a1_operators_logstd  = self.continous_actor(self.n_op)
                self.a1_stopping_mean, self.a1_stopping_logstd  = self.continous_actor(3)
        
                self.a2_concept_mean, self.a2_concept_logstd = self.continous_actor(self.n_concepts)
                self.a2_properties_mean, self.a2_properties_logstd = self.continous_actor(self.n_prop)
                self.a2_operators_mean, self.a2_operators_logstd = self.continous_actor(self.n_op)
                self.a2_stopping_mean, self.a2_stopping_logstd = self.continous_actor(3)
            
            def continous_actor(self, dim_):
                actor_mean = nn.Sequential(
                    layer_init(nn.Linear(5, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, 64)),
                    nn.Tanh(),
                    layer_init(nn.Linear(64, dim_), std=0.01),
                )
                actor_logstd = nn.Parameter(torch.zeros(1, dim_))
                
                return actor_mean, actor_logstd
                
            def get_value(self, x):
                x = x.clone()
                return self.critic(self.network(x))
        
        
            def get_action_and_value(self, x, actions_mini_batch = None):
                if actions_mini_batch is not None:
                    (a1_concept
                    ,a1_properties
                    ,a1_operators
                    ,a1_stopping
                    ,a2_concept
                    ,a2_properties
                    ,a2_operators
                    ,a2_stopping) = get_action_logits(actions_mini_batch, self.n_concepts, self.n_prop, self.n_op)
            
                '''
                Looks very ugly... sorry :) TODO: enhance it
                '''
                ##########################
                ## Agent 1
                ##########################        
                # first things first... Normal actions!  
                
                # Extremely ugly but I'm tired... working in 2 papers in parallel and have been working for 17 hours... no tears yet :(
                # Revisting the function again 20 days before submession... PhD is like marriage... a ligal statment for what you are doing!
                # I love what I'm doing!
        
                hidden = self.network(x) #--> 64
                ###### Concept A1 ########
                a1_concept_mean = self.a1_concept_mean(x)
                a1_concept_logstd = self.a1_concept_logstd.expand_as(a1_concept_mean)
                a1_concept_std = torch.exp(a1_concept_logstd)
                a1_concept_probs = Normal(a1_concept_mean, a1_concept_std)
                if actions_mini_batch is None:
                    a1_concept = a1_concept_probs.sample()
        
        
                a1_properties_mean = self.a1_properties_mean(x)
                a1_properties_logstd = self.a1_properties_logstd.expand_as(a1_properties_mean)
                a1_properties_std = torch.exp(a1_properties_logstd)
                a1_properties_props = Normal(a1_properties_mean, a1_properties_std)
                if actions_mini_batch is None:
                    a1_properties = a1_properties_props.sample()
        
        
                a1_operators_mean = self.a1_operators_mean(x)
                a1_operators_logstd = self.a1_operators_logstd.expand_as(a1_operators_mean)
                a1_operators_std = torch.exp(a1_operators_logstd)
                a1_operators_props = Normal(a1_operators_mean, a1_operators_std)
                if actions_mini_batch is None:
                    a1_operators = a1_operators_props.sample()
        
                a1_stopping_mean = self.a1_stopping_mean(x)
                a1_stopping_logstd = self.a1_stopping_logstd.expand_as(a1_stopping_mean)
                a1_stopping_std = torch.exp(a1_stopping_logstd)
                a1_stopping_props = Normal(a1_stopping_mean, a1_stopping_std)
                if actions_mini_batch is None:
                    a1_stopping = a1_stopping_props.sample()
                
                ##########################
                ## Agent 2
                ##########################
                a2_concept_mean = self.a2_concept_mean(x)
                a2_concept_logstd = self.a2_concept_logstd.expand_as(a2_concept_mean)
                a2_concept_std = torch.exp(a2_concept_logstd)
                a2_concept_probs = Normal(a2_concept_mean, a2_concept_std)
                if actions_mini_batch is None:
                    a2_concept = a2_concept_probs.sample()
        
        
                a2_properties_mean = self.a2_properties_mean(x)
                a2_properties_logstd = self.a2_properties_logstd.expand_as(a2_properties_mean)
                a2_properties_std = torch.exp(a2_properties_logstd)
                a2_properties_props = Normal(a2_properties_mean, a2_properties_std)
                if actions_mini_batch is None:
                    a2_properties = a2_properties_props.sample()
        
        
                a2_operators_mean = self.a2_operators_mean(x)
                a2_operators_logstd = self.a2_operators_logstd.expand_as(a2_operators_mean)
                a2_operators_std = torch.exp(a2_operators_logstd)
                a2_operators_props = Normal(a2_operators_mean, a2_operators_std)
                if actions_mini_batch is None:
                    a2_operators = a2_operators_props.sample()
        
                a2_stopping_mean = self.a2_stopping_mean(x)
                a2_stopping_logstd = self.a2_stopping_logstd.expand_as(a2_stopping_mean)
                a2_stopping_std = torch.exp(a2_stopping_logstd)
                a2_stopping_props = Normal(a2_stopping_mean, a2_stopping_std)
                if actions_mini_batch is None:
                    a2_stopping = a2_stopping_props.sample()
                    
                return (a1_concept, a1_properties, a1_operators, 
                        a1_stopping, a2_concept, a2_properties,
                        a2_operators, a2_stopping,
                       a1_concept_probs.log_prob(a1_concept).sum(1),
                       a1_properties_props.log_prob(a1_properties).sum(1),
                       a1_operators_props.log_prob(a1_operators).sum(1),
                       a1_stopping_props.log_prob(a1_stopping).sum(1),
                       a1_concept_probs.entropy().sum(1),
                       a1_properties_props.entropy().sum(1),
                       a1_operators_props.entropy().sum(1),
                       a1_stopping_props.entropy().sum(1),
                       a2_concept_probs.log_prob(a2_concept).sum(1),
                       a2_properties_props.log_prob(a2_properties).sum(1),
                       a2_operators_props.log_prob(a2_operators).sum(1),
                       a2_stopping_props.log_prob(a2_stopping).sum(1),
                       a2_concept_probs.entropy().sum(1),
                       a2_properties_props.entropy().sum(1),
                       a2_operators_props.entropy().sum(1),
                       a2_stopping_props.entropy().sum(1), self.critic(hidden), hidden)
        
        class DynamicsModel(nn.Module):
            """
            Predict next-hidden mean & logvar given (hidden, action one-hot).
            Diagonal Gaussian output.
            """
            def __init__(self, env, hidden_dim=64, hidden_size=64):
                super().__init__()
                
                self.n_actions = (len(env.concepts) + len(env.properties) + len(env.operations) + 3)*2
                self.net = nn.Sequential(
                    nn.Linear(hidden_dim + self.n_actions , hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                )
        
                self.mean_head = nn.Linear(hidden_size, hidden_dim)
                self.logvar_head = nn.Linear(hidden_size, hidden_dim)
        
            def forward(self, z, a):
                x = torch.cat([z.to('cuda'), a.to('cuda')], dim=-1)
                h = self.net(x)
                mean = self.mean_head(h)
                logvar = torch.clamp(self.logvar_head(h), -10.0, 1.0)
                return mean, logvar
        
            def sample(self, z, a, K):
                mean, logvar = self.forward(z, a)
                std = (0.5 * logvar).exp()
                B, D = mean.shape
                mean_k = mean.unsqueeze(1).expand(-1, K, -1)
                std_k = std.unsqueeze(1).expand(-1, K, -1)
                eps = torch.randn_like(std_k)
                samples = mean_k + std_k * eps  # (B, K, D)
                return samples
        
        
        args = parse_args()
        run_name = 'CEL_TMP3O'
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic
        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
        # envs = ALC_Env(kb, pos, neg, 'f1', 10)
        envs = ALC_Env(kb, pos, neg, 'f1', 10)
        agents = Agents(envs).to(device)
        dyn = DynamicsModel(env=envs, hidden_dim=64).to(device)
        optimizer = optim.Adam(agents.parameters(), lr=args.learning_rate, eps=1e-5)
        optimizer_dyn = optim.Adam(dyn.parameters(), lr=args.dyn_lr, eps=1e-5)
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space('a_0').shape, dtype=torch.float32, device=device) #TODO: envs.single_observation_space.shape
        actions = torch.zeros((args.num_steps, args.num_envs) + (dyn.n_actions,), dtype=torch.float32, device=device) #TODO: envs.single_action_space.shape
        logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        terminations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        truncations = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
        # hidden buffers for dynamics training
        
        hiddens = torch.zeros((args.num_steps, args.num_envs, 64), dtype=torch.float32, device=device)
        next_hiddens = torch.zeros((args.num_steps, args.num_envs, 64), dtype=torch.float32, device=device)
        reset_out = envs.reset(seed=args.seed)
        next_obs = to_torch_obs(reset_out[0]['a_0'], device)  # shape (num_envs, H, W, C)
        next_termination = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        next_truncation = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
        global_step = 0
        start_time = time.time()
        num_updates = args.total_timesteps // args.batch_size
        
            
        for update in range(1, int(num_updates) + 1):
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
                    actions_dict, logprob, _, value, hidden, conc_actions = get_actions_props_entropy_value(agents, next_obs)
                    
                    values[step] = value.reshape(-1)
                    hiddens[step] = hidden  # store hidden for dynamics training
        
                actions[step] = torch.tensor(conc_actions, dtype = torch.float32) #actions for the rollout
                logprobs[step] = logprob
                step_out = envs.step(actions_dict)  
              #  envs.export_expr(rf'/results_c/{kb_name}_{kb_prct}_{seed_}', f'{str(owl_expression_to_dl(envs.expression)) + ' : ' + str(envs.f1)}')
                envs.export_expr(rf'/upb/users/a/amgad/profiles/unix/cs/results_c/{kb_name}_{kb_prct}_{seed_}', f'{str(owl_expression_to_dl(envs.expression)) + " : " + str(envs.f1)}')
                
                next_obs_raw, reward_raw, term_raw, trunc_raw, info = step_out            
        
                next_obs = to_torch_obs(envs.normalized_obs(), device)
                rewards[step] = torch.tensor(reward_raw['a_0'], dtype=torch.float32, device=device)
                next_termination = torch.tensor(term_raw['a_0'], dtype=torch.float32, device=device)
                next_truncation = torch.tensor(trunc_raw['a_0'], dtype=torch.float32, device=device)
        
                # compute and store next_hidden for dynamics training
                with torch.no_grad():
                    next_hidden = agents.network(next_obs)  # (num_envs, 512)
                    next_hiddens[step] = next_hidden
             # ---------- Train dynamics model (on collected hidden pairs) ----------
            # Flatten (T * N, D)
            flat_z = hiddens.view(-1, 64)  # tensor
            flat_next_z = next_hiddens.view(-1, 64)
            flat_actions = actions.view(-1, dyn.n_actions) # shape (batch_size,)
            dataset_size = flat_z.shape[0]
            if dataset_size > 0:
                batch_dyn = 256
                for epoch in range(args.dyn_epochs):
                    perm = torch.randperm(dataset_size, device=device)
                    for start in range(0, dataset_size, batch_dyn):
                        idx = perm[start : start + batch_dyn]
                        z_b = flat_z[idx]
                        znext_b = flat_next_z[idx]
                        act_b = flat_actions[idx]
                        mean, logvar = dyn(z_b, act_b)
                        var = torch.exp(logvar)
                        # Gaussian NLL (mean + logvar)
                        nll = 0.5 * (((znext_b - mean) ** 2) / var + logvar).mean()
                        optimizer_dyn.zero_grad()
                        nllll.append(nll)
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
                b_hiddens = flat_z 
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
                    # handle nonterminal and terminal separately: T = r for terminal transitions
                    nonterm_idx = np.where(done_batch == 0)[0]
                    if len(nonterm_idx) == 0:
                        T_flat[start:end] = r_batch
                        continue
                    # select nonterminal subset
                    z_nonterm = z_batch[nonterm_idx]  # (bn,512)
                    a_nonterm = torch.tensor(a_batch[nonterm_idx], dtype=torch.float32)
                    r_nonterm = torch.tensor(r_batch[nonterm_idx], dtype=torch.float32, device=device)
        
                    # sample K next-hidden candidates from dynamics model
                    samples = dyn.sample(z_nonterm, a_nonterm, K=K)  # (bn, K, 512)
                    bn = samples.shape[0]
                    samples_flat = samples.view(bn * K, 64)  # (bn*K, 512)
                    # evaluate critic on samples (treat samples as hidden latents)
                    V_flat = agents.critic(samples_flat).view(bn, K)  # (bn, K)
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
        
                lastgaelam = 0.0
                # next_done for final step uses next_termination/next_truncation per env from last step
                next_done_env = float(max(next_termination.item(), next_truncation.item()))
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done_env
                    else:
                        nextnonterminal = 1.0 - max(terminations[t + 1].item(), truncations[t + 1].item())
                    lastgaelam = deltas[t] + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + V_mat
            
            # flatten for PPO update
            b_obs = obs.reshape((-1,) + (5,))  # 5: obs_dim
            b_logprobs = logprobs.reshape(-1).cpu()
            b_actions = actions.reshape((-1, dyn.n_actions))
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
                    _, newlogprob, entropy, newvalue, _, _ = get_actions_props_entropy_value(agents, obs_mb, actions_mb)
                    rnn.append(b_returns[mb_inds].mean().item())
                    
                    actions_dict, logprob, _, value, hidden, _ = get_actions_props_entropy_value(agents, next_obs)
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
                    pgllll.append(pg_loss.item())
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
                    elll.append(entropy_loss.item())
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef                    
                    optimizer.zero_grad()
                    loss.backward()
                    agentlll.append(loss.item())
                    
                    nn.utils.clip_grad_norm_(agents.parameters(), args.max_grad_norm)
                    optimizer.step()
                if args.target_kl is not None:
                    if approx_kl > args.target_kl:
                        break
            # diagnostics
            y_pred = b_values.cpu().numpy()
            y_true = b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        print(f'============================= Done with: {kb_name} === seed: {seed_} =====================================')
        df = pd.DataFrame({'rewards':rnn})
        df.to_csv(f'/upb/users/a/amgad/profiles/unix/cs/results_f/TMP2O_{kb_name}_{kb_prct}_{seed_}.csv')
        # series = [pgllll, elll, agentlll, rnn, rne]
        # series = [rnn]
        # labels  = ['pgllll', 'elll', 'agentlll', 'rnn', 'rne']
        # labels  = [f'Rewards (Not Averaged)_{kb_name}_seed: {seed_}']
        # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12,8))
        # axes = axes.flatten()
        
        # for ax, s, lbl in zip(axes, series, labels):
        #     ax.plot(s)
        #     ax.grid(True)
        
        # for ax in axes[len(series):]:
        #     ax.set_visible(False)
        
        # plt.tight_layout()
        del agents
        del optimizer
        del optimizer_dyn
        # plt.show()
        envs.close()
