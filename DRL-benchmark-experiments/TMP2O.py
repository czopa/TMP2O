from ontolearn.triple_store import TripleStore
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
from ontolearn.triple_store import TripleStore
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
from tqdm import tqdm
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt