from gymnasium import spaces

from envs.batched_env import ParallelBatchedEnv, SequencedBatchedEnv
from envs.evaluate import Evaluator, RecordScores
from envs.make_env import make_env
from envs.normalize import VecNormalize
from envs.state import State
from envs.transition import MCTSTransition, MuZeroTransition, Transition
from gymnax.gymnax.environments.spaces import Discrete

__all__ = [
    "ATARI_ENVS",
    "Discrete",
    "Evaluator",
    "MCTSTransition",
    "MuZeroTransition",
    "ParallelBatchedEnv",
    "RecordScores",
    "SequencedBatchedEnv",
    "State",
    "Transition",
    "VecNormalize",
    "has_discrete_action_space",
    "is_atari_env",
    "make_env",
    "spaces",
]

ATARI_ENVS = [
    'Asterix-MinAtar',
    'Breakout-MinAtar',
    'Freeway-MinAtar',
    'SpaceInvaders-MinAtar',
]

def has_discrete_action_space(env, env_params):
    return isinstance(env.action_space(env_params), Discrete)


def is_atari_env(env_id):
    return env_id in ATARI_ENVS