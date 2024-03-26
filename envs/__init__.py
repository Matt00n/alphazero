from gymnasium import spaces

from envs.batched_env import SequencedBatchedEnv, ParallelBatchedEnv 
from envs.normalize import VecNormalize
from envs.make_env import make_env
from envs.state import State
from envs.transition import Transition, MCTSTransition
from envs.evaluate import RecordScores, Evaluator
from gymnax.gymnax.environments.spaces import Discrete


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