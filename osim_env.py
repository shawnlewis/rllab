from __future__ import print_function
from __future__ import absolute_import


import osim
from osim.env import RunEnv

import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.misc import logger

import numpy as np



def convert_gym_space(space):
    if isinstance(space, osim.env.gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, osim.env.gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError

MARKOVIFY = True

VEL_PENALTY_MULT = 4

FALL_SMOOTHING = False
MAX_PENALTY = 0.015
EXPONENT = 1.5

def extra_reward(env):
    # Penalize for pelvis height getting close to 0.65.

    pelvis_y = env.current_state[env.STATE_PELVIS_Y]
    if pelvis_y > 1:
        return 0
    # Quadratically go from MAX_PENALTY at pelvis_y=0.65 (simulation ends) to
    # 0 at pelvis_y=1.0 which is the max
    # Setting EXPONENT to 1 makes this linear
    penalty = MAX_PENALTY * (-(pelvis_y - 1)/.35) ** EXPONENT
    return -penalty


class OsimEnv(Env, Serializable):
    def __init__(self, env_name, visualize=False):
        Serializable.quick_init(self, locals())

        if env_name == 'OsimRun':
            env = RunEnv(visualize=visualize)
        else:
            raise EnvironmentError('No OsimEnv: %s' % env_name)

        self.env = env
        self.env_id = env.spec.id

        if not MARKOVIFY:
            self._observation_space = convert_gym_space(env.observation_space)
        else:
            self._observation_space = Box(
                    low=np.repeat(-3.14159265, 41 + 14),
                    high=np.repeat(3.14159265, 41 + 14))
        self._action_space = convert_gym_space(env.action_space)
        self._horizon = env.spec.timestep_limit

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        # Hardcode difficult parameter for Osim's RunEnv
        obs = self.env.reset(difficulty=0)
        if MARKOVIFY:
            obs = np.concatenate((obs, np.repeat(0, 14)))
        return obs

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        if FALL_SMOOTHING:
            _extra_reward = extra_reward(self.env)
            reward += _extra_reward
        if MARKOVIFY:
            # add velocities for object positions
            velocities = np.array(self.env.current_state[-19:-5]) - np.array(self.env.last_state[-19:-5])
            vel_squared_sum = np.sum(np.square(velocities))
            reward -= VEL_PENALTY_MULT * vel_squared_sum
            #print ('Velocities: %s' % velocities)
            next_obs = np.concatenate((next_obs, velocities))
        #try:
        #    input('Press Enter')
        #except SyntaxError:
        #    pass
        #print('NEXT_OBS: %s' % next_obs)
        #import pdb; pdb.set_trace()
        print('Reward: %s' % reward)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        pass
