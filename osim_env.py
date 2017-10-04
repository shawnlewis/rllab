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



def convert_gym_space(space):
    if isinstance(space, osim.env.gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, osim.env.gym.spaces.Discrete):
        return Discrete(n=space.n)
    else:
        raise NotImplementedError


class OsimEnv(Env, Serializable):
    def __init__(self, env_name, visualize=False):
        Serializable.quick_init(self, locals())

        if env_name == 'OsimRun':
            env = RunEnv(visualize=visualize)
        else:
            raise EnvironmentError('No OsimEnv: %s' % env_name)

        self.env = env
        self.env_id = env.spec.id

        self._observation_space = convert_gym_space(env.observation_space)
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
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self):
        self.env.render()

    def terminate(self):
        pass
