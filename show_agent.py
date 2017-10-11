import joblib
import numpy as np
import sys

from rllab.envs.normalized_env import normalize
from rllab.sampler.utils import rollout
import osim_env

def main(argv):
    print('Creating env')
    env = normalize(osim_env.OsimEnv('OsimRun', visualize=True))
    print('Restoring data')
    restored = joblib.load(argv[1])
    print('Running rollout')
    path = rollout(env, restored['policy'], 500)
    print('Total Reward: %s' % np.sum(path['rewards']))

if __name__ == '__main__':
    main(sys.argv)