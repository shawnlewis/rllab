import os
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
os.environ["THEANO_FLAGS"] = "device=cpu"

from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.normalized_env import normalize

from sandbox.vime.algos.trpo_expl import TRPO
from rllab.misc.instrument import stub, run_experiment_lite

from osim_env import OsimEnv

stub(globals())

env = normalize(OsimEnv('OsimRun'))

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    hidden_sizes=(64, 64),
)

baseline = LinearFeatureBaseline(
    env.spec,
)

batch_size = 1000
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=batch_size,
    whole_paths=True,
    max_path_length=500,
    n_itr=10000,
    step_size=0.01,
    eta=0.001,
    snn_n_samples=10,
    subsample_factor=1.0,
    use_replay_pool=True,
    use_kl_ratio=True,
    use_kl_ratio_q=True,
    n_itr_update=1,
    kl_batch_size=1,
    normalize_reward=False,
    replay_pool_size=1000000,
    n_updates_per_sample=5000,
    second_order_update=True,
    unn_n_hidden=[32],
    unn_layers_type=[1, 1],
    unn_learning_rate=0.0001
)

run_experiment_lite(
    algo.train(),
    exp_prefix="trpo-expl",
    n_parallel=1,
    snapshot_mode="all",
    seed=1,
    mode="local",
    #resume_from='data/local/trpo-expl/trpo-expl_2017_10_03_17_01_22_0001/params.pkl',
    script="vime_run_experiment_lite.py",
)
