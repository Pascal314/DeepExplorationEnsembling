import jax
import optax
import haiku as hk
# Make sure to 
# export XLA_PYTHON_CLIENT_PREALLOCATE=false or something similar so that this main process does not eat all GPU memory
from bsuite.environments import catch, deep_sea
from replaybuffer import ReplayBuffer
from learner import Learner
from models import fSVGDEnsemble, PlainEnsemble
from nets import DeepSeaNet
import ray
from actor import RemoteActor, DQNAgent, EvalActor
import util
from parameter_server import ParameterServer
import time
import numpy as np
import logging
import pickle
from sacred import Experiment
from sacred.observers import FileStorageObserver
import visualisation
import os
from collections import deque
import matplotlib.pyplot as plt

ex = Experiment(name='deep_exploration')
ex.observers.append(FileStorageObserver('experiments'))

@ex.config
def config():
    replay_buffer_size = 10_00
    short_term_capacity = 32
    batch_size = 32
    learning_rate = 3e-4
    random_seed = 42
    lambda_ = 0.5
    discount_factor = 0.99
    n_networks = 20
    n_actors = 5
    N = 60 # The environment size
    unroll_length = N
    total_episodes = 100_000
    total_steps = total_episodes * unroll_length
    rollouts_per_actor = total_steps // (unroll_length * n_actors)
    use_fsvgd = False
    use_rpf = False
    bias_init = False

#batch size and n_actors can be used to trade off between actor and learner speed.

@ex.automain
def run_experiment(
        replay_buffer_size,
        short_term_capacity,
        batch_size,
        learning_rate,
        random_seed,
        lambda_,
        discount_factor,
        n_networks,
        n_actors,
        N,
        unroll_length,
        total_episodes,
        total_steps,
        rollouts_per_actor,
        use_fsvgd,
        use_rpf,
        bias_init,
        _run
    ):
    ray.init()
    # build_env = catch.Catch
    build_env = lambda: deep_sea.DeepSea(N, seed=random_seed, mapping_seed=random_seed)
    env_for_spec = build_env()
    timestep = env_for_spec.reset()

    timestep = util.preprocess_step(timestep)
    timestep = jax.tree_map(lambda x: x.reshape(1, *x.shape), timestep)
    num_actions = env_for_spec.action_spec().num_values
    action_mapping = env_for_spec._action_mapping

    replaybuffer = ReplayBuffer.remote(replay_buffer_size, short_term_capacity)
    parameter_servers = [ParameterServer.remote() for _ in range(n_actors + 1)]

    if use_rpf:
        print("Using randomized priors!")
        def forward(x):
            out = DeepSeaNet(num_actions, bias_init=bias_init)(x)
            prior = jax.lax.stop_gradient(DeepSeaNet(num_actions, name='prior')(x))
            return out + prior
    else:
        print("NOT using randomized priors!")

        def forward(x):
            out = DeepSeaNet(num_actions, bias_init=bias_init)(x)
            return out

    net = hk.without_apply_rng(hk.transform(forward))

    logger = util.TempLogger()
    if use_fsvgd:
        model = fSVGDEnsemble(
            individual_transformed=net,
            n_networks=n_networks
        )
    else:
        model = PlainEnsemble(
            individual_transformed=net,
            n_networks=n_networks
        )

    learner = Learner.remote(
        replaybuffer=replaybuffer,
        batch_size=batch_size,
        opt=optax.adam(learning_rate),
        dummy_observation=timestep,
        model=model,
        rng_key=jax.random.PRNGKey(random_seed),
        discount_factor=discount_factor,
        lambda_=lambda_,
        logger=logger,
        parameter_servers=parameter_servers
    )

    agent = DQNAgent(net.apply)
    actors = [RemoteActor.remote(
        agent=agent,
        env_builder=build_env,
        learner=learner,
        unroll_length=unroll_length,
        memory_buffer=replaybuffer,
        n_networks=n_networks,
        rng_seed=i,
        logger=logger,
        convert_params=model.convert_params,
        parameter_server=parameter_servers[i]
    ) for i in range(n_actors)]

    eval_actor = EvalActor.remote(
        agent=agent,
        env_builder=build_env,
        learner=learner,
        unroll_length=unroll_length,
        memory_buffer=replaybuffer,
        n_networks=n_networks,
        rng_seed=0,
        logger=logger,
        convert_params=model.convert_params,
        parameter_server=parameter_servers[-1]
    )

    learner.run.remote()
    # this is stupid
    while not ray.get(parameter_servers[-1].get_params_set.remote()):
        print(time.sleep(0.5))
        print('Waiting for learner..')
    # osband2018 learns up until N=60 in 100k episodes = 100k * N frames

    task_ids = [actor.run.remote(rollouts_per_actor) for actor in actors]
    done = False

    last_returns = deque([0], maxlen=5)

    while not done:
        ready, not_ready = ray.wait(task_ids, timeout=1, num_returns=n_actors)
        if len(not_ready) == 0:
            done = True
        results = ray.get(eval_actor.evaluate.remote(n_episodes=100, max_unroll_length=N+2))
        _run.log_scalar("test.reward", results['mean_return'], results['n_frames'])
        last_returns.append(results['mean_return'])


        # params = ray.get(parameter_servers[-1].get_params.remote())
        # all_states = visualisation.create_every_state_in_dummy_timestep(N)
        # Q_values = model.ensemble_transformed.apply(params, all_states)
        # fig, axes = visualisation.visualize_uncertainty(Q_values, action_mapping=action_mapping)
        # fig.savefig('temp/uncertainty.pdf')
        # _run.add_artifact('temp/uncertainty.pdf')
        # os.remove('temp/uncertainty.pdf')
        # fig.clf()
        # plt.close(fig)

        # fig, axes = visualisation.visualize_Q_values(Q_values[::n_networks // 5], action_mapping=action_mapping)
        # fig.savefig('temp/q_values.pdf')
        # _run.add_artifact('temp/q_values.pdf')
        # os.remove('temp/q_values.pdf')
        # fig.clf()
        # plt.close(fig)

        if all(map(lambda x: x > 0.95, last_returns)):
            done = True

    print("Cancelling and killing processes")
    ray.kill(learner)
    ray.kill(eval_actor)
    ray.kill(replaybuffer)
    for server in parameter_servers:
        ray.kill(server)
    
    for actor in actors:
        ray.kill(actor)

    ray.shutdown()