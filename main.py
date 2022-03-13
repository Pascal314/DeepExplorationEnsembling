import jax
import optax
import haiku as hk
# Make sure to 
# export XLA_PYTHON_CLIENT_PREALLOCATE=false or something similar so that this main process does not eat all GPU memory
from bsuite.environments import catch, deep_sea
from replaybuffer import ReplayBuffer
from learner import Learner
from models import fSVGDEnsemble
from nets import DeepSeaNet
import ray
from actor import Actor, DQNAgent
import util
from parameter_server import ParameterServer
import time
import numpy as np
import logging
import pickle

replay_buffer_size = 10_000
short_term_capacity = 32
batch_size = 32
learning_rate = 1e-2
random_seed = 42
lambda_ = 0.5
discount_factor = 0.99
n_networks = 100
n_actors = 10
N = 30 # The environment size
unroll_length = N
total_episodes = 200_000
total_steps = total_episodes * unroll_length
rollouts_per_actor = total_steps // (unroll_length * n_actors)

#batch size and n_actors can be used to trade off between actor and learner speed.

def main():
    ray.init()
    build_env = catch.Catch
    build_env = lambda: deep_sea.DeepSea(N, seed=42, mapping_seed=42)
    env_for_spec = build_env()
    timestep = env_for_spec.reset()

    timestep = util.preprocess_step(timestep)
    timestep = jax.tree_map(lambda x: x.reshape(1, *x.shape), timestep)
    # print(timestep)
    num_actions = env_for_spec.action_spec().num_values

    replaybuffer = ReplayBuffer.remote(replay_buffer_size, short_term_capacity)
    parameter_server = ParameterServer.remote()

    def forward(x):
        out = DeepSeaNet(num_actions)(x)
        prior = jax.lax.stop_gradient(DeepSeaNet(num_actions, name='prior')(x))
        return out + prior

    net = hk.without_apply_rng(hk.transform(forward))

    logger = util.TempLogger()

    model = fSVGDEnsemble(
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
        parameter_server=parameter_server
    )

    agent = DQNAgent(net.apply)
    actors = [Actor.remote(
        agent=agent,
        env_builder=build_env,
        learner=learner,
        unroll_length=unroll_length,
        memory_buffer=replaybuffer,
        n_networks=n_networks,
        rng_seed=i,
        logger=logger,
        convert_params=model.convert_params,
        parameter_server=parameter_server
    ) for i in range(n_actors)]

    learner.run.remote()
    # this is stupid
    while not ray.get(parameter_server.get_params_set.remote()):
        print(time.sleep(0.5))
        print('Waiting for learner..')
    start_params = ray.get(parameter_server.get_params.remote())
    # osband2018 learns up until N=60 in 100k episodes = 100k * N frames
    ray.get([actor.run.remote(rollouts_per_actor) for actor in actors])
    end_params = ray.get(parameter_server.get_params.remote())

    with open(f'params/params.pkl', 'wb') as outfile:
        pickle.dump(end_params, outfile)
    ray.shutdown()

main()