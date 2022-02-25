import jax
import optax
from bsuite.environments import catch
from replaybuffer import ReplayBuffer
from learner import Learner
from models import fSVGDEnsemble
from nets import CatchNet
import ray
from actor import Actor, DQNAgent
import haiku as hk
import util
from parameter_server import ParameterServer
import time
import numpy as np
import logging

replay_buffer_size = 10000
batch_size = 4
learning_rate = 1e-3
random_seed = 42
lambda_ = 0.5
discount_factor = 0.99
n_networks = 20
unroll_length = 20
n_actors = 5

def main():
    ray.init()
    build_env = catch.Catch
    env_for_spec = build_env()
    timestep = env_for_spec.reset()

    timestep = util.preprocess_step(timestep)
    timestep = jax.tree_map(lambda x: x.reshape(1, *x.shape), timestep)
    # print(timestep)
    num_actions = env_for_spec.action_spec().num_values

    replaybuffer = ReplayBuffer.remote(replay_buffer_size)
    parameter_server = ParameterServer.remote()

    net = hk.without_apply_rng(hk.transform(lambda x: CatchNet(num_actions)(x) ))

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

    print(ray.get(replaybuffer.get_num_frames.remote()))
    learner.run.remote()
    # this is stupid
    while not ray.get(parameter_server.get_params_set.remote()):
        print(time.sleep(0.5))
        print('Waiting for learner..')
    start_params = ray.get(parameter_server.get_params.remote())
    ray.get([actor.run.remote(10000) for actor in actors])
    print(ray.get(replaybuffer.get_num_frames.remote()))

    end_params = ray.get(parameter_server.get_params.remote())

    print(jax.tree_multimap(lambda x, y: np.sum((x - y)**2), start_params, end_params))
    print(end_params)

    ray.shutdown()

main()