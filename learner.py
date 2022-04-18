import jax.numpy as jnp
import jax
from collections import deque
from typing import Tuple, Dict, Any, List
import random
import optax
import haiku as hk
import rlax
import util
from functools import partial
from replaybuffer import ReplayBuffer
import itertools
import models
import ray
import time
from parameter_server import ParameterServer
import time
import math
import numpy as np

@ray.remote(num_gpus=1)
class Learner:
    def __init__(
        self,
        opt: optax.GradientTransformation,
        batch_size: int,
        discount_factor: float,
        rng_key: jnp.ndarray,
        model: models.Model,
        dummy_observation: jnp.ndarray,
        lambda_: float,
        replaybuffer: ReplayBuffer,
        logger: Any,
        parameter_servers: List[ParameterServer],
        target_update: float,
    ):
        self._opt = opt
        self._model = model
        self._params = model.ensemble_transformed.init(rng_key, dummy_observation)
        self._target_params = self._params
        self._lambda = lambda_
        self._replaybuffer = replaybuffer
        self._batch_size = batch_size
        self._done = False
        self._logger = logger
        self._parameter_servers = parameter_servers
        self._discount = discount_factor
        self._target_update = target_update


        params_id = ray.put(self._params)
        ray.get([server.init_params.remote(params_id) for server in self._parameter_servers])


    def push_params(self):
        params_id = ray.put(self._params)
        ray.get([server.update_params.remote(params_id) for server in self._parameter_servers])
    
    @partial(jax.jit, static_argnums=0)
    def update(self, 
        params: hk.Params,
        opt_state: optax.OptState,
        target_params: hk.Params,
        batch: util.Trajectory,
    ) -> Tuple[hk.Params, optax.OptState, Dict]:
        # print('Surely this happens')
        (_, logs), grads = jax.value_and_grad(self._model.loss, has_aux=True)(params, target_params, batch, self._lambda, self._discount)
        # print('Wow!')
        grad_norm_unclipped = sum( jax.tree_leaves(jax.tree_map(lambda x: jnp.sum(x**2), grads)))
        # print('Cool!')
        updates, opt_state = self._opt.update(grads, opt_state)
        # print('Amazing!')
        params = optax.apply_updates(params, updates)
        # print('What?')
        logs.update({
            'grad_norm_unclipped': grad_norm_unclipped
        })
        # print('How?')
        return params, opt_state, logs

    def sample_batch(self):
        batch = ray.get(self._replaybuffer.sample.remote(self._batch_size))
        return batch

    def run(self, max_iterations: int = -1):
        num_frames = 0
        opt_state = self._opt.init(self._params)

        steps = range(max_iterations) if max_iterations != -1 else itertools.count()

        num_frames = ray.get(self._replaybuffer.get_num_frames.remote())
        while num_frames < self._batch_size:
            print('num_frames:', num_frames)
            time.sleep(0.5)
            num_frames = ray.get(self._replaybuffer.get_num_frames.remote())

        print('Starting training', num_frames)
        print('params:', jax.tree_map(lambda x: x.shape, self._params))
        print(f'total_params:  {np.sum(jax.tree_leaves(jax.tree_map(lambda x: math.prod(x.shape), self._params)))}')
    
        start_time = time.time()
        for i in steps:
            batch = self.sample_batch()
            self._params, opt_state, logs = self.update(self._params, opt_state, self._target_params, batch)
            self.push_params()

            # This should clearly be a hyperparameter and not some magic number
            self._target_params = jax.tree_multimap(
                lambda x, y: self._target_update * x + (1 - self._target_update) * y,
                self._target_params, 
                self._params
            )

            num_frames = ray.get(self._replaybuffer.get_num_frames.remote())

            logs.update({
                'num_frames': num_frames,
            })

            throughput = (i * batch[0].reward.shape[0] * batch[0].reward.shape[1]) / (time.time() - start_time)
            logs.update({
                'throughput': f'{throughput:.2f}',
                'time': f'{time.time() - start_time:.2f}'
            })

            if i % 100 == 0:
                self._logger.write(logs)
        self._done = True