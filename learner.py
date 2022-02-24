import jax.numpy as jnp
import jax
from collections import deque
from typing import Tuple, Dict
import random
import optax
from agent import DQNAgent
import haiku as hk
import rlax
import util
from functools import partial
from replaybuffer import ReplayBuffer

class Learner:
    def __init__(
        self,
        agent: DQNAgent,
        opt: optax.GradientTransformation,
        batch_size: int,
        discount_factor: float,
        rng_key: jnp.ndarray,
        learner_transformed: hk.Transformed,
        dummy_observation: jnp.ndarray,
        lambda_: float,
        replaybuffer: ReplayBuffer
    ):
        self._agent = agent
        self._opt = opt
        self._transformed = learner_transformed
        self._params = learner_transformed.init(rng_key, dummy_observation)
        self._target_params = self._params
        self._lambda = lambda_
        self._replaybuffer = replaybuffer

    def get_params(self) -> hk.Params:
        return self._params

    def _loss(
        self, 
        params: hk.Params, 
        target_params: hk.Params,
        trajectories: util.Trajectory,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        # From the rlax q_lambda implementation
        q_tm1 = self._transformed.apply(params, trajectories)
        q_t = self._transformed.apply(target_params, trajectories)
        v_t = jnp.max(q_t, axis=-1)
        r_t = trajectories.rewards
        discount_t = trajectories.discount

        target_tm1 = rlax.multistep.lambda_returns(r_t, discount_t, v_t, self._lambda)
        td_loss = jnp.mean( (jax.lax.stop_gradient(target_tm1) - q_tm1)**2)

        logs = {}
        logs['td_loss'] = td_loss
        return td_loss, logs

    @partial(jax.jit, static_argnums=0)
    def update(self, 
        params: hk.Params,
        opt_state: optax.OptState,
        batch: util.Trajectory,
    ) -> Tuple[hk.Params, optax.OptState, Dict]:
        (_, logs), grads = jax.value_and_grad(self._loss, has_aux=True)(params, batch)
        grad_norm_unclipped = optax.l2_norm(grads)
        updates, opt_state = self._opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        logs.update({
            'grad_norm_unclipped': grad_norm_unclipped
        })
        return params, opt_state, logs

    def sample_batch(self):
        pass
