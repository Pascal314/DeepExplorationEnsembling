from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import dm_env
from typing import Tuple, NamedTuple

class AgentOutput(NamedTuple):
    action: int


class DQNAgent():
    def __init__(self, net_apply):
        self._net = net_apply
        self._discount = 0.99

    @partial(jax.jit, static_argnums=0)
    def step(
        self,
        params: hk.Params,
        rng: jnp.ndarray,
        timestep: dm_env.TimeStep
    ) -> AgentOutput:
        timestep = jax.tree_map(lambda t: jnp.expand_dims(t, 0), timestep)
        Q_values, _ = self._net(params, timestep) 
        action = jnp.argmax(Q_values)
        return AgentOutput(action=action)
    
