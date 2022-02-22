from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import dm_env
from tyimping import Tuple

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
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        timestep = jax.tree_map(lambda t: jnp.expand_dims(t, 0), timestep)
        Q_values, _ = self._net(params, timestep) # This should be (n_networks, 1, action_shape) (no it doesn't, each agent will use a specific model)
        
        # Here something has to happen to combine Q values # No, we will be using thompson sampling

        # A naive way of using thompson sampling is to supply the step function with an index i and then indexing the output of the neural network with i
        # otherwise it is necessary to mess around with different apply functions.

        # I think however that jax compilation might make this "naive" way just as fast as the correct way.
        action = jnp.argmax(Q_values) 
    
