import haiku as hk
import jax.numpy as jnp
import dm_env
import jax

class CatchNet(hk.Module):
  """A simple neural network for catch."""

  def __init__(self, num_actions, name=None):
    super().__init__(name=name)
    self._num_actions = num_actions

  def __call__(self, x: dm_env.TimeStep):
    torso_net = hk.Sequential(
        [hk.Flatten(),
         hk.Linear(128), jax.nn.relu,
         hk.Linear(64), jax.nn.relu])
    torso_output = torso_net(x.observation)
    Q_values = hk.Linear(self._num_actions)(torso_output)
    return Q_values