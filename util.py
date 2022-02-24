from typing import NamedTuple
import jax.numpy as jnp

class Trajectory(NamedTuple):
    rewards: jnp.ndarray
    discount: jnp.ndarray
