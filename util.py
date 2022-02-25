from typing import NamedTuple
import jax.numpy as jnp
import numpy as np
import tree
import dm_env
import logging

class TempLogger:
    """Uses print because I can't get logging to work at the moment, I obviously want to switch to a real logger (maybe sacred) later"""
    def write(self, d):
        print(d)
    
    def close(self):
        pass

class AbslLogger:
  """Writes to logging.info."""

  def write(self, d):
    logging.info(d)

  def close(self):
    pass

class Trajectory(NamedTuple):
    step_type: dm_env.StepType
    reward: jnp.ndarray
    discount: jnp.ndarray
    observation: jnp.ndarray
    action: jnp.ndarray

class NullLogger:
  """Logger that does nothing."""

  def write(self, *args):
    pass

  def close(self):
    pass

def _preprocess_none(t) -> np.ndarray:
  if t is None:
    return np.array(0., dtype=np.float32)
  else:
    return np.asarray(t)

def preprocess_step(timestep: dm_env.TimeStep) -> dm_env.TimeStep:
  if timestep.discount is None:
    timestep = timestep._replace(discount=1.)
  return tree.map_structure(_preprocess_none, timestep)