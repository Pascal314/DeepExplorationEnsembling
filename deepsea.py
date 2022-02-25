#@title (CODE) Define the _Deep Sea_ environment
import numpy as np
from typing import NamedTuple, Tuple, Union
import matplotlib.pyplot as plt # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore
from dm_env import TimeStep
import dm_env

# Make this work with dm_env

class DeepSea(object):

  def __init__(self,
               size: int, 
               seed: int = None, 
               randomize: bool = True):

    self._size = size
    self._move_cost = 0.01 / size
    self._goal_reward = 1.
    self._discount = 0.99
    self._column = 0
    self._row = 0

    if randomize:
      rng = np.random.RandomState(seed)
      self._action_mapping = rng.binomial(1, 0.5, size)
    else:
      self._action_mapping = np.ones(size, dtype=np.int32)

    self._reset_next_step = False

  def step(self, action: int) -> TimeStep:
    if self._reset_next_step:
      return self.reset()
    # Remap actions according to column (action_right = go right)
    action_right = action == self._action_mapping[self._column]

    # Compute the reward
    reward = 0.
    if self._column == self._size-1 and action_right:
      reward += self._goal_reward

    # State dynamics
    if action_right:  # right
      self._column = np.clip(self._column + 1, 0, self._size-1)
      reward -= self._move_cost
    else:  # left
      self._column = np.clip(self._column - 1, 0, self._size-1)

    # Compute the observation
    self._row += 1
    if self._row == self._size:
      observation = self._get_observation(self._row-1, self._column)
      self._reset_next_step = True
      return TimeStep(reward=reward, observation=observation, step_type=dm_env.StepType.MID, discount_factor=self._discount)
    else:
      observation = self._get_observation(self._row, self._column)
      return TimeStep(reward=reward, observation=observation, step_type=dm_env.StepType.LAST, discount_factor=self._discount)

  def reset(self) -> TimeStep:
    self._reset_next_step = False
    self._column = 0
    self._row = 0
    observation = self._get_observation(self._row, self._column)
    reward = None
    return TimeStep(reward=reward, observation=observation, step_type=dm_env.StepType.FIRST, discount_factor=self._discount)
  
  def _get_observation(self, row, column) -> np.ndarray:
    observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
    observation[row, column] = 1

    return observation

  @property
  def obs_shape(self) -> Tuple[int, ...]:
    return self.reset().observation.shape

  @property
  def num_actions(self) -> int:
    return 2

  @property
  def optimal_return(self) -> float:
    return self._goal_reward - self._move_cost


#@title Helpers for playing Deep Sea interactively

def get_user_action():
  action = input('Action ([a] = 0, [d] = 1, [q] = Quit): ')
  if action == 'a':
    action = 0
  elif action == 'd':
    action = 1
  elif action == 'q':
    return -1
  else:
    print('Bad action! Must be `a` or `d` or `q`.')
    return get_user_action()
  return action

def play(env):
  fig, ax = plt.subplots(1, 1)
  def plot(observation):
    plt.grid(False)
    ax = plt.gca()
    # ax.set_axis_off()
    # Major ticks
    ax.set_xticks(np.arange(0, 11, 1));
    ax.set_yticks(np.arange(0, 11, 1));

    # Labels for major ticks
    ax.set_xticklabels(np.arange(0, 11, 1));
    ax.set_yticklabels(np.arange(0, 11, 1));

    # Minor ticks
    ax.set_xticks(np.arange(-.5, 11, 1), minor=True);
    ax.set_yticks(np.arange(-.5, 11, 1), minor=True);

    # Gridlines based on minor ticks
    ax.grid(which='minor', color='k', linestyle='-', linewidth=1)
    ax.imshow(observation, interpolation='none')

  def frames():
    episode_return = 0
    step = env.reset()
    yield step.observation
    while step.pcont:
      a = get_user_action()
      if a == -1:
        break  # User quit
      step = env.step(a)
      episode_return += step.reward

      yield step.observation
    print('Episode return: {}'.format(episode_return))
    return 
    

  anim = FuncAnimation(fig, plot, frames, repeat=False, init_func=lambda: None)

  plt.show()

if __name__ == "__main__":
    env = DeepSea(10, 42, randomize=False)
    play(env)