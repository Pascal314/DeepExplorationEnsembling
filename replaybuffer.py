import jax.numpy as jnp
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.num_frames = 0

    def push(self, trajectory):
        self.buffer.append(trajectory)
        self.num_frames = self.num_frames + trajectory.rewards.shape[0]

    def sample(self, batch_size):
        return np.stack(random.sample(self.buffer, batch_size), axis=0)
        
    def get_num_frames(self):
        return self.num_frames