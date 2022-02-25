import jax.numpy as jnp
from collections import deque
import random
import ray
import numpy as np
import jax

@ray.remote
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.num_frames = 0

    def push(self, trajectory):
        self.buffer.append(trajectory)
        self.num_frames = self.num_frames + trajectory.reward.shape[0]

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size, )
        # stack or concatenate?
        batch = jax.tree_multimap(lambda *x: np.concatenate(x, axis=0), *batch)
        return batch
        
    def get_num_frames(self):
        return self.num_frames