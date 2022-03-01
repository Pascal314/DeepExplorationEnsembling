import jax.numpy as jnp
from collections import deque
import random
import ray
import numpy as np
import jax

@ray.remote
class ReplayBuffer:
    def __init__(self, capacity, short_term_capacity):
        self.long_term_buffer = deque(maxlen=capacity)
        self.short_term_buffer = deque(maxlen=short_term_capacity)
        self.num_frames = 0

    def push(self, trajectory):
        self.long_term_buffer.append(trajectory)
        self.short_term_buffer.append(trajectory)
        self.num_frames = self.num_frames + trajectory.reward.shape[0]

    def sample(self, batch_size):
        old_batch = random.choices(self.long_term_buffer, k=batch_size // 2)
        fresh_batch = random.choices(self.short_term_buffer, k=batch_size // 2)
        # stack or concatenate?
        # old_batch = jax.tree_multimap(lambda *x: np.concatenate(x, axis=0), *old_batch)
        # fresh_batch = jax.tree_multimap(lambda *x: np.concatenate(x, axis=0), *fresh_batch)
        batch = jax.tree_multimap(lambda *x: np.stack(x, axis=0), *(old_batch + fresh_batch))
        return batch
        
    def get_num_frames(self):
        return self.num_frames