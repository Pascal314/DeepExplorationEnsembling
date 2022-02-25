import ray
import random
import numpy as np
import time
import logging


@ray.remote
class ReplayBuffer():
    def __init__(self, size):
        self.memory = np.zeros(size, dtype=int)
        self.idx = 0
        self.size = size
    
    def get_memory(self):
        return self.memory

    def put_in_memory(self, action):
        self.memory[self.idx] = action
        self.idx += 1
        self.idx = self.idx % self.size

    def block_forever(self):
        time.sleep(1000)    

@ray.remote
class Actor():
    def __init__(self, p, n, buffer):
        self.p = p
        self.n = n
        self.buffer = buffer
        logging.basicConfig(level=logging.INFO)

    def step(self):
        return np.random.choice(np.arange(self.n), p=self.p)

    def unroll(self, length):
        logging.info('Unrolling')
        for i in range(length):
            self.buffer.put_in_memory.remote(self.step())



ray.init()

p = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
buffer = ReplayBuffer.remote(32)
workers = [Actor.remote(p[i], 3, buffer) for i in range(2)]

# logging.basicConfig(level=logging.INFO)
# logging.info(buffer)
# logging.info(workers)

for _ in range(5):
    # buffer.block_forever.remote()
    ray.get([actor.unroll.remote(5) for actor in workers])
    print(ray.get(buffer.get_memory.remote()))

