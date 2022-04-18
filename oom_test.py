import ray
import numpy as np
from collections import deque
import time
import sys

def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


ray.init()
@ray.remote
class ReplayBuffer:
    def __init__(self, capacity, short_term_capacity):
        self.long_term_buffer = deque(maxlen=capacity)
        self.short_term_buffer = deque(maxlen=short_term_capacity)
        self.num_frames = 0

    def push(self, trajectory):
        self.long_term_buffer.append(trajectory)
        self.short_term_buffer.append(trajectory)
        self.num_frames += 1

    def get_num_frames(self):
        return self.num_frames

@ray.remote
class Actor:
    def __init__(self, replaybuffer):
        self.replaybuffer = replaybuffer
    
    def run(self, n, k):
        for i in range(n):
            self.replaybuffer.push.remote((np.random.normal(size=(1, k))))

@ray.remote
class ParameterServer:
    def __init__(self, size):
        self.params = np.zeros(size)

    def update_params(self, params):
        self.params = params
        print(sizeof_fmt(sys.getsizeof(self.params)))

    def get_params(self):
        return self.params

@ray.remote
class Learner:
    def __init__(self, size, server):
        self.size = size
        self.server = server
    
    def update_params(self):
        params = ray.get(self.server.get_params.remote())
        print(sizeof_fmt(sys.getsizeof(params)))
        params = params + np.random.normal(0, 1, size=self.size)
        print(sizeof_fmt(sys.getsizeof(params)))
        self.server.update_params.remote(params)

    def run(self, n_updates):
        for i in range(n_updates):
            time.sleep(1)
            self.update_params()

buffer = ReplayBuffer.remote(10, 10)
actor = Actor.remote(buffer)

ray.get([actor.run.remote(1000, 10_000_000)])
print(ray.get(buffer.get_num_frames.remote()))

# size = (14000, 1000)
# test = np.zeros(size)
# print(sizeof_fmt(sys.getsizeof(test)))

# server = ParameterServer.remote(size)
# learner = Learner.remote(size, server)

# print(sizeof_fmt(sys.getsizeof(server)), sizeof_fmt(sys.getsizeof(learner)))

# ray.get([learner.run.remote(1000)])

ray.shutdown()