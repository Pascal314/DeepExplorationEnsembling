import ray
import haiku as hk

@ray.remote
class ParameterServer:
    def __init__(self):
        self.params_set = False

    def get_params(self):
        if self.params_set:
            return self.params
        else:
            print("Initialize the learner first")
            raise NotImplemented

    def update_params(self, new_params):
        self.params = new_params
    
    def init_params(self, params):
        self.params = params
        self.params_set = True

    def get_params_set(self):
        return self.params_set