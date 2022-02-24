class Model():
    def __init__(self, transformed, optimizer, dummy_input, log_likelihood, uncertainty_func):
        self.optimizer = optimizer
        self.transformed = transformed
        self.dummy_input = dummy_input
        self.uncertainty_func = uncertainty_func
        self.log_likelihood = log_likelihood

    def apply(self, *args):
        return self.transformed.apply(*args)

    def init(self, *args):
        return self.transformed.init(*args)

    def loss(self, params, batch):
        raise NotImplementedError

    def train_step(self, params, opt_state, batch):
        raise NotImplementedError
