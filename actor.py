import jax.numpy as jnp
import haiku as hk
from typing import Any, Callable, Tuple
import ray
import jax
import dm_env
import learner
import util
from typing import List
import numpy as np
from functools import partial
from typing import NamedTuple, Dict
from replaybuffer import ReplayBuffer
from parameter_server import ParameterServer
import logging
from collections import deque
import time


class AgentOutput(NamedTuple):
    action: int


class DQNAgent():
    def __init__(self, net_apply):
        self._net = net_apply
        self._discount = 0.99

    @partial(jax.jit, static_argnums=0)
    def step(
        self,
        params: hk.Params,
        timestep: dm_env.TimeStep
    ) -> AgentOutput:
        timestep = jax.tree_map(lambda t: jnp.expand_dims(t, 0), timestep)
        Q_values = self._net(params, timestep) 
        action = jnp.argmax(Q_values)
        return AgentOutput(action=action)

@ray.remote
class Actor:
    """Manages the state of a single agent/environment interaction loop."""

    def __init__(
        self,
        agent: DQNAgent,
        env_builder: dm_env.Environment,
        learner: learner.Learner,
        unroll_length: int,
        memory_buffer: ReplayBuffer,
        n_networks: int,
        rng_seed: int,
        logger: Any,
        convert_params: Callable,
        parameter_server: ParameterServer
    ):
        self._agent = agent
        self._env = env_builder()
        self._unroll_length = unroll_length
        self._memory_buffer = memory_buffer
        self._timestep = self._env.reset()
        # self._agent_state = agent.initial_state(None)
        self._traj: List[Dict[dm_env.TimeStep, AgentOutput]] = []
        self._rng_key = jax.random.PRNGKey(rng_seed)
        self._parameter_server = parameter_server
        self._memory_buffer = memory_buffer
        self._logger = logger

        self._episode_return = 0.
        self._n_networks = n_networks
        self._convert_params = convert_params
        self._average_episode_return: deque[float] = deque(maxlen=100)

    def unroll(self, params: hk.Params,
               unroll_length: int) -> util.Trajectory:
        """Run unroll_length agent/environment steps, returning the trajectory."""
        timestep = self._timestep
        # agent_state = self._agent_state
        # Unroll one longer if trajectory is empty.
        num_interactions = unroll_length + int(not self._traj)

        for i in range(num_interactions):
            timestep = util.preprocess_step(timestep)
            agent_out = self._agent.step(params, timestep)
                                                    #  agent_state)
            self._traj.append( dict(timestep=timestep, agent_out=agent_out) )
            # agent_state = next_state
            timestep = self._env.step(agent_out.action)

            if timestep.last():
                self._episode_return += timestep.reward
                logging.info({
                    'episode_return': self._episode_return,
                })
                self._average_episode_return.append(self._episode_return)
                self._episode_return = 0.
            else:
                self._episode_return += timestep.reward or 0.

            # Elide a manual agent_state reset on step_type.first(), as the ResetCore
            # already takes care of this for us.

        # Pack the trajectory and reset parent state.
        trajectory = jax.device_get(self._traj)
        trajectory = jax.tree_multimap(lambda *xs: np.stack(xs), *trajectory)
        trajectory = util.Trajectory(
            step_type=trajectory['timestep'].step_type,
            reward=trajectory['timestep'].reward,
            discount=trajectory['timestep'].discount,
            observation=trajectory['timestep'].observation,
            action=trajectory['agent_out'].action
        )
        self._timestep = timestep
        # self._agent_state = agent_state
        # Keep the bootstrap timestep for next trajectory.
        self._traj = self._traj[-1:]
        return trajectory

    def unroll_and_push(self, params: hk.Params):
        """Run one unroll and send trajectory to learner."""

        trajectory = self.unroll(
            params=params,
            unroll_length=self._unroll_length)
        self._memory_buffer.push.remote(trajectory)

    def pull_params(self):
        return self._parameter_server.get_params.remote()

    
    def run(self, n_episodes):
        start_time = time.time()
        for i in range(n_episodes):
            params = ray.get(self.pull_params())
            # params = jax.device_put(params)
            self._rng_key, subkey = jax.random.split(self._rng_key)
            ensemble_idx = jax.random.randint(subkey, (), 0, self._n_networks)    
            params = self._convert_params(params, ensemble_idx)        
            self.unroll_and_push(params)

            if (i % 100 == 0) and (i > 0):
                self._logger.write(f'average reward: {np.mean(self._average_episode_return)}, throughput: {(i * self._unroll_length) / (time.time() - start_time):.2f}')


