import jax
from bsuite.environments import catch, deep_sea
import util
from actor import AgentOutput
import numpy as np
import rlax

class RandomActor():
    def __init__(self, env_builder):
        self._env = env_builder()
        self._traj = []
        self._episode_return = 0
        self._timestep = self._env.reset()

    def unroll(self, unroll_length: int) -> util.Trajectory:
        """Run unroll_length agent/environment steps, returning the trajectory."""
        timestep = self._timestep
        # Unroll one longer if trajectory is empty.
        num_interactions = unroll_length + int(not self._traj)

        for i in range(num_interactions):
            timestep = util.preprocess_step(timestep)
            action = np.random.randint(3)
            agent_out = AgentOutput(action=action)
                                                    #  agent_state)
            self._traj.append( dict(timestep=timestep, agent_out=agent_out) )
            # agent_state = next_state
            timestep = self._env.step(agent_out.action)

            if timestep.last():
                self._episode_return += timestep.reward
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

def multi_step_lambda(q_tm1, q_t, trajectories, lambda_, discount):
    # From the rlax q_lambda implementation

    discount_t = trajectories.discount[1:] * discount

    target_tm1 = jax.vmap(rlax.lambda_returns, in_axes=(None, None, 0, None))(r_t, discount_t, v_t, lambda_)
    # print(r_t.shape, v_t.shape, target_tm1.shape, q_tm1.shape, q_t.shape)
    action_ohe = jax.nn.one_hot(a_t, num_classes=q_tm1.shape[-1])
    td_loss = jnp.mean( (jax.lax.stop_gradient(target_tm1) - jnp.sum(q_tm1 * action_ohe, axis=-1)) **2)
    return td_loss


if __name__ == '__main__':
    env_builder = catch.Catch
    env_builder = lambda: deep_sea.DeepSea(10)
    actor = RandomActor(env_builder)
    trajectory_1 = actor.unroll(20)
    print(trajectory_1.step_type, trajectory_1.discount, trajectory_1.observation)
    # for el in trajectory_1:
    #     print(el.shape)

    # trajectory_2 = actor.unroll(20)
    # for el in trajectory_2:
    #     print(el.shape)

    def q_func(trajectory):
        obs = trajectory.observation
        obs = obs.reshape(obs.shape[0], -1)
        return np.sum(obs[:, :, None] * np.arange(obs.shape[1] * 3).reshape(1, obs.shape[1], 3), axis=1)

    q_tm1 = q_func(trajectory_1)[:-1]
    q_t = q_func(trajectory_1)[1:]

    v_t = np.max(q_t, axis=-1)
    r_t = trajectory_1.reward[1:]
    a_tm1 = trajectory_1.action[:-1]

    discount = 0.99
    discount_t = trajectory_1.discount[1:] * discount

    lambda_ = 0.

    target_tm1 = rlax.lambda_returns(r_t, discount_t, v_t, lambda_)

    print(q_tm1, q_t)

    print(q_tm1.shape)
    print(q_t.shape)
    # this should be zero if done correctly (for lambda = 0)
    print(target_tm1 - ( r_t + discount_t * np.max(q_t, axis=1)))