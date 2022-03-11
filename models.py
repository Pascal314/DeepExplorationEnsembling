import haiku as hk
from typing import NamedTuple, Callable, Dict, Tuple
import util
import jax
import jax.numpy as jnp
import rlax

class Model(NamedTuple):
    ensemble_transformed: hk.Transformed
    individual_transformed: hk.Transformed
    convert_params: Callable[[hk.Params, int], hk.Params]
    loss: Callable[[hk.Params, hk.Params, util.Trajectory, float], Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]]


class fSVGDEnsemble():
    def __init__(self, individual_transformed, n_networks):
        self.individual_transformed = individual_transformed
        vinit = jax.vmap(individual_transformed.init, in_axes=(0, None))
        vapply = jax.vmap(individual_transformed.apply, in_axes=(0, None))
        self.ensemble_transformed = hk.Transformed(init=lambda key, x: vinit(jax.random.split(key, num=n_networks), x), apply=vapply)
    
    def convert_params(self, params, i):
        return jax.tree_map(lambda x: x[i], params)

    def loss(
        self, 
        params: hk.Params, 
        target_params: hk.Params,
        batch: Tuple[util.Trajectory, int],
        lambda_: float,
        discount: float,
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logs = {}
        trajectories, n_data = batch
        # print(trajectories.observation.shape)
        vapply = jax.vmap(self.ensemble_transformed.apply, in_axes=(None, 0), out_axes=1)

        q_tm1 = vapply(params, trajectories)[:, :, :-1]
        # print(q_tm1.shape)
        q_t = vapply(target_params, trajectories)[:, :, 1:]
        td_loss = jnp.mean(multi_step_lambda(q_tm1, q_t, trajectories, lambda_, discount))

        # print(q_tm1.shape)
        # Kij = jax.vmap(jax.vmap(gram_matrix_median_trick, in_axes=1, out_axes=-1), in_axes=1, out_axes=-1)(q_tm1)
        # print(Kij.shape)
        # Kij = jnp.sum(Kij, axis=(-1, -2))

        Kij = jax.vmap(gram_matrix_median_trick)(q_tm1)

        fSVGD_loss = jnp.mean(jnp.sum(Kij, axis=1) / jax.lax.stop_gradient(jnp.sum(Kij, axis=1)))

        batch_axes = trajectories.observation.shape[0:2]
        batch_size = batch_axes[0] * batch_axes[1] 
        # loss = n_data / batch_size * td_loss + jnp.sum(fSVGD_loss)
        # logic: likelihood is supposed to scale with n_data, so should have a factor n_data / batch_size
        # both td loss and fsvgd loss are summed over the batch size, meaning that the factor batch size can just be ignored
        # and we add 1/n to fsvgd instead of n * td loss to make the loss more manageably small
        loss = td_loss + 1 / n_data * fSVGD_loss
        # loss = td_loss + fSVGD_loss

        # logs['q_t'] = q_t
        # logs['q_tm1'] = q_tm1
        logs['td_loss'] = td_loss
        logs['fSVGD_loss'] = fSVGD_loss
        return loss, logs

@jax.jit
def multi_step_lambda(q_tm1, q_t, trajectories, lambda_, discount):
    # From the rlax q_lambda implementation
    v_t = jnp.max(q_t, axis=-1) # ensemble member, batch, time
    r_t = trajectories.reward[:, 1:] # batch, time
    a_tm1 = trajectories.action[:, :-1] # batch, time
    discount_t = trajectories.discount[:, 1:] * discount # batch, time
    # print(r_t.shape, v_t.shape, q_tm1.shape, q_t.shape, trajectories.reward.shape)
    
    # this needs to be vmapped over the batch and over the ensemble
    target_tm1 = jax.vmap(
        jax.vmap(
            rlax.lambda_returns, 
            in_axes=(None, None, 0, None)), # Second vmap is over the ensemble
        in_axes=(0, 0, 1, None), out_axes=1 # First vmap is over the batch
    )(r_t, discount_t, v_t, lambda_)
    # print(r_t.shape, v_t.shape, target_tm1.shape, q_tm1.shape, q_t.shape)
    action_ohe = jax.nn.one_hot(a_tm1, num_classes=q_tm1.shape[-1])
    td_loss = jnp.sum( (jax.lax.stop_gradient(target_tm1) - jnp.sum(q_tm1 * action_ohe, axis=-1)) **2, axis=(1, 2))
    return td_loss

# Do I want the batch to disappear in jnp.ravel?
dmatrix = jax.vmap(jax.vmap(lambda x, y: jnp.sum( ((jnp.ravel(x) - jnp.ravel(y))**2) ), in_axes=(None, 0)), in_axes=(0, None))

@jax.jit
def compute_median(x):
    n = x.shape[0]
    return jax.lax.stop_gradient(jnp.sort(x)[n // 2])

def sinoidal_prior(mus):
    pass

@jax.jit
def uniform_prior(mus):
    return 0

@jax.jit
def normal_log_likelihood(mu, Y, n):
    return -n / Y.shape[0] * jnp.sum(0.5 * (mu - Y)**2)
normal_log_likelihood = jax.jit(jax.vmap(normal_log_likelihood, in_axes=(0, None, None)))

@jax.jit
def gram_matrix_median_trick(x):
    n = x.shape[0]
    distance_matrix = dmatrix(x, jax.lax.stop_gradient(x))
    median = jax.lax.stop_gradient(compute_median(distance_matrix.flatten()))
    return jnp.exp(-distance_matrix / (1e-12 + median / jnp.log(n)))