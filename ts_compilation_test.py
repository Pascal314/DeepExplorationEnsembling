import haiku as hk
import jax.numpy as jnp
import jax
import time

# A sharing experiment is a poor example.

if __name__ == "__main__":
    rng_key = jax.random.PRNGKey(42)
    data_set = jax.random.normal(rng_key, shape=(10, 2))

    def huge_net(x):
        y = hk.nets.MLP((1000, 1000, 1))(x)
        return y

    net = hk.without_apply_rng(hk.transform(huge_net))

    vnet = hk.Transformed(jax.vmap(net.init, in_axes=(0, None)), jax.vmap(net.apply, in_axes=(0, None)))
    params = net.init(rng_key, data_set)
    vparams = vnet.init(jax.random.split(rng_key, 10), data_set)

    @jax.jit
    def normal_apply(params, x):
        return net.apply(params, x)

    normal_apply(params, data_set)
    start = time.time()
    for i in range(100):
        data_set = jax.random.normal(rng_key, shape=(10, 2))
        y = normal_apply(params, data_set)
    end = time.time()
    print(end - start)

    @jax.jit
    def vnormal_apply(vparams, x):
        return vnet.apply(vparams, x)

    vnormal_apply(vparams, data_set)
    start = time.time()
    for i in range(100):
        data_set = jax.random.normal(rng_key, shape=(10, 2))
        y = vnormal_apply(vparams, data_set)
    end = time.time()
    print(end - start)


    @jax.jit
    def thompson_sampled_apply(vparams, x, i):
        return vnet.apply(vparams, x)[:, i % 10]

    thompson_sampled_apply(vparams, data_set, i)

    start = time.time()
    for i in range(100):
        data_set = jax.random.normal(rng_key, shape=(10, 2))
        y = thompson_sampled_apply(vparams, data_set, i)
    end = time.time()
    print(end - start)


    @jax.jit
    def correct_apply(vparams, x, i):
        params = jax.tree_map(lambda x: x[i % 10], vparams)
        return net.apply(params, x)


    correct_apply(vparams, data_set, i)
    start = time.time()
    for i in range(100):
        data_set = jax.random.normal(rng_key, shape=(10, 2))
        y = correct_apply(vparams, data_set, i)
    end = time.time()
    print(end - start)


    print("Conclusion: jitted naive apply works just as well as the correct way of doing it, although the correct way of doing it is also incredibly easy. However, the naive version is more flexible")