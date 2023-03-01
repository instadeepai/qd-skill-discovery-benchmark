"""Probability distributions in JAX."""


import jax
import jax.numpy as jnp

from brax.training.distribution import ParametricDistribution


class CategoricalDistribution:
    def __init__(self, logits):
        self.logits = logits

    def sample(self, seed):
        probs = jax.nn.softmax(self.logits, axis=-1)
        num_category = probs.shape[-1]
        probs = probs.reshape((-1, num_category))

        def _sample(key, probs):
            return jax.random.choice(key, a=jnp.arange(len(probs)), shape=(1,), p=probs)

        keys = jax.random.split(seed, probs.shape[0])
        samples = jax.vmap(_sample)(keys, probs)
        return samples.reshape(self.logits.shape[:-1])

    def log_prob(self, x):
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        num_category = self.logits.shape[-1]
        x = jax.nn.one_hot(x, num_category)
        return log_probs * x

    def entropy(self):
        probs = jax.nn.softmax(self.logits, axis=-1)
        log_probs = jax.nn.log_softmax(self.logits, axis=-1)
        return -probs * log_probs


class IdBijector:
    def forward(self, x):
        return x

    def inverse(self, y):
        return y

    def forward_log_det_jacobian(self, x):
        return 0


class Categorical(ParametricDistribution):
    def __init__(self, event_size):
        """Initialize the distribution.

        Args:
          event_size: the size of events (i.e. actions).
        """

        super().__init__(
            param_size=event_size,
            postprocessor=IdBijector(),
            event_ndims=1,
            reparametrizable=False,
        )

    def create_dist(self, parameters):
        logits = parameters
        dist = CategoricalDistribution(logits)
        return dist
