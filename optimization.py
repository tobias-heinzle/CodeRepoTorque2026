from typing import Callable
from tqdm import trange

from jax import Array, vmap
from jax.random import key, split, normal, choice
import jax.numpy as jnp

# Simple implementation of an evolution strategy to fit the failure time distribution
  
def fit_absorption_times(
        n_params: int, 
        objective_function: Callable[[Array], float], 
        transition_matrix_builder: Callable[[Array], Array],
        n_generations: int = 5,
        n_population: int = 40_000,
        n_survivors: int = 100,
        n_best: int = 1000, 
        scale: float = 0.001,
        param_bounds: tuple[float, float] = ( 0.000001, 0.3),
        seed: int = 0,
        ):


    rng_key = key(seed)
    init_key, rng_key = split(rng_key)

    # Initialization to sufficiently small values is crucial!
    population = jnp.clip(normal(init_key, (n_population, n_params)) * scale, *param_bounds)

    score_history = jnp.full(n_generations, jnp.inf)

    pbar = trange(n_generations)
    pbar.set_description('Current best score = Inf')

    for k in pbar:
        # This try block only handles graceful stopping upon KeyboardInterrupt!
        try:
            # Each iteration has its own random values
            sample_key, noise_key, rng_key = split(rng_key, 3)
            
            # Map parameters to transition probabilities
            arrays = vmap(transition_matrix_builder)(population)

            # Make sure that all arrays are truly probability distributions
            assert jnp.all(arrays >= 0.0)
            assert jnp.allclose(arrays.sum(axis=-1), 1)
            
            # Process the scores of the population based on the objective function
            scores = vmap(objective_function)(arrays)
            ranking = jnp.argsort(scores)
            best = population[ranking[:n_best]]

            # Keep some statistics
            score_history = score_history.at[k].set(scores[ranking[0]])

            # Re-sample the remaining population that did not survive
            samples = choice(sample_key, best, (n_population - n_survivors,), replace=True)
            samples = jnp.clip(samples + normal(noise_key, samples.shape, ) * scale, 0.0000001, 0.03)

            population = jnp.concat([
                population[ranking[:n_survivors]],
                samples
            ])

            # Update progress bar
            pbar.set_description_str(f'Current best score = {min(scores):.5f}')
        except KeyboardInterrupt:
            print(f'Terminated at iteration {k}')
            break

    best_parameters = transition_matrix_builder(population[0])
    return best_parameters, population, score_history
