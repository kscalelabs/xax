"""Utils for accessing, modifying, and otherwise manipulating pytrees."""

import chex
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree


def slice_array(x: Array, start: Array, slice_length: int) -> Array:
    """Get a slice of an array along the first dimension.

    For multi-dimensional arrays, this slices only along the first dimension
    and keeps all other dimensions intact.

    Args:
        x: The array to slice.
        start: The start index of the slice.
        slice_length: The length of the slice.

    Returns:
        The sliced array.
    """
    chex.assert_shape(start, ())
    chex.assert_shape(slice_length, ())
    start_indices = (start,) + (0,) * (len(x.shape) - 1)
    slice_sizes = (slice_length,) + x.shape[1:]

    return jax.lax.dynamic_slice(x, start_indices, slice_sizes)


def slice_pytree(pytree: PyTree, start: Array, slice_length: int) -> PyTree:
    """Get a slice of a pytree."""
    return jax.tree_util.tree_map(lambda x: slice_array(x, start, slice_length), pytree)


def flatten_array(x: Array, flatten_size: int) -> Array:
    """Flatten an array into a (flatten_size, ...) array."""
    reshaped = jnp.reshape(x, (flatten_size, *x.shape[2:]))
    assert reshaped.shape[0] == flatten_size
    return reshaped


def flatten_pytree(pytree: PyTree, flatten_size: int) -> PyTree:
    """Flatten a pytree into a (flatten_size, ...) pytree."""
    return jax.tree_util.tree_map(lambda x: flatten_array(x, flatten_size), pytree)


def pytree_has_nans(pytree: PyTree) -> Array:
    """Check if a pytree has any NaNs."""
    has_nans = jax.tree_util.tree_reduce(
        lambda a, b: jnp.logical_or(a, b),
        jax.tree_util.tree_map(lambda x: jnp.any(jnp.isnan(x)), pytree),
    )
    return has_nans


def update_pytree(cond: Array, new: PyTree, original: PyTree) -> PyTree:
    """Update a pytree based on a condition."""
    # Tricky, need use tree_map because where expects array leafs.
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), new, original)


def compute_nan_ratio(pytree: PyTree) -> Array:
    """Computes the ratio of NaNs vs non-NaNs in a given PyTree."""
    nan_counts = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.isnan(x)), pytree)
    total_counts = jax.tree_util.tree_map(lambda x: x.size, pytree)

    total_nans = jax.tree_util.tree_reduce(lambda a, b: a + b, nan_counts, 0)
    total_elements = jax.tree_util.tree_reduce(lambda a, b: a + b, total_counts, 0)
    overall_nan_ratio = jnp.array(total_nans / total_elements)

    return overall_nan_ratio


def reshuffle_pytree(data: PyTree, batch_shape: tuple[int, ...], rng: PRNGKeyArray) -> PyTree:
    """Reshuffle a pytree along the leading dimensions.

    This function reshuffles the data along the leading dimensions specified by batch_shape.
    Assumes the dimensions to shuffle are the leading ones.

    Args:
        data: A pytree with arrays.
        batch_shape: A tuple of integers specifying the size of each leading dimension to shuffle.
        rng: A JAX PRNG key.

    Returns:
        A new pytree with the same structure but with the data reshuffled along
        the leading dimensions.
    """
    # Create a permutation for the flattened batch dimensions
    flat_size = 1
    for dim in batch_shape:
        flat_size *= dim

    perm = jax.random.permutation(rng, flat_size)

    def permute_array(x: Array) -> Array:
        if not isinstance(x, jnp.ndarray):
            return x

        # Check if the array has enough dimensions
        if len(x.shape) < len(batch_shape):
            return x

        # Check if the dimensions match the batch_shape
        for i, dim in enumerate(batch_shape):
            if x.shape[i] != dim:
                return x

        # Reshape to flatten the batch dimensions
        orig_shape = x.shape
        reshaped = x.reshape((flat_size,) + orig_shape[len(batch_shape) :])

        # Apply the permutation
        permuted = reshaped[perm]

        # Reshape back to the original shape
        return permuted.reshape(orig_shape)

    return jax.tree_util.tree_map(permute_array, data)


def reshuffle_pytree_independently(data: PyTree, batch_shape: tuple[int, ...], rng: PRNGKeyArray) -> PyTree:
    """Reshuffle a rollout array across arbitrary batch dimensions independently of each other."""
    rngs = jax.random.split(rng, len(batch_shape))
    perms = [jax.random.permutation(rng_i, dim) for rng_i, dim in zip(rngs, batch_shape)]
    # n-dimensional index grid from permutations
    idx_grids = jnp.meshgrid(*perms, indexing="ij")

    def permute_array(x: Array) -> Array:
        if isinstance(x, Array):
            return x[tuple(idx_grids)]
        return x

    return jax.tree_util.tree_map(permute_array, data)


TransposeResult = tuple[PyTree, tuple[int, ...], tuple[int, ...]]
PathType = tuple[str | int, ...]


def reshuffle_pytree_along_dims(
    data: PyTree,
    dims: tuple[int, ...],
    shape_dims: tuple[int, ...],
    rng: PRNGKeyArray,
) -> PyTree:
    """Reshuffle a pytree along arbitrary dimensions.

    Allows reshuffling along any dimensions, not just the leading ones.
    It transposes the data to make the specified dimensions the leading ones,
    then reshuffles, and finally transposes back.

    Args:
        data: A pytree with arrays.
        dims: A tuple of integers specifying which dimensions to shuffle along.
            For example, (1,) would shuffle along the second dimension.
        shape_dims: A tuple of integers specifying the size of each dimension to shuffle.
            Must have the same length as dims.
        rng: A JAX PRNG key.

    Returns:
        A new pytree with the same structure but with the data reshuffled along
        the specified dimensions.
    """
    if len(dims) != len(shape_dims):
        raise ValueError(f"dims {dims} and shape_dims {shape_dims} must have the same length")

    def transpose_for_shuffle(x: PyTree) -> TransposeResult:
        if not isinstance(x, jnp.ndarray):
            return x, (), ()

        # Check if the array has enough dimensions
        if len(x.shape) <= max(dims):
            return x, (), ()

        # Check if the dimensions match the shape_dims
        for i, dim in enumerate(dims):
            if x.shape[dim] != shape_dims[i]:
                raise ValueError(f"Array shape {x.shape} doesn't match shape_dims {shape_dims} at dimension {dim}")

        # Create the transpose order to move the specified dimensions to the front
        # while preserving the relative order of the other dimensions
        n_dims = len(x.shape)
        other_dims = [i for i in range(n_dims) if i not in dims]
        transpose_order = tuple(dims) + tuple(other_dims)

        # Transpose the array
        transposed = jnp.transpose(x, transpose_order)

        return transposed, transpose_order, x.shape

    def transpose_back(x: PyTree, transpose_order: tuple[int, ...], original_shape: tuple[int, ...]) -> PyTree:
        if not isinstance(x, jnp.ndarray) or not transpose_order:
            return x

        # Create the inverse transpose order
        inverse_order = [0] * len(transpose_order)
        for i, j in enumerate(transpose_order):
            inverse_order[j] = i

        # Transpose back
        return jnp.transpose(x, inverse_order)

    # First, transpose all arrays to make the specified dimensions the leading ones
    transposed_data: dict[PathType, Array] = {}
    transpose_info: dict[PathType, tuple[tuple[int, ...], tuple[int, ...]]] = {}

    def prepare_for_shuffle(path: PathType, x: PyTree) -> PyTree:
        if isinstance(x, jnp.ndarray):
            transposed, transpose_order, original_shape = transpose_for_shuffle(x)
            if isinstance(transposed, jnp.ndarray):  # Check if it's an array
                transposed_data[path] = transposed
                transpose_info[path] = (transpose_order, original_shape)
        return x

    jax.tree_util.tree_map_with_path(prepare_for_shuffle, data)

    # Create a transposed pytree
    def get_transposed(path: PathType, x: PyTree) -> PyTree:
        if isinstance(x, jnp.ndarray) and path in transposed_data:
            return transposed_data[path]
        return x

    transposed_pytree = jax.tree_util.tree_map_with_path(get_transposed, data)

    # Reshuffle the transposed pytree along the leading dimensions
    reshuffled_transposed = reshuffle_pytree(transposed_pytree, shape_dims, rng)

    # Transpose back
    def restore_transpose(path: PathType, x: PyTree) -> PyTree:
        if isinstance(x, jnp.ndarray) and path in transpose_info:
            transpose_order, original_shape = transpose_info[path]
            return transpose_back(x, transpose_order, original_shape)
        return x

    return jax.tree_util.tree_map_with_path(restore_transpose, reshuffled_transposed)
