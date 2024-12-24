"""Defines the reinforcement learning submodule."""

try:
    import brax
except ImportError as e:
    raise ImportError(
        "`brax` is not installed - you should install the reinforcement learning dependencies "
        "for this package using `pip install 'xax[rl]'`"
    ) from e
