import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
from xax.utils.graph_logging import _hlo_text_to_graphdef


def hlo_text(fn, *args, static_argnums=()):
    """Lower a JAX function to HLO text for given example inputs"""
    return (
        jax.jit(fn, static_argnums=static_argnums)
           .lower(*args)
           .compiler_ir("hlo")
           .as_hlo_text()
    )


def graphdef_from_fn(fn, *args, static_argnums=()):
    """Convert HLO text of a function to a TensorFlow GraphDef"""
    hlo = hlo_text(fn, *args, static_argnums=static_argnums)
    return _hlo_text_to_graphdef(hlo)


def test_add_op_present():
    def f(x, y):
        return x + y
    gd = graphdef_from_fn(f, jnp.ones((3,)), jnp.ones((3,)))
    ops = {node.op for node in gd.node}
    assert 'add' in ops, f"Expected 'add' op in GraphDef, but got {ops}"


def test_mul_op_present():
    def f(x, y):
        return x * y
    gd = graphdef_from_fn(f, jnp.ones((4,)), jnp.ones((4,)))
    ops = {node.op for node in gd.node}
    assert 'multiply' in ops or 'mul' in ops, f"Expected 'multiply'/'mul' op, but got {ops}"


def test_dot_op_present():
    def f(x, y):
        return jnp.dot(x, y)
    a = jnp.ones((2, 3))
    b = jnp.ones((3, 2))
    gd = graphdef_from_fn(f, a, b)
    ops = {node.op for node in gd.node}
    assert 'dot' in ops or 'dot_general' in ops, f"Expected 'dot' op in GraphDef, but got {ops}"


def test_graph_not_empty():
    def f(x):
        return x * 2
    gd = graphdef_from_fn(f, jnp.ones((5,)))
    assert len(gd.node) > 0, "Expected GraphDef to contain nodes, but it's empty"


def test_conv_model():
    # simple 2D conv test
    def conv(x):
        kernel = jnp.ones((3, 3, 1, 1))
        return jax.lax.conv_general_dilated(
            x, kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    x = jnp.ones((1, 16, 16, 1))
    gd = graphdef_from_fn(conv, x)
    ops = {node.op for node in gd.node}
    assert any(op.startswith("conv") for op in ops), f"Expected a conv op, but got {ops}"