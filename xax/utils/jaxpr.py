"""Visualize JAXPR."""

from pathlib import Path

import jax
import jax.core


def save_jaxpr_dot(closed_jaxpr: jax.core.ClosedJaxpr, filename: str | Path) -> None:
    """Save the JAXPR to a DOT file.

    Example usage:

        grad_fn_jaxpr = jax.make_jaxpr(loss_fn)(variables)
        save_jaxpr_dot(grad_fn_jaxpr, "grad_fn_jaxpr.dot")

    Then, you can visualize the JAXPR using Graphviz:

        dot -Tpng grad_fn_jaxpr.dot > grad_fn_jaxpr.png

    Args:
        closed_jaxpr: The closed JAXPR to save.
        filename: The filename to save the JAXPR to.
    """
    if hasattr(closed_jaxpr, "jaxpr"):
        jaxpr = closed_jaxpr.jaxpr
    else:
        jaxpr = closed_jaxpr

    with open(filename, "w") as f:
        f.write("digraph Jaxpr {\n")

        var_names: dict[jax.core.Var, str] = {}
        var_count = 0

        def get_var_name(var: jax.core.Var) -> str:
            """Get a unique name for a variable."""
            nonlocal var_names, var_count

            # Handle Literal objects specially since they're not hashable
            if isinstance(var, jax.core.Literal):
                # Create a name based on the literal value
                name = f"lit_{var.val}"
                return name

            # For other variables
            if var not in var_names:
                name = f"var_{var_count}"
                var_names[var] = name
                var_count += 1
            return var_names[var]

        for var in jaxpr.invars:
            node_name = get_var_name(var)
            f.write(f'  {node_name} [label="{node_name}\\n(input)"];\n')

        eq_count = 0
        for eq in jaxpr.eqns:
            eq_node = f"eq{eq_count}"
            label = f"{eq.primitive.name}"
            f.write(f'  {eq_node} [shape=box, label="{label}"];\n')

            for invar in eq.invars:
                var_name = get_var_name(invar)
                f.write(f"  {var_name} -> {eq_node};\n")

            for outvar in eq.outvars:
                var_name = get_var_name(outvar)
                f.write(f"  {eq_node} -> {var_name};\n")

            eq_count += 1

        for var in jaxpr.outvars:
            node_name = get_var_name(var)
            f.write(f'  {node_name} [peripheries=2, label="{node_name}\\n(output)"];\n')

        f.write("}\n")
