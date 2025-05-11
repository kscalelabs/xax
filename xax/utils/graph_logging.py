# xax/utils/graph_logging.py

import re
import jax
from tensorboard.compat.proto import graph_pb2, event_pb2
import equinox as eqx
from xax.task.task import Task, Config
from xax.task.loggers.tensorboard import TensorboardLogger


# regex to pull out "%name = op(args…)" lines from HLO text
_HLO_INST_RE = re.compile(
    r"\s*(?:ROOT\s+)?%?([^ =]+)\s*=\s*(?:[^\s]+\s+)*([\w_]+)\((.*?)\)"
)

def _hlo_text_to_graphdef(hlo_text: str) -> graph_pb2.GraphDef:
    #Convert HLO text dump into a TensorBoard GraphDef.
    gd = graph_pb2.GraphDef()
    in_entry = False
    for line in hlo_text.splitlines():
        if not in_entry:
            if line.startswith("ENTRY "):
                in_entry = True
            continue
        if line.startswith("}") or not line.strip():
            break
        m = _HLO_INST_RE.match(line)
        if not m:
            continue
        name, opcode, args = m.groups()
        inputs = [a.strip().lstrip("%") for a in args.split(",") if a.strip()]
        node = gd.node.add()
        node.name = name
        node.op = opcode
        node.input.extend(inputs)
    return gd


def log_jax_graph(fn, example_args, task, cfg, logdir: str, step: int = 0):
    # Unpack the closure-bound arguments
    model_arr, model_static, optimizer, opt_state, batch, state = example_args

    # Wrap out static args so XLA only sees array inputs
    def two_arg_step(ma, b):
        full_model = eqx.combine(ma, model_static)
        return fn(full_model, optimizer, b, state)

    # Lower to HLO via JAX AOT API
    hlo_comp = (
        jax.jit(two_arg_step)
           .lower(model_arr, batch)
           .compiler_ir('hlo')
    )
    hlo_text = hlo_comp.as_hlo_text()

    graph_def = _hlo_text_to_graphdef(hlo_text)  # Convert HLO text to GraphDef

    # Initialize TensorBoard Logger (handles the logging and TensorBoard subprocess)
    tb_logger = TensorboardLogger(run_directory=logdir)

    # 6) Log the graph at the specified steps
    for step in range(cfg.max_steps):
        loss, grads, new_opt_state = task.train_step(
            task.model, task.optimizer, task.state, batch
        )

        # Log the graph at the specified steps
        if step % cfg.valid_every_n_steps == 0:
            # Log the graph to TensorBoard
            writer = tb_logger.get_writer("train")
            writer.pb_writer.add_graph(graph_def)
            writer.pb_writer.flush()
            print(f"Logged graph at step {step} to {logdir}. Run `tensorboard --logdir={logdir}`")
        writer.add_scalar("loss", loss, global_step=step)
        task.state = new_opt_state
