# xax/utils/graph_logging.py

import re
import jax
from tensorboard.compat.proto import graph_pb2, event_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


# regex to pull out "%name = op(args…)" lines from HLO text
_HLO_INST_RE = re.compile(
    r"\s*(?:ROOT\s+)?%?([^ =]+)\s*=\s*(?:[^\s]+\s+)*([\w_]+)\((.*?)\)"
)


def _hlo_text_to_graphdef(hlo_text: str) -> graph_pb2.GraphDef:
    """Convert HLO text dump into a TensorBoard GraphDef."""
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


def log_jax_graph(fn, example_args, logdir: str, step: int = 0):
    """
    Logs the HLO graph of `fn` to TensorBoard under `logdir` at `step`.
    `fn` should be a bound train_step accepting (model_arr, model_static, optimizer,
    opt_state, batch, state). `example_args` is that 6-tuple.
    """
    # Unpack the closure-bound arguments
    model_arr, model_static, optimizer, opt_state, batch, state = example_args

    # Wrap out static args so XLA only sees array inputs
    def two_arg_step(ma, b):
        return fn(ma, model_static, optimizer, opt_state, b, state)

    # Lower to HLO via JAX AOT API
    hlo_comp = (
        jax.jit(two_arg_step)
           .lower(model_arr, batch)
           .compiler_ir('hlo')
    )
    hlo_text = hlo_comp.as_hlo_text()

    # Parse HLO text into a GraphDef
    gd = _hlo_text_to_graphdef(hlo_text)

    writer = EventFileWriter(logdir)

    # Session-start event to activate Graph plugin in TensorBoard
    sess_ev = event_pb2.Event()
    sess_ev.session_log.status = event_pb2.SessionLog.START
    writer.add_event(sess_ev)

    # The GraphDef event (no need for CopyFrom if gd is already a GraphDef)
    graph_ev = event_pb2.Event()
    graph_ev.graph_def = gd.SerializeToString()  # Direct assignment instead of using CopyFrom
    graph_ev.step = step
    writer.add_event(graph_ev)

# Write the event to disk
    writer.flush()
    writer.close()
