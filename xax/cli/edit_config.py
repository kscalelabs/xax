"""Lets you edit a checkpoint config programmatically."""

import argparse
import difflib
import io
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from xax.task.mixins.checkpointing import load_ckpt
from xax.utils.text import colored, show_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=Path)
    args = parser.parse_args()

    # Loads the config from the checkpoint.
    config = load_ckpt(args.ckpt_path, part="config")
    config_str = OmegaConf.to_yaml(config)

    # Opens the user's preferred editor to edit the config.
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
        f.write(config_str.encode("utf-8"))
        f.flush()
        subprocess.run([os.environ.get("EDITOR", "vim"), f.name], check=True)

    # Loads the edited config.
    try:
        edited_config = OmegaConf.load(f.name)
        edited_config_str = OmegaConf.to_yaml(edited_config, sort_keys=True)
    finally:
        os.remove(f.name)

    if edited_config_str == config_str:
        show_info("No changes were made to the config.")
        return

    # Diffs the original and edited configs.
    diff = difflib.ndiff(config_str.splitlines(), edited_config_str.splitlines())
    for line in diff:
        if line.startswith("+ "):
            print(colored(line, "light-green"), flush=True)
        elif line.startswith("- "):
            print(colored(line, "light-red"), flush=True)
        elif line.startswith("? "):
            print(colored(line, "light-cyan"), flush=True)

    # Saves the edited config to the checkpoint.
    with tarfile.open(args.ckpt_path, "w:gz") as tar:

        def add_file_bytes(name: str, data: bytes) -> None:  # noqa: ANN401
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        add_file_bytes("config", edited_config_str.encode())


if __name__ == "__main__":
    # python -m xax.cli.edit_config
    main()
