"""Updates the API name map from the module imports."""

import argparse
import inspect
import os
import re
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip())
    parser.add_argument("-i", "--inplace", action="store_true")
    args = parser.parse_args()

    fpath = Path(__file__).parent.parent / "xax" / "__init__.py"
    if not fpath.exists():
        raise ValueError(f"File not found: {fpath.resolve()}")
    root_dir = fpath.parent

    os.environ["XAX_IMPORT_ALL"] = "1"

    import xax

    location_map = {}
    for mod in dir(xax):
        if mod.startswith("_"):
            continue
        try:
            location = Path(inspect.getfile(getattr(xax, mod)))
            if location.name == "__init__.py":
                continue
            try:
                relative_path = location.relative_to(root_dir)
                import_line = ".".join(relative_path.parts)
                assert import_line.endswith(".py")
                import_line = import_line[: -len(".py")]
                location_map[mod] = import_line
            except Exception:
                continue
        except Exception:
            continue

    # Sorts by module name, then object name.
    locations = [(k, v) for v, k in sorted([(v, k) for k, v in location_map.items()])]

    with open(fpath, "r") as f:
        lines = f.read()

    # Swaps the `__all__` items.
    all_lines = [f'\n    "{location}",' for location, _ in locations]
    new_all = "__all__ = [" + "".join(all_lines) + "\n]"
    lines = re.sub(r"__all__ = \[.+?\]", new_all, lines, flags=re.DOTALL | re.MULTILINE)

    # Swaps the `NAME_MAP` items.
    name_map_lines = [f'\n    "{k}": "{v}",' for k, v in locations]
    new_name_map = "NAME_MAP: dict[str, str] = {" + "".join(name_map_lines) + "\n}"
    lines = re.sub(r"NAME_MAP: dict\[str, str\] = \{.+?\}", new_name_map, lines, flags=re.DOTALL | re.MULTILINE)

    if args.inplace:
        with open(fpath, "w") as f:
            f.write(lines)
    else:
        sys.stdout.write(lines)
        sys.stdout.flush()


if __name__ == "__main__":
    # python -m scripts.update_api
    main()
