#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()

with open("xax/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()

with open("xax/requirements-rl.txt", "r", encoding="utf-8") as f:
    requirements_rl: list[str] = f.read().splitlines()

with open("xax/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()

with open("xax/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in xax/__init__.py"
version: str = version_re.group(1)


setup(
    name="xax",
    version=version,
    description="The xax project",
    author="Benjamin Bolte",
    url="https://github.com/dpshai/xax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={
        "rl": requirements_rl,
        "dev": requirements_dev,
    },
    package_data={
        "xax": [
            "py.typed",
            "requirements*.txt",
        ],
    },
)
