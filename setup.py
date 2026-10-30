"""Package code from src/ and the explicitly listed inputs from artifacts/."""

from setuptools import find_packages, setup

setup(
    packages=[*find_packages("src"), "mubind.artifacts"],
    package_dir={"": "src", "mubind.artifacts": "artifacts"},
)
