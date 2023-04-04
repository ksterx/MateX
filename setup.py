from pathlib import Path

from setuptools import find_packages, setup

root_dir = Path(__file__).parent.resolve()

with open(root_dir / "docker" / "requirements.txt") as f:
    requirements = [line.rstrip() for line in f.readlines()]

with open(root_dir / "src" / "version.txt") as f:
    version = f.read().strip()

setup(
    name="matex",
    version=version,
    author="ksterx",
    description="A collection of reinforcement learning algorithms",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=requirements,
)
