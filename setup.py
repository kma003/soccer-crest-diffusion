import os
from setuptools import setup, find_packages

# Set up data directories
data_dirs = [
    "data/"
    "data/crests"
]

for d in data_dirs:
    os.makedirs(d, exist_ok=True)

requirements = [
    "pandas",
    "torch",
    "torchvision",
    "tqdm",
    "wandb",
]

setup(
    name='soccer_crests_diffusion',
    version='0.1',
    packages=find_packages(),
)