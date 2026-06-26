"""HC-MARL: Human-Centric Multi-Agent Reinforcement Learning."""

from setuptools import setup, find_packages

setup(
    name="hcmarl",
    version="0.1.0",
    author="Aditya Maiti, Amrit Pal Singh, Amar Arora, Arshpreet Kaur",
    author_email="aditya.03819051622@ipu.ac.in",
    description=(
        "Human-Centric Multi-Agent Reinforcement Learning for Safe and Fair "
        "Human-Robot Collaboration in Warehouse Environments"
    ),
    packages=find_packages(exclude=["tests", "tests.*", "venv", "venv.*"]),
    python_requires=">=3.10,<3.14",
    install_requires=[
        "numpy>=1.24,<2.0",
        "scipy>=1.10,<2.0",
        "cvxpy>=1.4,<2.0",
        "osqp>=0.6.3,<1.0",
        "pyyaml>=6.0,<7.0",
        "torch>=2.6.0,<2.7",
        "gymnasium>=0.29,<1.0",
        "pandas>=2.0,<3.0",
        "matplotlib>=3.7,<4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4,<9.0",
            "pytest-cov>=4.1,<6.0",
        ],
        "wandb": [
            "wandb>=0.16,<1.0",
        ],
    },
)
