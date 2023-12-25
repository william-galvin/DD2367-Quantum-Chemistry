from setuptools import setup

setup(
    name="vqe-pennylane",
    version="0.1.0",
    packages=["src"],
    entry_points={
        "console_scripts":
        ["vqe-pennylane = src.main:main"]
    },
)
