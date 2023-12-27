from setuptools import setup

setup(
    name="vqe-pennylane",
    version="0.1.0",
    packages=["vqe_pennylane"],
    entry_points={
        "console_scripts":
        ["vqe-pennylane = vqe_pennylane.main:main"]
    },
)
