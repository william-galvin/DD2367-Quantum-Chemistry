from setuptools import setup

setup(
    name="vcvqe",
    version="0.1.0",
    packages=["vcvqe"],
    entry_points={
        "console_scripts":
        ["vcvqe = vcvqe.main:main"]
    },
)
