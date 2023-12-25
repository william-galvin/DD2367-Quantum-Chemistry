from setuptools import setup

setup(
    name="vcvqe",
    version="0.1.0",
    packages=["src"],
    entry_points={
        "console_scripts":
        ["vcvqe = src.main:main"]
    },
)