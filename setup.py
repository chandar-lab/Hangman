import os

import setuptools

with open("version.txt") as f:
    VERSION = f.read().strip()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hangman",
    version=VERSION,
    description="A simple working memory for building stateful LLM agents.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # url="https://github.com/chandar-lab/hangman",
    # project_urls={
    #     "Bug Tracker": "https://github.com/chandar-lab/hangman/issues",
    # },
    python_requires=">=3.9",
    # install_requires=[
    #     "numpy",
    #     "torch",
    # ],
    packages=['hangman']
)