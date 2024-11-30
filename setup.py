#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-10-29 15:54:21
# @Last modified by: ArthurBernard
# @Last modified time: 2024-11-30 10:34:21

""" Setup PyLLMSol package. """

# Built-in packages
from setuptools import setup, find_packages

# Third party packages

# Local packages

__all__ = []


if __name__ == "__main__":
    with open("README.md", encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name="pyllmsol",
        version="0.1.0",
        author="Arthur Bernard",
        author_email="arthur.bernard.92@gmail.com",
        description="A package for large language model solutions",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/LLM-Solution/PyLLMSol",
        packages=find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires=">=3.6",
        install_requires=[
            'torch',
            'transformers',
        ],
    )
