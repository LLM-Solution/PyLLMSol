#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-14 08:54:22
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-14 09:03:28
# @File path: ./pyllmsol/tools/code_summarizer.py
# @Project: None

""" Script to summarise code project. """

# Built-in packages
import os
from pathlib import Path
import requests

# Third party packages

# Local packages

__all__ = []


def download_github_repo(repo_url: str, dest_folder: str = "repo"):
    """ Clones a GitHub repository to a local directory. """
    if not repo_url.endswith('.git'):
        repo_url += '.git'

    os.system(f"git clone {repo_url} {dest_folder}")

    return Path(dest_folder)


def parse_readme(repo_path: str):
    """ Extracts and returns the content of the README.md file. """
    readme_path = repo_path / "README.md"
    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f"### README.md Content\n{f.read()}\n"

    return "No README.md found.\n"


def parse_python_files(repo_path):
    """Lists all Python files and extracts their content."""
    python_files_content = "### Python Files Content\n"

    for py_file in repo_path.rglob("*.py"):
        with open(py_file, 'r', encoding='utf-8') as f:
            file_content = f.read()
            title = f"#### {py_file.relative_to(repo_path)}"
            python_files_content += f"\n{title}\n{file_content}\n"

    return python_files_content


def create_summary_file(repo_path: str, output_file: str = "summary.txt"):
    """Creates a summary file with README and Python files content."""
    readme_content = parse_readme(repo_path)
    python_files_content = parse_python_files(repo_path)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
        f.write(python_files_content)

    print(f"Summary file created: {output_file}")


if __name__ == "__main__":
    # repo_url = input("Enter the GitHub repository URL: ")
    # repo_path = download_github_repo(repo_url)
    output_file = input("Enter the path of output: ")
    repo_path = Path(".")
    create_summary_file(repo_path, output_file)
