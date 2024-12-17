#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-15 09:55:05
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-17 16:41:04
# @File path:./pyllmsol/tools/project_parser.py
# @Project: PyLLMSol

""" Parser objects. """

# Built-in packages
# import os
from pathlib import Path
# import re

# Third party packages

# Local packages
from pyllmsol.tools.code_summarizer import download_github_repo

__all__ = []


class FileParser:
    """ Base class for parsing different parts of a project. """
    def __init__(self, path: str | Path):
        self.path = Path(path) if isinstance(path, str) else path
        self.content = self.path.read_text(encoding='utf-8')

    def parse(self):
        """ Parse code. """
        raise NotImplementedError("Subclasses must implement the parse method.")

    def __call__(self, header: bool = True):
        """ Reformat text file. """
        text = ""

        if header:
            text += f"## Header\npath: {self.path}\n"

        text += f"##Content\n{self.content}"


# class ProjectParser:
#     """ Base class for parsing different parts of a project. """
#     def __init__(self, path: str | Path):
#         self.repo_path = Path(path) if isinstance(path, str) else path

#     def parse(self):
#         """ Parse code. """
#         raise NotImplementedError("Subclasses must implement the parse method.")


# class ReadmeParser(ProjectParser):
#     """ Parser for extracting and standardizing README content. """
#     def parse(self):
#         """ Parse README.md file. """
#         readme_path = self.repo_path / "README.md"

#         if readme_path.exists():
#             content = readme_path.read_text(encoding='utf-8')

#             return self.reformat(content, "README.md")

#         return "No README.md found.\n"

#     @staticmethod
#     def reformat(content: str, file_name: str):
#         """ Reformats README content. """
#         return f"### {file_name} Content\n{content.strip()}\n"


# class PythonCodeParser(ProjectParser):
#     """Parser for extracting and standardizing Python code content."""
#     def parse(self):
#         python_files_content = "### Python Files Content\n"
#         for py_file in self.repo_path.rglob("*.py"):
#             content = py_file.read_text(encoding='utf-8')
#             formatted_content = self.reformat(content, py_file.relative_to(self.repo_path))
#             python_files_content += formatted_content
#         return python_files_content

#     @staticmethod
#     def reformat(content, file_name):
#         """Reformats Python code content."""
#         # Remove excessive empty lines and comments for readability
#         cleaned_content = re.sub(r"(?m)^\s*#.*$", "", content)  # Remove comments
#         cleaned_content = re.sub(r"\n\s*\n", "\n", cleaned_content)  # Remove extra blank lines
#         return f"\n#### {file_name}\n{cleaned_content.strip()}\n"


# class SummaryGenerator:
#     """Generates a standardized summary of the project."""
#     def __init__(self, parsers):
#         self.parsers = parsers

#     def generate(self):
#         summary = "### Project Summary\n"
#         for parser in self.parsers:
#             summary += parser.parse()
#         return summary

#     def save(self, output_file, summary):
#         with open(output_file, 'w', encoding='utf-8') as f:
#             f.write(summary)
#         print(f"Summary file created: {output_file}")


if __name__ == "__main__":
    # Clone repository
    repo_url = input("Enter the GitHub repository URL: ")
    repo_path = download_github_repo(repo_url)

    # Set up parsers
    # parsers = [
    #     ReadmeParser(repo_path),
    #     PythonCodeParser(repo_path),
    # ]

    # # Generate summary
    # generator = SummaryGenerator(parsers)
    # summary = generator.generate()

    # # Save summary to file
    # generator.save("summary.txt", summary)
