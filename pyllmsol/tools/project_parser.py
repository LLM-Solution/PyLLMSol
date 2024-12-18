#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-15 09:55:05
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-18 17:16:10
# @File path:./pyllmsol/tools/project_parser.py
# @Project: PyLLMSol

""" Parser objects. """

# Built-in packages
from pathlib import Path
import re

# Third party packages

# Local packages
from pyllmsol.tools.code_summarizer import download_github_repo

__all__ = []


class FileParser:
    """ Base class for parsing different parts of a project. """
    def __init__(self, path: str | Path):
        self.path = Path(path) if isinstance(path, str) else path
        self.content = self.path.read_text(encoding='utf-8')

    def parse(self) -> dict:
        """ Parse file. """
        return {"content": self.content, "metadata": {"path": self.path}}

    def __call__(self, header: bool = True) -> str:
        """ Reformat text file. """
        text = ""

        if header:
            text += f"### Header\npath: {self.path}\n"

        text += f"### Content\n{self.content}"

        return text


class PythonParser(FileParser):
    """ Class to parse python code. """
    def __init__(self, path: str | Path):
        self.path = Path(path) if isinstance(path, str) else path

        if self.path.suffix != ".py":
            raise ValueError("File must be a Python code.")

        self.content = self.path.read_text(encoding='utf-8')

    def __call__(self, header: bool = True, rm_comments: bool = False) -> str:
        """ Reformat text file. """
        text = "## Python script\n"

        if header:
            text += f"### Header\npath: {self.path}\n"

        # Remove extra blank lines
        content = re.sub(r"\n\s*\n", "\n", self.content)

        if rm_comments:
            # Remove comments
            content = re.sub(r"(?m)^\s*#.*$", "", content)

        text += f"### Content\n{content}"

        return text


class ProjectParser:
    """ Base class for parsing different parts of a project. """
    def __init__(self, path: str | Path):
        self.repo_path = Path(path) if isinstance(path, str) else path

    def parse(self):
        """ Parse project. """
        parsed_readme = FileParser(self.repo_path / "README.md").parse()
        parsed_python_code = [
            PythonParser(f).parse() for f in self.repo_path.rglob("*.py")
        ]

        return {
            "name": self.repo_path.name,
            "content": [parsed_readme] + parsed_python_code,
        }

    def __call__(self, header: bool = True, rm_comments: bool = False) -> str:
        """ Reformat text file. """
        content = f"# Parsed project '{self.repo_path.name}'\n"
        readme_path = self.repo_path / "README.md"

        if readme_path.exists():
            readme_parser = FileParser(readme_path)
            content += "## README.md file\n"
            content += readme_parser(header=header)

        for py_file in self.repo_path.rglob("*.py"):
            python_parser = PythonParser(py_file)
            content += python_parser(header=header, rm_comments=rm_comments)

        return content

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
