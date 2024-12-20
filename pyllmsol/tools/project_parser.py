#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-15 09:55:05
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-20 09:15:15
# @File path:./pyllmsol/tools/project_parser.py
# @Project: PyLLMSol

""" Parser objects. """

# Built-in packages
import fnmatch
from pathlib import Path
import re

# Third party packages

# Local packages

__all__ = []


class FileParser:
    """ Base class for parsing different parts of a project. """
    def __init__(self, path: str | Path):
        self.path = Path(path) if isinstance(path, str) else path

        if not self.path.exists():
            raise FileNotFoundError(f"File '{self.path}' does not exist.")

        self.content = self.path.read_text(encoding='utf-8')

    def parse(self) -> dict:
        """ Parse file. """
        return {"content": self.content, "metadata": {"path": str(self.path)}}

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
        super().__init__(path)
        if self.path.suffix != ".py":
            raise ValueError(f"File '{self.path}' is not a Python file.")

    def __call__(self, header: bool = True, remove_comments: bool = True) -> str:
        """ Reformat text file. """
        text = "## Python script\n"

        if header:
            text += f"### Header\npath: {self.path}\n"

        content = self.content

        if remove_comments:
            content = re.sub(r"(?m)^\s*#.*$", "", content)

        # Remove extra blank lines
        content = re.sub(r"\n\s*\n", "\n", content)

        text += f"### Content\n{content}"

        return text


class ProjectParser:
    """ Parser for projects containing multiple files. """

    def __init__(self, path: str | Path, include_patterns: list = None):
        self.repo_path = Path(path) if isinstance(path, str) else path
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Path '{self.repo_path}' does not exist.")

        # Default include patterns: README.md and Python files
        self.include_patterns = include_patterns or ["README.md", "*.py"]

    def _is_included(self, item: Path) -> bool:
        """ Check if a file matches the include patterns. """
        return any(fnmatch.fnmatch(item.name, p) for p in self.include_patterns)

    def _contains_included_files(self, directory: Path) -> bool:
        """ Check if a directory contains any file or directory matching patterns.

        """
        for item in directory.iterdir():
            if item.is_file() and self._is_included(item):
                return True

            if item.is_dir() and self._contains_included_files(item):
                return True

        return False

    def parse(self) -> dict:
        """Parse the project directory based on include patterns."""
        parsed_files = []
        for item in self.repo_path.rglob("*"):
            if item.is_file() and self._is_included(item):
                if item.name == "README.md":
                    parsed_files.append(FileParser(item).parse())

                elif item.suffix == ".py":
                    parsed_files.append(PythonParser(item).parse())

        return {
            "project_name": self.repo_path.name,
            "files": parsed_files,
        }

    def tree(self, max_depth: int = None, level: int = 0) -> str:
        """ Get the project directory tree, keeping only specific file types.

        """
        tree = ""
        if level == 0:
            tree += f"{self.repo_path.name}/\n"

        spacer = "  " * level
        for item in self.repo_path.iterdir():
            # Skip files/directories that are not included
            if item.is_file() and not self._is_included(item):
                continue

            if item.is_dir() and not self._contains_included_files(item):
                continue

            if item.is_dir():
                tree += f"{spacer}|-- {item.name}/\n"
                if max_depth is None or level < max_depth - 1:
                    pp = ProjectParser(item, self.include_patterns)
                    tree += pp.tree(max_depth, level + 1)

            elif item.is_file():
                tree += f"{spacer}|-- {item.name}\n"

        return tree

    def __call__(
        self,
        header: bool = True,
        remove_comments: bool = True,
    ) -> str:
        """ Format the entire project as text. """
        content = f"# Project: {self.repo_path.name}\n"

        for item in self.repo_path.rglob("*"):
            if item.is_file() and self._is_included(item):
                if item.name == "README.md":
                    content += "\n## README.md\n"
                    content += FileParser(item)(header=header)

                elif item.suffix == ".py":
                    content += PythonParser(item)(
                        header=header,
                        remove_comments=remove_comments
                    )

        return content


if __name__ == "__main__":
    # Set project parser
    project_path = Path("../PyLLMSol")
    print(f"Parse project at {project_path}")

    project_parser = ProjectParser(project_path)

    print(project_parser.tree())

    parsed_content = project_parser()

    print(f"Parsed project:\n\n{parsed_content[:200]}")
    print(f"...\n{parsed_content[-200:]}")
