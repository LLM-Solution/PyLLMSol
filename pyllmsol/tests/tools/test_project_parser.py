#!/usr/bin/env python3
# coding: utf-8
# @Author: ArthurBernard
# @Email: arthur.bernard.92@gmail.com
# @Date: 2024-12-20 09:17:29
# @Last modified by: ArthurBernard
# @Last modified time: 2024-12-20 09:58:57
# @File path: ./pyllmsol/tests/tools/test_project_parser.py
# @Project: PyLLMSol

""" Test project parser objects.. """

# Built-in packages
from pathlib import Path

# Third party packages
import pytest

# Local packages
from pyllmsol.tools.project_parser import FileParser, PythonParser, ProjectParser

__all__ = []


@pytest.fixture
def tmp_project(tmp_path):
    (tmp_path / "README.md").write_text("# Project Title\n\nThis is a test project.", encoding="utf-8")
    (tmp_path / "main.py").write_text("# Main script\n\ndef hello():\n    return 'Hello, World!'\n", encoding="utf-8")
    (tmp_path / "config.json").write_text('{"key": "value"}', encoding="utf-8")
    (tmp_path / "empty_folder").mkdir()
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested/helper.py").write_text("# Helper script\n\ndef helper():\n    pass\n", encoding="utf-8")
    (tmp_path / "nested/ignored.txt").write_text("This should be ignored.", encoding="utf-8")

    return tmp_path


@pytest.fixture
def file_parser(tmp_project):
    readme_path = tmp_project / "README.md"
    return FileParser(readme_path)


@pytest.fixture
def python_parser(tmp_project):
    python_path = tmp_project / "main.py"
    return PythonParser(python_path)


@pytest.fixture
def project_parser(tmp_project):
    return ProjectParser(tmp_project)


def test_file_parser_init(tmp_project, file_parser):
    with pytest.raises(FileNotFoundError):
        FileParser(tmp_project / "NotExistingFile.md")

    assert file_parser.path == tmp_project / "README.md"
    assert file_parser.content == (tmp_project / "README.md").read_text(encoding='utf-8')


def test_file_parser_parse(tmp_project, file_parser):   
    parsed_data = file_parser.parse()
    assert parsed_data["metadata"]["path"] == str(tmp_project / "README.md")
    assert "# Project Title" in parsed_data["content"]
    assert parsed_data["content"] == (tmp_project / "README.md").read_text(encoding='utf-8')


def test_file_parser_call(file_parser):
    # Vérifier l'appel direct
    formatted = file_parser(header=True)
    assert "# Project Title" in formatted
    assert "### Header" in formatted
    assert "### Content" in formatted
    formatted = file_parser(header=False)
    assert "### Header" not in formatted


def test_python_parser_init(tmp_project, python_parser):
    with pytest.raises(FileNotFoundError):
        PythonParser(tmp_project / "NotExistingFile.py")

    with pytest.raises(ValueError):
        PythonParser(tmp_project / "README.md")

    assert python_parser.path == tmp_project / "main.py"
    assert python_parser.content == (tmp_project / "main.py").read_text(encoding='utf-8')


def test_python_parser_parse(tmp_project, python_parser):   
    parsed_data = python_parser.parse()
    assert parsed_data["metadata"]["path"] == str(tmp_project / "main.py")
    assert "# Main script" in parsed_data["content"]
    assert parsed_data["content"] == (tmp_project / "main.py").read_text(encoding='utf-8')


def test_python_parser_call(python_parser):
    # Vérifier l'appel direct
    formatted = python_parser(header=True, remove_comments=False)
    assert "# Main script" in formatted
    assert "## Python script" in formatted
    assert "### Header" in formatted
    assert "### Content" in formatted
    formatted = python_parser(header=False, remove_comments=True)
    assert "### Header" not in formatted
    assert "# Main script" not in formatted


def test_project_parser_init(tmp_project, project_parser):
    with pytest.raises(FileNotFoundError):
        ProjectParser("/tmp/NotExistingFolder")

    assert project_parser.repo_path == tmp_project
    assert project_parser.include_patterns == ["README.md", "*.py"]


def test_project_parser_parse(tmp_project, project_parser):   
    parsed_data = project_parser.parse()
    assert parsed_data["project_name"] == tmp_project.name
    assert any("README.md" in file["metadata"]["path"] for file in parsed_data["files"])
    assert any("main.py" in file["metadata"]["path"] for file in parsed_data["files"])
    assert not any("ignored.txt" in file["metadata"]["path"] for file in parsed_data["files"])


def test_project_parser_tree(tmp_project, project_parser):
    """Test de la méthode tree dans ProjectParser."""
    tree = project_parser.tree(max_depth=None)
    assert "README.md" in tree
    assert "main.py" in tree
    assert "nested/" in tree
    assert "helper.py" in tree
    assert "empty_folder/" not in tree
    assert "ignored.txt" not in tree
    tree = project_parser.tree(max_depth=1)
    assert "README.md" in tree
    assert "main.py" in tree
    # assert "nested/" not in tree
    assert "helper.py" not in tree
    assert "empty_folder/" not in tree
    assert "ignored.txt" not in tree


def test_project_parser_call(project_parser):
    # Vérifier l'appel direct
    formatted = project_parser(header=True, remove_comments=False)
    assert "# Project:" in formatted
    assert "## README.md" in formatted
    assert "### Header" in formatted
    assert "### Content" in formatted
    assert "## Python script" in formatted
    assert "def hello():" in formatted
    assert "# Main script" in formatted

    formatted = project_parser(header=False, remove_comments=True)
    assert "# Project:" in formatted
    assert "## README.md" in formatted
    assert "### Header" not in formatted
    assert "## Python script" in formatted
    assert "def hello():" in formatted
    assert "# Main script" not in formatted


if __name__ == "__main__":
    pass