"""
Pytest fixtures for Minmo Engine tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, AsyncMock

import pytest


# Add minmo package to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    tmp = tempfile.mkdtemp(prefix="minmo_test_")
    yield Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def sample_python_file(temp_dir: Path) -> Path:
    """Create a sample Python file for testing."""
    file_path = temp_dir / "sample.py"
    file_path.write_text('''
"""Sample module for testing."""

from typing import List, Optional


class SampleClass:
    """A sample class for testing."""

    def __init__(self, name: str):
        """Initialize with name."""
        self.name = name

    def greet(self) -> str:
        """Return a greeting."""
        return f"Hello, {self.name}!"

    def process_items(self, items: List[str]) -> List[str]:
        """Process a list of items."""
        return [item.upper() for item in items]


def standalone_function(x: int, y: int) -> int:
    """Add two numbers."""
    return x + y


async def async_function(value: str) -> str:
    """An async function for testing."""
    return value.lower()


CONSTANT_VALUE = 42
''', encoding='utf-8')
    return file_path


@pytest.fixture
def sample_project(temp_dir: Path) -> Path:
    """Create a sample project structure for testing."""
    # Create directory structure
    (temp_dir / "src").mkdir()
    (temp_dir / "src" / "utils").mkdir()
    (temp_dir / "tests").mkdir()

    # Main module
    (temp_dir / "src" / "__init__.py").write_text('"""Source package."""\n', encoding='utf-8')
    (temp_dir / "src" / "main.py").write_text('''
"""Main application module."""

from src.utils.helpers import format_string


class Application:
    """Main application class."""

    def __init__(self, config: dict):
        self.config = config

    def run(self) -> None:
        """Run the application."""
        print("Running...")


def main():
    """Entry point."""
    app = Application({})
    app.run()
''', encoding='utf-8')

    # Utils module
    (temp_dir / "src" / "utils" / "__init__.py").write_text('"""Utils package."""\n', encoding='utf-8')
    (temp_dir / "src" / "utils" / "helpers.py").write_text('''
"""Helper utilities."""

from typing import Any


def format_string(value: str) -> str:
    """Format a string value."""
    return value.strip().title()


def validate_input(data: Any) -> bool:
    """Validate input data."""
    return data is not None
''', encoding='utf-8')

    # Test file
    (temp_dir / "tests" / "__init__.py").write_text('"""Tests package."""\n', encoding='utf-8')
    (temp_dir / "tests" / "test_main.py").write_text('''
"""Tests for main module."""

import pytest
from src.main import Application


def test_application_init():
    app = Application({"key": "value"})
    assert app.config == {"key": "value"}
''', encoding='utf-8')

    return temp_dir


@pytest.fixture
def mock_redis() -> MagicMock:
    """Create a mock Redis client."""
    mock = MagicMock()
    mock.get.return_value = None
    mock.set.return_value = True
    mock.hgetall.return_value = {}
    mock.hset.return_value = True
    mock.delete.return_value = True
    mock.keys.return_value = []
    mock.ping.return_value = True
    return mock


@pytest.fixture
def mock_gemini_response() -> dict:
    """Create a mock Gemini API response."""
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "This is a mock response from Gemini."
                }]
            }
        }]
    }


@pytest.fixture
def sample_task_plan() -> dict:
    """Create a sample task plan."""
    return {
        "goal": "Implement user authentication",
        "analysis": "Need to add login/logout functionality",
        "tasks": [
            {
                "id": "task_001",
                "title": "Create User model",
                "description": "Define User dataclass with email and password fields",
                "type": "implementation",
                "priority": 1,
                "dependencies": [],
                "estimated_complexity": "low"
            },
            {
                "id": "task_002",
                "title": "Implement login function",
                "description": "Create login function that validates credentials",
                "type": "implementation",
                "priority": 2,
                "dependencies": ["task_001"],
                "estimated_complexity": "medium"
            },
            {
                "id": "task_003",
                "title": "Write unit tests",
                "description": "Test login and User model",
                "type": "test",
                "priority": 3,
                "dependencies": ["task_001", "task_002"],
                "estimated_complexity": "low"
            }
        ],
        "success_criteria": [
            "User can log in with valid credentials",
            "Invalid credentials are rejected",
            "All tests pass"
        ]
    }


@pytest.fixture
def sample_task_result() -> dict:
    """Create a sample task result."""
    return {
        "task_id": "task_001",
        "status": "success",
        "output": "Created User model in src/models/user.py",
        "files_modified": ["src/models/user.py", "src/models/__init__.py"],
        "execution_time": 5.2,
        "error": None
    }


@pytest.fixture
def mock_sqlite_connection():
    """Create a mock SQLite connection."""
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def set_api_keys():
    """Set mock API keys for testing."""
    os.environ["GEMINI_API_KEY"] = "test-gemini-key"
    os.environ["ANTHROPIC_API_KEY"] = "test-anthropic-key"
    yield
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("ANTHROPIC_API_KEY", None)
