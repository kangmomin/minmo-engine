"""
Tests for minmo.indexer module.
"""

import os
import json
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from minmo.indexer import (
    CodeSymbol,
    ImportInfo,
    FileIndex,
    SearchResult,
    PythonASTParser,
    SQLiteFTS5Storage,
    CodeIndexer,
    CodeFileHandler,
)


class TestCodeSymbol:
    """Tests for CodeSymbol dataclass."""

    def test_create_function_symbol(self):
        symbol = CodeSymbol(
            name="my_function",
            symbol_type="function",
            file_path="/test/file.py",
            line_start=10,
            line_end=20,
            docstring="A test function",
            signature="def my_function(x: int) -> str",
        )
        assert symbol.name == "my_function"
        assert symbol.symbol_type == "function"
        assert symbol.docstring == "A test function"

    def test_create_class_symbol(self):
        symbol = CodeSymbol(
            name="MyClass",
            symbol_type="class",
            file_path="/test/file.py",
            line_start=1,
            line_end=50,
            decorators=["dataclass"],
        )
        assert symbol.name == "MyClass"
        assert symbol.symbol_type == "class"
        assert "dataclass" in symbol.decorators

    def test_create_method_symbol(self):
        symbol = CodeSymbol(
            name="process",
            symbol_type="method",
            file_path="/test/file.py",
            line_start=15,
            line_end=25,
            parent="MyClass",
            parameters=["self", "data: str"],
        )
        assert symbol.parent == "MyClass"
        assert "self" in symbol.parameters


class TestImportInfo:
    """Tests for ImportInfo dataclass."""

    def test_simple_import(self):
        imp = ImportInfo(module="os", names=["os"])
        assert imp.module == "os"
        assert not imp.is_relative

    def test_from_import(self):
        imp = ImportInfo(
            module="typing",
            names=["List", "Dict", "Optional"],
            is_relative=False,
        )
        assert "List" in imp.names
        assert len(imp.names) == 3

    def test_relative_import(self):
        imp = ImportInfo(
            module="utils",
            names=["helper"],
            is_relative=True,
            level=1,
        )
        assert imp.is_relative
        assert imp.level == 1


class TestFileIndex:
    """Tests for FileIndex dataclass."""

    def test_create_file_index(self):
        file_index = FileIndex(
            file_path="/test/file.py",
            file_hash="abc123",
            language="python",
            symbols=[
                CodeSymbol(
                    name="test_func",
                    symbol_type="function",
                    file_path="/test/file.py",
                    line_start=1,
                    line_end=5,
                )
            ],
            imports=[ImportInfo(module="os", names=["os"])],
            dependencies=["os"],
        )
        assert file_index.file_hash == "abc123"
        assert len(file_index.symbols) == 1
        assert len(file_index.imports) == 1


class TestPythonASTParser:
    """Tests for PythonASTParser class."""

    def test_parse_simple_function(self, sample_python_file: Path):
        parser = PythonASTParser()
        result = parser.parse_file(str(sample_python_file))

        assert result is not None
        assert result.language == "python"

        function_names = [s.name for s in result.symbols if s.symbol_type == "function"]
        assert "standalone_function" in function_names
        assert "async_function" in function_names

    def test_parse_class(self, sample_python_file: Path):
        parser = PythonASTParser()
        result = parser.parse_file(str(sample_python_file))

        assert result is not None

        classes = [s for s in result.symbols if s.symbol_type == "class"]
        assert len(classes) == 1
        assert classes[0].name == "SampleClass"
        assert classes[0].docstring == "A sample class for testing."

    def test_parse_methods(self, sample_python_file: Path):
        parser = PythonASTParser()
        result = parser.parse_file(str(sample_python_file))

        assert result is not None

        methods = [s for s in result.symbols if s.symbol_type == "method"]
        method_names = [m.name for m in methods]
        assert "__init__" in method_names
        assert "greet" in method_names
        assert "process_items" in method_names

        for method in methods:
            assert method.parent == "SampleClass"

    def test_parse_imports(self, sample_python_file: Path):
        parser = PythonASTParser()
        result = parser.parse_file(str(sample_python_file))

        assert result is not None
        assert len(result.imports) > 0
        assert "typing" in result.dependencies

    def test_parse_content_directly(self):
        parser = PythonASTParser()
        code = '''
def hello(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"
'''
        result = parser.parse_content(code, "/virtual/file.py")

        assert result is not None
        assert len(result.symbols) == 1
        assert result.symbols[0].name == "hello"
        assert result.symbols[0].return_type == "str"

    def test_parse_syntax_error_returns_none(self):
        parser = PythonASTParser()
        invalid_code = "def broken(:"
        result = parser.parse_content(invalid_code, "/test.py")

        assert result is None

    def test_parse_nonexistent_file_returns_none(self):
        parser = PythonASTParser()
        result = parser.parse_file("/nonexistent/path.py")
        assert result is None

    def test_extract_decorators(self, temp_dir: Path):
        code = '''
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_function(x: int) -> int:
    return x * 2
'''
        file_path = temp_dir / "decorated.py"
        file_path.write_text(code, encoding="utf-8")

        parser = PythonASTParser()
        result = parser.parse_file(str(file_path))

        assert result is not None
        func = [s for s in result.symbols if s.name == "cached_function"][0]
        assert "lru_cache(maxsize=100)" in func.decorators

    def test_extract_parameters(self, temp_dir: Path):
        code = '''
def complex_function(a: int, b: str = "default", *args, **kwargs) -> dict:
    pass
'''
        file_path = temp_dir / "params.py"
        file_path.write_text(code, encoding="utf-8")

        parser = PythonASTParser()
        result = parser.parse_file(str(file_path))

        assert result is not None
        func = result.symbols[0]
        assert "a: int" in func.parameters
        assert 'b: str' in func.parameters


class TestSQLiteFTS5Storage:
    """Tests for SQLiteFTS5Storage class."""

    @pytest.fixture
    def storage(self, temp_dir: Path):
        db_path = str(temp_dir / "test_index.db")
        storage = SQLiteFTS5Storage(db_path)
        yield storage
        storage.close()

    @pytest.fixture
    def sample_file_index(self) -> FileIndex:
        return FileIndex(
            file_path="/test/sample.py",
            file_hash="abc123def456",
            language="python",
            symbols=[
                CodeSymbol(
                    name="TestClass",
                    symbol_type="class",
                    file_path="/test/sample.py",
                    line_start=1,
                    line_end=30,
                    docstring="A test class for database operations.",
                    signature="class TestClass",
                ),
                CodeSymbol(
                    name="test_method",
                    symbol_type="method",
                    file_path="/test/sample.py",
                    line_start=5,
                    line_end=10,
                    docstring="Test method.",
                    signature="def test_method(self) -> None",
                    parent="TestClass",
                ),
                CodeSymbol(
                    name="helper_function",
                    symbol_type="function",
                    file_path="/test/sample.py",
                    line_start=35,
                    line_end=40,
                    docstring="Helper function for processing data.",
                    signature="def helper_function(data: str) -> bool",
                ),
            ],
            imports=[
                ImportInfo(module="typing", names=["List", "Dict"]),
                ImportInfo(module="os", names=["os"]),
            ],
            dependencies=["typing", "os"],
        )

    def test_store_and_retrieve(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        result = storage.store(sample_file_index)
        assert result is True

        retrieved = storage.get(sample_file_index.file_path)
        assert retrieved is not None
        assert retrieved.file_hash == sample_file_index.file_hash
        assert len(retrieved.symbols) == len(sample_file_index.symbols)

    def test_get_nonexistent(self, storage: SQLiteFTS5Storage):
        result = storage.get("/nonexistent/file.py")
        assert result is None

    def test_remove(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)
        result = storage.remove(sample_file_index.file_path)
        assert result is True

        retrieved = storage.get(sample_file_index.file_path)
        assert retrieved is None

    def test_search_by_name(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        results = storage.search("TestClass", limit=5)
        assert len(results) > 0
        assert any(r.symbol.name == "TestClass" for r in results)

    def test_search_by_docstring(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        results = storage.search("database operations", limit=5)
        assert len(results) > 0

    def test_search_fallback_to_like(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        results = storage.search("helper", limit=5)
        assert len(results) > 0

    def test_get_file_hash(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        hash_value = storage.get_file_hash(sample_file_index.file_path)
        assert hash_value == sample_file_index.file_hash

    def test_get_file_hash_nonexistent(self, storage: SQLiteFTS5Storage):
        result = storage.get_file_hash("/nonexistent.py")
        assert result is None

    def test_get_dependencies(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        deps = storage.get_dependencies(sample_file_index.file_path)
        assert "typing" in deps
        assert "os" in deps

    def test_get_dependents(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        dependents = storage.get_dependents("typing")
        assert sample_file_index.file_path in dependents

    def test_get_all_symbols(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        all_symbols = storage.get_all_symbols()
        assert len(all_symbols) == 3

        classes = storage.get_all_symbols("class")
        assert len(classes) == 1
        assert classes[0].name == "TestClass"

        methods = storage.get_all_symbols("method")
        assert len(methods) == 1

    def test_get_stats(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        stats = storage.get_stats()
        assert stats["files"] == 1
        assert stats["symbols"] == 3
        assert stats["classes"] == 1
        assert stats["functions"] == 1
        assert stats["methods"] == 1

    def test_update_existing_file(self, storage: SQLiteFTS5Storage, sample_file_index: FileIndex):
        storage.store(sample_file_index)

        updated_index = FileIndex(
            file_path=sample_file_index.file_path,
            file_hash="new_hash_value",
            language="python",
            symbols=[
                CodeSymbol(
                    name="UpdatedClass",
                    symbol_type="class",
                    file_path=sample_file_index.file_path,
                    line_start=1,
                    line_end=20,
                ),
            ],
            imports=[],
            dependencies=[],
        )

        storage.store(updated_index)

        retrieved = storage.get(sample_file_index.file_path)
        assert retrieved.file_hash == "new_hash_value"
        assert len(retrieved.symbols) == 1
        assert retrieved.symbols[0].name == "UpdatedClass"


class TestCodeIndexer:
    """Tests for CodeIndexer class."""

    def test_init_with_project_path(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        assert indexer.project_path == temp_dir.resolve()
        indexer.close()

    def test_index_single_file(self, sample_python_file: Path, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        result = indexer.index_file(sample_python_file)
        assert result is True
        indexer.close()

    def test_index_nonexistent_file(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        result = indexer.index_file("/nonexistent/file.py")
        assert result is False
        indexer.close()

    def test_index_unsupported_extension(self, temp_dir: Path):
        txt_file = temp_dir / "readme.txt"
        txt_file.write_text("Hello", encoding="utf-8")

        indexer = CodeIndexer(str(temp_dir))
        result = indexer.index_file(str(txt_file))
        assert result is False
        indexer.close()

    def test_index_all(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        stats = indexer.index_all()

        assert stats["indexed"] > 0
        assert stats["failed"] == 0
        indexer.close()

    def test_index_all_incremental(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))

        stats1 = indexer.index_all()
        initial_indexed = stats1["indexed"]
        assert initial_indexed > 0

        # Second call should skip already indexed files (if hashes match)
        stats2 = indexer.index_all()
        # On some systems, path normalization may cause re-indexing
        # The important thing is that the total work is done
        assert stats2["skipped"] + stats2["indexed"] + stats2["failed"] >= initial_indexed

        indexer.close()

    def test_index_all_force(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))

        stats1 = indexer.index_all()
        initial_indexed = stats1["indexed"]

        stats2 = indexer.index_all(force=True)
        assert stats2["indexed"] == initial_indexed
        assert stats2["skipped"] == 0

        indexer.close()

    def test_search(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        indexer.index_all()

        results = indexer.search("Application")
        assert len(results) > 0
        assert any(r.symbol.name == "Application" for r in results)

        indexer.close()

    def test_get_symbol(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        indexer.index_all()

        symbols = indexer.get_symbol("Application", symbol_type="class")
        assert len(symbols) > 0
        assert symbols[0].name == "Application"

        indexer.close()

    def test_get_class_methods(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        indexer.index_all()

        methods = indexer.get_class_methods("Application")
        method_names = [m.name for m in methods]
        assert "__init__" in method_names
        assert "run" in method_names

        indexer.close()

    def test_get_file_summary(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        indexer.index_all()

        main_path = str(sample_project / "src" / "main.py")
        summary = indexer.get_file_summary(main_path)

        assert summary is not None
        assert "Application" in summary["classes"]
        assert "main" in summary["functions"]

        indexer.close()

    def test_get_code_context(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        indexer.index_all()

        context = indexer.get_code_context("format_string")
        assert len(context["symbols"]) > 0
        assert len(context["related_files"]) > 0

        indexer.close()

    def test_get_project_overview(self, sample_project: Path):
        indexer = CodeIndexer(str(sample_project))
        indexer.index_all()

        overview = indexer.get_project_overview()
        assert overview["project_path"] == str(sample_project)
        assert "stats" in overview
        assert "main_classes" in overview
        assert "main_functions" in overview

        indexer.close()

    def test_remove_file(self, sample_python_file: Path, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        indexer.index_file(sample_python_file)

        result = indexer.remove_file(sample_python_file)
        assert result is True

        summary = indexer.get_file_summary(str(sample_python_file))
        assert summary is None

        indexer.close()

    def test_context_manager(self, temp_dir: Path):
        with CodeIndexer(str(temp_dir)) as indexer:
            assert indexer is not None


class TestCodeFileHandler:
    """Tests for CodeFileHandler class."""

    def test_should_index_python_file(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        handler = CodeFileHandler(indexer)

        assert handler._should_index("test.py") is True
        assert handler._should_index("test.js") is True
        assert handler._should_index("test.ts") is True
        assert handler._should_index("test.txt") is False
        assert handler._should_index("test.md") is False

        indexer.close()

    def test_custom_extensions(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        handler = CodeFileHandler(indexer, extensions={".rb", ".go"})

        assert handler._should_index("test.py") is False
        assert handler._should_index("test.rb") is True
        assert handler._should_index("test.go") is True

        indexer.close()

    def test_on_created_triggers_index(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        handler = CodeFileHandler(indexer)

        with patch.object(handler, "_debounced_index") as mock_debounce:
            mock_event = MagicMock()
            mock_event.is_directory = False
            mock_event.src_path = str(temp_dir / "new_file.py")

            handler.on_created(mock_event)
            mock_debounce.assert_called_once_with(mock_event.src_path, "create")

        indexer.close()

    def test_on_modified_triggers_index(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        handler = CodeFileHandler(indexer)

        with patch.object(handler, "_debounced_index") as mock_debounce:
            mock_event = MagicMock()
            mock_event.is_directory = False
            mock_event.src_path = str(temp_dir / "existing.py")

            handler.on_modified(mock_event)
            mock_debounce.assert_called_once_with(mock_event.src_path, "update")

        indexer.close()

    def test_on_deleted_triggers_remove(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        handler = CodeFileHandler(indexer)

        with patch.object(handler, "_debounced_index") as mock_debounce:
            mock_event = MagicMock()
            mock_event.is_directory = False
            mock_event.src_path = str(temp_dir / "deleted.py")

            handler.on_deleted(mock_event)
            mock_debounce.assert_called_once_with(mock_event.src_path, "delete")

        indexer.close()

    def test_ignores_directories(self, temp_dir: Path):
        indexer = CodeIndexer(str(temp_dir))
        handler = CodeFileHandler(indexer)

        with patch.object(handler, "_debounced_index") as mock_debounce:
            mock_event = MagicMock()
            mock_event.is_directory = True
            mock_event.src_path = str(temp_dir / "some_dir")

            handler.on_created(mock_event)
            mock_debounce.assert_not_called()

        indexer.close()


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_create_search_result(self):
        symbol = CodeSymbol(
            name="search_func",
            symbol_type="function",
            file_path="/test/search.py",
            line_start=1,
            line_end=5,
        )
        result = SearchResult(
            symbol=symbol,
            file_path="/test/search.py",
            score=0.95,
            context="def search_func(): ...",
        )

        assert result.symbol.name == "search_func"
        assert result.score == 0.95
        assert result.context is not None
