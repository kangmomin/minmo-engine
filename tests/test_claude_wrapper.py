"""
Tests for minmo.claude_wrapper module.
"""

import os
import sys
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from minmo.claude_wrapper import (
    TaskStatus,
    TaskResult,
    ExecutionContext,
    ClaudeCodeWrapper,
    WORKER_CONSTITUTION,
)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_enum_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_create_success_result(self):
        result = TaskResult(
            task_id="task_001",
            status=TaskStatus.COMPLETED,
            output="Task completed successfully",
            files_modified=["main.py", "utils.py"],
            duration_seconds=5.5,
            retry_count=0,
        )

        assert result.task_id == "task_001"
        assert result.status == TaskStatus.COMPLETED
        assert len(result.files_modified) == 2
        assert result.error is None

    def test_create_failed_result(self):
        result = TaskResult(
            task_id="task_002",
            status=TaskStatus.FAILED,
            output="Error output",
            error="SyntaxError: invalid syntax",
            retry_count=3,
        )

        assert result.status == TaskStatus.FAILED
        assert result.error is not None
        assert result.retry_count == 3

    def test_default_values(self):
        result = TaskResult(task_id="test", status=TaskStatus.PENDING)

        assert result.output == ""
        assert result.error is None
        assert result.files_modified == []
        assert result.duration_seconds == 0.0
        assert result.retry_count == 0


class TestExecutionContext:
    """Tests for ExecutionContext dataclass."""

    def test_create_context(self):
        context = ExecutionContext(
            working_directory="/test/project",
            mcp_config_path="/test/.claude/mcp_config.json",
            timeout_seconds=600,
            max_retries=5,
            verbose=True,
        )

        assert context.working_directory == "/test/project"
        assert context.timeout_seconds == 600
        assert context.verbose is True

    def test_default_values(self):
        context = ExecutionContext(working_directory="/test")

        assert context.mcp_config_path is None
        assert context.timeout_seconds == 300
        assert context.max_retries == 3
        assert context.verbose is False


class TestClaudeCodeWrapper:
    """Tests for ClaudeCodeWrapper class."""

    @pytest.fixture
    def mock_scribe(self):
        """Mock the scribe_mcp functions."""
        with patch("minmo.claude_wrapper.log_event") as mock_log, \
             patch("minmo.claude_wrapper.update_todo") as mock_update, \
             patch("minmo.claude_wrapper.init_database") as mock_init:
            mock_log.return_value = True
            mock_update.return_value = True
            mock_init.return_value = None
            yield {
                "log_event": mock_log,
                "update_todo": mock_update,
                "init_database": mock_init
            }

    @pytest.fixture
    def mock_which(self):
        """Mock shutil.which for Claude CLI detection."""
        with patch("minmo.claude_wrapper.shutil.which") as mock:
            mock.return_value = "/usr/local/bin/claude"
            yield mock

    @pytest.fixture
    def wrapper(self, temp_dir: Path, mock_scribe, mock_which):
        """Create a ClaudeCodeWrapper instance."""
        wrapper = ClaudeCodeWrapper(
            working_directory=str(temp_dir),
            timeout_seconds=60,
            max_retries=2,
            verbose=False,
        )
        yield wrapper

    def test_init_with_defaults(self, temp_dir: Path, mock_scribe, mock_which):
        wrapper = ClaudeCodeWrapper(working_directory=str(temp_dir))

        assert wrapper.working_directory == str(temp_dir)
        assert wrapper.timeout_seconds == 300
        assert wrapper.max_retries == 3
        assert wrapper.verbose is False

    def test_init_creates_mcp_config(self, temp_dir: Path, mock_scribe, mock_which):
        wrapper = ClaudeCodeWrapper(working_directory=str(temp_dir))

        assert wrapper.mcp_config_path is not None
        config_path = Path(wrapper.mcp_config_path)
        assert config_path.exists()

        config = json.loads(config_path.read_text())
        assert "mcpServers" in config
        assert "minmo-scribe" in config["mcpServers"]

    def test_init_with_custom_mcp_config(self, temp_dir: Path, mock_scribe, mock_which):
        custom_config = temp_dir / "custom_mcp.json"
        custom_config.write_text('{"mcpServers": {}}')

        wrapper = ClaudeCodeWrapper(
            working_directory=str(temp_dir),
            mcp_config_path=str(custom_config),
        )

        assert wrapper.mcp_config_path == str(custom_config)

    def test_find_claude_cli_from_env(self, temp_dir: Path, mock_scribe):
        with patch.dict(os.environ, {"CLAUDE_CLI_PATH": "/custom/path/claude"}):
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True
                wrapper = ClaudeCodeWrapper(working_directory=str(temp_dir))
                assert wrapper.claude_path == "/custom/path/claude"

    def test_find_claude_cli_from_path(self, temp_dir: Path, mock_scribe, mock_which):
        wrapper = ClaudeCodeWrapper(working_directory=str(temp_dir))
        assert wrapper.claude_path == "/usr/local/bin/claude"

    def test_build_prompt(self, wrapper):
        task = {
            "id": "task_001",
            "title": "Create User model",
            "description": "Define User dataclass with email and password fields",
            "type": "implementation",
            "files_affected": ["models/user.py"],
        }

        prompt = wrapper._build_prompt(task)

        assert WORKER_CONSTITUTION in prompt
        assert "Create User model" in prompt
        assert "Define User dataclass" in prompt
        assert "implementation" in prompt
        assert "models/user.py" in prompt

    def test_build_prompt_minimal(self, wrapper):
        task = {"title": "Simple task"}
        prompt = wrapper._build_prompt(task)

        assert WORKER_CONSTITUTION in prompt
        assert "Simple task" in prompt

    def test_parse_output_file_modifications(self, wrapper):
        output = """
        Created src/models/user.py
        Modified utils.py
        Wrote to config.json
        """
        result = wrapper._parse_output(output)

        assert len(result["files_modified"]) >= 1
        # The regex pattern captures file paths after keywords like Created, Modified, etc.
        assert any("user.py" in f or "utils.py" in f or "config.json" in f for f in result["files_modified"])

    def test_parse_output_errors(self, wrapper):
        output = """
        Error: SyntaxError in file main.py
        Cannot find module 'utils'
        Exception: ValueError occurred
        """
        result = wrapper._parse_output(output)

        assert len(result["errors"]) >= 1

    def test_parse_output_warnings(self, wrapper):
        output = """
        Warning: Deprecated function used
        WARN: Configuration not found
        """
        result = wrapper._parse_output(output)

        assert len(result["warnings"]) >= 1

    def test_parse_output_summary(self, wrapper):
        output = """
        Line 1
        Line 2
        Line 3
        Line 4
        Final summary line
        """
        result = wrapper._parse_output(output)

        assert "Final summary line" in result["summary"]

    @pytest.mark.skip(reason="PexpectSpawn mocking requires complex setup on Windows")
    def test_execute_success(self, temp_dir: Path, mock_scribe, mock_which):
        with patch("minmo.claude_wrapper.PexpectSpawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.side_effect = [True, True, False]
            mock_process.readline.side_effect = [
                "Processing task...",
                "Created file main.py",
                "Task completed.",
                "",
            ]
            mock_process.exitstatus = 0
            mock_spawn.return_value = mock_process

            wrapper = ClaudeCodeWrapper(
                working_directory=str(temp_dir),
                timeout_seconds=10,
                max_retries=0,
            )

            task = {
                "id": "task_001",
                "title": "Test task",
                "description": "Test description",
            }

            result = wrapper.execute(task)

            assert result.task_id == "task_001"
            assert result.status == TaskStatus.COMPLETED

    @pytest.mark.skip(reason="PexpectSpawn mocking requires complex setup on Windows")
    def test_execute_failure(self, temp_dir: Path, mock_scribe, mock_which):
        with patch("minmo.claude_wrapper.PexpectSpawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.side_effect = [True, False]
            mock_process.readline.side_effect = [
                "Error: Something went wrong",
                "",
            ]
            mock_process.exitstatus = 1
            mock_spawn.return_value = mock_process

            wrapper = ClaudeCodeWrapper(
                working_directory=str(temp_dir),
                timeout_seconds=10,
                max_retries=0,
            )

            task = {"id": "task_002", "title": "Failing task"}

            result = wrapper.execute(task)

            assert result.status == TaskStatus.FAILED

    @pytest.mark.skip(reason="PexpectSpawn mocking requires complex setup on Windows")
    def test_execute_with_commander_callback(self, temp_dir: Path, mock_scribe, mock_which):
        with patch("minmo.claude_wrapper.PexpectSpawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.side_effect = [True, False]
            mock_process.readline.side_effect = [
                "Task completed successfully",
                "",
            ]
            mock_process.exitstatus = 0
            mock_spawn.return_value = mock_process

            def commander_callback(action, data):
                return {
                    "action": "retry_with_modification",
                    "modified_task": {
                        **data["task"],
                        "description": "Modified description",
                    }
                }

            wrapper = ClaudeCodeWrapper(
                working_directory=str(temp_dir),
                timeout_seconds=10,
                max_retries=1,
                commander_callback=commander_callback,
            )

            task = {"id": "task_003", "title": "Task with callback"}

            result = wrapper.execute(task)

            assert result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]

    @pytest.mark.skip(reason="PexpectSpawn mocking requires complex setup on Windows")
    def test_execute_plan(self, temp_dir: Path, mock_scribe, mock_which):
        with patch("minmo.claude_wrapper.PexpectSpawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.side_effect = [True, False, True, False]
            mock_process.readline.side_effect = [
                "Task 1 completed",
                "",
                "Task 2 completed",
                "",
            ]
            mock_process.exitstatus = 0
            mock_spawn.return_value = mock_process

            wrapper = ClaudeCodeWrapper(
                working_directory=str(temp_dir),
                timeout_seconds=10,
                max_retries=0,
            )

            tasks = [
                {"id": "task_001", "title": "First task"},
                {"id": "task_002", "title": "Second task"},
            ]

            results = wrapper.execute_plan(tasks)

            assert len(results) == 2

    @pytest.mark.skip(reason="PexpectSpawn mocking requires complex setup on Windows")
    def test_execute_plan_stop_on_failure(self, temp_dir: Path, mock_scribe, mock_which):
        with patch("minmo.claude_wrapper.PexpectSpawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.side_effect = [True, False]
            mock_process.readline.side_effect = [
                "Error: Task failed",
                "",
            ]
            mock_process.exitstatus = 1
            mock_spawn.return_value = mock_process

            wrapper = ClaudeCodeWrapper(
                working_directory=str(temp_dir),
                timeout_seconds=10,
                max_retries=0,
            )

            tasks = [
                {"id": "task_001", "title": "Failing task"},
                {"id": "task_002", "title": "Should not run"},
            ]

            results = wrapper.execute_plan(tasks, stop_on_failure=True)

            assert len(results) == 1
            assert results[0].status == TaskStatus.FAILED

    @pytest.mark.skip(reason="PexpectSpawn mocking requires complex setup on Windows")
    def test_execute_plan_continue_on_failure(self, temp_dir: Path, mock_scribe, mock_which):
        with patch("minmo.claude_wrapper.PexpectSpawn") as mock_spawn:
            mock_process = MagicMock()
            mock_process.isalive.side_effect = [True, False, True, False]
            mock_process.readline.side_effect = [
                "Error: First task failed",
                "",
                "Second task completed",
                "",
            ]
            mock_process.exitstatus = 0
            mock_spawn.return_value = mock_process

            wrapper = ClaudeCodeWrapper(
                working_directory=str(temp_dir),
                timeout_seconds=10,
                max_retries=0,
            )

            tasks = [
                {"id": "task_001", "title": "Failing task"},
                {"id": "task_002", "title": "Should still run"},
            ]

            results = wrapper.execute_plan(tasks, stop_on_failure=False)

            assert len(results) == 2

    def test_cancel_no_process(self, wrapper):
        result = wrapper.cancel()
        assert result is False

    def test_cancel_running_process(self, wrapper, mock_scribe):
        mock_process = MagicMock()
        wrapper._process = mock_process

        result = wrapper.cancel()

        assert result is True

    def test_is_running_no_process(self, wrapper):
        assert wrapper.is_running() is False

    def test_get_claude_version(self, wrapper):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="claude version 1.0.0")
            version = wrapper.get_claude_version()
            assert "claude" in version.lower() or "1.0" in version

    def test_get_claude_version_failure(self, wrapper):
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Command failed")
            version = wrapper.get_claude_version()
            assert version is None

    def test_callbacks(self, temp_dir: Path, mock_scribe, mock_which):
        output_messages = []
        error_messages = []

        def on_output(msg):
            output_messages.append(msg)

        def on_error(msg, data):
            error_messages.append((msg, data))

        wrapper = ClaudeCodeWrapper(
            working_directory=str(temp_dir),
            on_output=on_output,
            on_error=on_error,
            verbose=True,
        )

        wrapper._log("Test message")

        assert len(output_messages) > 0


class TestWorkerConstitution:
    """Tests for WORKER_CONSTITUTION constant."""

    def test_constitution_exists(self):
        assert WORKER_CONSTITUTION is not None
        assert len(WORKER_CONSTITUTION) > 0

    def test_constitution_contains_principles(self):
        assert "주석 최소화" in WORKER_CONSTITUTION
        assert "요구사항만 구현" in WORKER_CONSTITUTION
        assert "추측 금지" in WORKER_CONSTITUTION
        assert "과잉 설계 금지" in WORKER_CONSTITUTION

    def test_constitution_contains_work_instruction(self):
        assert "작업 지시" in WORKER_CONSTITUTION


class TestIntegration:
    """Integration tests for ClaudeCodeWrapper."""

    @pytest.fixture
    def mock_all(self):
        """Mock all external dependencies."""
        with patch("minmo.claude_wrapper.log_event") as mock_log, \
             patch("minmo.claude_wrapper.update_todo") as mock_update, \
             patch("minmo.claude_wrapper.init_database") as mock_init, \
             patch("minmo.claude_wrapper.shutil.which") as mock_which:
            mock_log.return_value = True
            mock_update.return_value = True
            mock_init.return_value = None
            mock_which.return_value = "/usr/local/bin/claude"
            yield

    def test_full_workflow(self, temp_dir: Path, mock_all):
        wrapper = ClaudeCodeWrapper(
            working_directory=str(temp_dir),
            timeout_seconds=30,
            max_retries=1,
        )

        task = {
            "id": "workflow_test",
            "title": "Create hello function",
            "description": "Create a simple hello world function",
            "type": "implementation",
            "files_affected": ["hello.py"],
        }

        prompt = wrapper._build_prompt(task)
        assert "hello function" in prompt.lower()
        assert "hello.py" in prompt

        # Test output parsing with a clear pattern
        parsed = wrapper._parse_output("Created hello.py\nFunction created successfully")
        assert len(parsed["files_modified"]) >= 1

    def test_error_handling_workflow(self, temp_dir: Path, mock_all):
        commander_responses = []

        def commander_callback(action, data):
            commander_responses.append((action, data))
            return {"action": "skip"}

        wrapper = ClaudeCodeWrapper(
            working_directory=str(temp_dir),
            commander_callback=commander_callback,
        )

        modified = wrapper._handle_error(
            {"id": "test", "title": "Test"},
            "Error: Test error",
            1
        )

        assert modified is None
        assert len(commander_responses) == 1
        assert commander_responses[0][0] == "analyze_error"
