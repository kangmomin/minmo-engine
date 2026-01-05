"""
Tests for minmo.orchestrator module.
"""

import json
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import pytest

from minmo.orchestrator import ScribeMCP, MinmoOrchestrator
from minmo.gemini_wrapper import TaskPlan, ProjectAnalysis, ProjectType
from minmo.claude_wrapper import TaskResult, TaskStatus


class TestScribeMCP:
    """Tests for ScribeMCP class."""

    @pytest.fixture
    def mock_scribe_functions(self):
        """Mock the scribe_mcp functions."""
        with patch("minmo.orchestrator.log_event") as mock_log, \
             patch("minmo.orchestrator.update_todo") as mock_update:
            mock_log.return_value = True
            mock_update.return_value = True
            yield {"log_event": mock_log, "update_todo": mock_update}

    def test_init(self):
        scribe = ScribeMCP()
        assert scribe._plan is None
        assert scribe._tasks == []
        assert scribe._current_index == 0

    def test_save_plan_with_task_plan(self, mock_scribe_functions):
        scribe = ScribeMCP()

        plan = TaskPlan(
            goal="Test goal",
            tasks=[
                {"id": "task_001", "title": "First task"},
                {"id": "task_002", "title": "Second task"},
            ],
            estimated_complexity="medium",
        )

        scribe.save_plan(plan)

        assert scribe._plan == plan
        assert len(scribe._tasks) == 2
        assert scribe._current_index == 0

    def test_save_plan_with_dict(self, mock_scribe_functions):
        scribe = ScribeMCP()

        plan_dict = {
            "goal": "Test goal",
            "tasks": [
                {"id": "task_001", "title": "Task 1"},
            ]
        }

        scribe.save_plan(plan_dict)

        assert len(scribe._tasks) == 1

    def test_save_plan_with_list(self, mock_scribe_functions):
        scribe = ScribeMCP()

        tasks = [
            {"id": "task_001", "title": "Task 1"},
            {"id": "task_002", "title": "Task 2"},
        ]

        scribe.save_plan(tasks)

        assert len(scribe._tasks) == 2

    def test_get_next_task(self, mock_scribe_functions):
        scribe = ScribeMCP()
        scribe._tasks = [
            {"id": "task_001", "title": "First"},
            {"id": "task_002", "title": "Second"},
        ]

        task1 = scribe.get_next_task()
        assert task1["id"] == "task_001"
        assert scribe._current_index == 1

        task2 = scribe.get_next_task()
        assert task2["id"] == "task_002"
        assert scribe._current_index == 2

        task3 = scribe.get_next_task()
        assert task3 is None

    def test_get_next_task_updates_status(self, mock_scribe_functions):
        scribe = ScribeMCP()
        scribe._tasks = [{"id": "task_001", "title": "Test"}]

        scribe.get_next_task()

        mock_scribe_functions["update_todo"].assert_called_with("task_001", "in_progress")

    def test_log_result(self, mock_scribe_functions):
        scribe = ScribeMCP()

        result = {
            "task_id": "task_001",
            "status": "completed",
            "output": "Task completed successfully",
        }

        scribe.log_result(result)

        mock_scribe_functions["log_event"].assert_called()
        mock_scribe_functions["update_todo"].assert_called_with("task_001", "completed")

    def test_log_result_with_nested_task(self, mock_scribe_functions):
        scribe = ScribeMCP()

        result = {
            "task": {"id": "task_002"},
            "status": "failed",
        }

        scribe.log_result(result)

        mock_scribe_functions["update_todo"].assert_called_with("task_002", "failed")


class TestMinmoOrchestrator:
    """Tests for MinmoOrchestrator class."""

    @pytest.fixture
    def mock_all(self):
        """Mock all external dependencies."""
        with patch("minmo.orchestrator.log_event") as mock_log, \
             patch("minmo.orchestrator.update_todo") as mock_update, \
             patch("minmo.orchestrator.get_state") as mock_state:
            mock_log.return_value = True
            mock_update.return_value = True
            mock_state.return_value = {}
            yield {
                "log_event": mock_log,
                "update_todo": mock_update,
                "get_state": mock_state,
            }

    @pytest.fixture
    def mock_gemini(self):
        """Mock GeminiWrapper."""
        with patch("minmo.orchestrator.GeminiWrapper") as mock:
            instance = MagicMock()
            mock.return_value = instance
            yield instance

    @pytest.fixture
    def mock_claude(self):
        """Mock ClaudeCodeWrapper."""
        with patch("minmo.orchestrator.ClaudeCodeWrapper") as mock:
            instance = MagicMock()
            mock.return_value = instance
            yield instance

    def test_init(self, mock_all):
        orchestrator = MinmoOrchestrator()

        assert orchestrator.scribe is not None
        assert orchestrator.commander is None
        assert orchestrator.worker is None
        assert orchestrator._is_complete is False

    def test_init_with_options(self, mock_all):
        output_cb = MagicMock()
        error_cb = MagicMock()

        orchestrator = MinmoOrchestrator(
            working_directory="/test/dir",
            on_output=output_cb,
            on_error=error_cb,
            verbose=True,
        )

        assert orchestrator._working_directory == "/test/dir"
        assert orchestrator._verbose is True

    def test_ensure_commander(self, mock_all, mock_gemini):
        orchestrator = MinmoOrchestrator()

        commander1 = orchestrator._ensure_commander()
        commander2 = orchestrator._ensure_commander()

        assert commander1 is commander2

    def test_ensure_worker(self, mock_all, mock_claude):
        orchestrator = MinmoOrchestrator()

        worker1 = orchestrator._ensure_worker()
        worker2 = orchestrator._ensure_worker()

        assert worker1 is worker2

    def test_is_complete(self, mock_all):
        orchestrator = MinmoOrchestrator()

        assert orchestrator.is_complete() is False

        orchestrator._is_complete = True
        assert orchestrator.is_complete() is True

    def test_init_project(self, mock_all, mock_gemini):
        mock_gemini.init_project.return_value = ProjectAnalysis(
            project_type=ProjectType.EXISTING,
            detected_languages=["Python"],
        )

        orchestrator = MinmoOrchestrator()
        result = orchestrator.init_project("/test/path", ["main.py"])

        assert result.project_type == ProjectType.EXISTING
        mock_gemini.init_project.assert_called_once_with("/test/path", ["main.py"])

    def test_clarify_goal(self, mock_all, mock_gemini):
        mock_gemini.clarify_goal.return_value = {
            "original_goal": "Test goal",
            "final_requirements": {"summary": "Clarified goal"},
        }

        orchestrator = MinmoOrchestrator()
        result = orchestrator.clarify_goal("Test goal")

        assert result["final_requirements"]["summary"] == "Clarified goal"

    def test_analyze_code(self, mock_all, mock_gemini):
        mock_gemini.analyze_code.return_value = {
            "summary": "Code analysis",
            "issues": [],
        }

        orchestrator = MinmoOrchestrator()
        result = orchestrator.analyze_code("def test(): pass", "/test.py")

        assert "summary" in result

    def test_review_changes(self, mock_all, mock_gemini):
        mock_gemini.review_changes.return_value = {
            "approval": "approved",
            "comments": [],
        }

        orchestrator = MinmoOrchestrator()
        result = orchestrator.review_changes("+ new line")

        assert result["approval"] == "approved"

    def test_start_loop_success(self, mock_all, mock_gemini, mock_claude):
        mock_gemini.clarify_goal.return_value = {
            "final_requirements": {"summary": "Create feature"},
        }
        mock_gemini.plan.return_value = TaskPlan(
            goal="Create feature",
            tasks=[
                {"id": "task_001", "title": "First task"},
                {"id": "task_002", "title": "Second task"},
            ],
        )

        mock_claude.execute.side_effect = [
            TaskResult(
                task_id="task_001",
                status=TaskStatus.COMPLETED,
                output="Task 1 done",
                files_modified=["file1.py"],
                duration_seconds=5.0,
            ),
            TaskResult(
                task_id="task_002",
                status=TaskStatus.COMPLETED,
                output="Task 2 done",
                files_modified=["file2.py"],
                duration_seconds=3.0,
            ),
        ]

        orchestrator = MinmoOrchestrator()
        result = orchestrator.start_loop("Create a new feature")

        assert result["status"] == "completed"
        assert result["summary"]["completed"] == 2
        assert result["summary"]["failed"] == 0

    def test_start_loop_with_failure(self, mock_all, mock_gemini, mock_claude):
        mock_gemini.plan.return_value = TaskPlan(
            goal="Test",
            tasks=[{"id": "task_001", "title": "Failing task"}],
        )

        mock_claude.execute.return_value = TaskResult(
            task_id="task_001",
            status=TaskStatus.FAILED,
            error="Task failed",
            duration_seconds=1.0,
        )

        orchestrator = MinmoOrchestrator()
        result = orchestrator.start_loop("Test", skip_clarification=True)

        assert result["status"] == "failed"
        assert result["summary"]["failed"] == 1

    def test_start_loop_partial_success(self, mock_all, mock_gemini, mock_claude):
        mock_gemini.plan.return_value = TaskPlan(
            goal="Test",
            tasks=[
                {"id": "task_001", "title": "Pass"},
                {"id": "task_002", "title": "Fail"},
            ],
        )

        mock_claude.execute.side_effect = [
            TaskResult(task_id="task_001", status=TaskStatus.COMPLETED),
            TaskResult(task_id="task_002", status=TaskStatus.FAILED, error="Failed"),
        ]

        orchestrator = MinmoOrchestrator()
        result = orchestrator.start_loop("Test", skip_clarification=True)

        assert result["status"] == "partial"
        assert result["summary"]["completed"] == 1
        assert result["summary"]["failed"] == 1

    def test_start_loop_stop_on_failure(self, mock_all, mock_gemini, mock_claude):
        mock_gemini.plan.return_value = TaskPlan(
            goal="Test",
            tasks=[
                {"id": "task_001", "title": "Fail"},
                {"id": "task_002", "title": "Skip"},
            ],
        )

        mock_claude.execute.return_value = TaskResult(
            task_id="task_001",
            status=TaskStatus.FAILED,
            error="First task failed",
        )

        orchestrator = MinmoOrchestrator()
        result = orchestrator.start_loop("Test", skip_clarification=True, stop_on_failure=True)

        assert len(result["results"]) == 1
        mock_claude.execute.assert_called_once()

    def test_start_loop_with_project_analysis(self, mock_all, mock_gemini, mock_claude):
        mock_gemini.plan.return_value = TaskPlan(
            goal="Test",
            tasks=[{"id": "task_001", "title": "Task"}],
        )
        mock_claude.execute.return_value = TaskResult(
            task_id="task_001",
            status=TaskStatus.COMPLETED,
        )

        analysis = ProjectAnalysis(
            project_type=ProjectType.EXISTING,
            detected_languages=["Python"],
        )

        orchestrator = MinmoOrchestrator()
        result = orchestrator.start_loop(
            "Test",
            skip_clarification=True,
            project_analysis=analysis,
        )

        mock_gemini.plan.assert_called_once()
        call_kwargs = mock_gemini.plan.call_args[1]
        assert call_kwargs["project_analysis"] == analysis

    def test_commander_feedback_analyze_error(self, mock_all, mock_gemini):
        mock_gemini.analyze_code.return_value = {
            "summary": "Error analysis",
            "issues": ["Missing import"],
        }

        orchestrator = MinmoOrchestrator()
        orchestrator.commander = mock_gemini

        response = orchestrator._commander_feedback("analyze_error", {
            "task": {"id": "test", "title": "Test", "description": "Original"},
            "error": "ImportError: No module named 'missing'",
            "retry_count": 1,
        })

        assert response["action"] == "retry_with_modification"
        assert "modified_task" in response
        assert "이전 시도 에러" in response["modified_task"]["description"]

    def test_commander_feedback_max_retries(self, mock_all, mock_gemini):
        mock_gemini.analyze_code.return_value = {"summary": "Skip recommended"}

        orchestrator = MinmoOrchestrator()
        orchestrator.commander = mock_gemini

        response = orchestrator._commander_feedback("analyze_error", {
            "task": {"id": "test", "title": "Test"},
            "error": "Persistent error",
            "retry_count": 4,
        })

        assert response["action"] == "skip"

    def test_commander_feedback_unknown_action(self, mock_all, mock_gemini):
        orchestrator = MinmoOrchestrator()

        response = orchestrator._commander_feedback("unknown_action", {})

        assert response["action"] == "continue"


class TestProtocols:
    """Tests for Protocol classes."""

    def test_scribe_protocol_compliance(self):
        scribe = ScribeMCP()

        assert hasattr(scribe, "save_plan")
        assert hasattr(scribe, "get_next_task")
        assert hasattr(scribe, "log_result")
        assert callable(scribe.save_plan)
        assert callable(scribe.get_next_task)
        assert callable(scribe.log_result)


class TestIntegration:
    """Integration tests for orchestrator module."""

    @pytest.fixture
    def mock_deps(self):
        """Mock all external dependencies for integration tests."""
        with patch("minmo.orchestrator.log_event") as mock_log, \
             patch("minmo.orchestrator.update_todo") as mock_update, \
             patch("minmo.orchestrator.get_state") as mock_state, \
             patch("minmo.orchestrator.GeminiWrapper") as mock_gemini_cls, \
             patch("minmo.orchestrator.ClaudeCodeWrapper") as mock_claude_cls:

            mock_log.return_value = True
            mock_update.return_value = True
            mock_state.return_value = {}

            mock_gemini = MagicMock()
            mock_gemini.plan.return_value = TaskPlan(
                goal="Integration test",
                tasks=[
                    {"id": "int_001", "title": "Integration task 1"},
                    {"id": "int_002", "title": "Integration task 2"},
                ],
            )
            mock_gemini.clarify_goal.return_value = {
                "final_requirements": {"summary": "Integration test goal"},
            }
            mock_gemini_cls.return_value = mock_gemini

            mock_claude = MagicMock()
            mock_claude.execute.side_effect = [
                TaskResult(task_id="int_001", status=TaskStatus.COMPLETED, duration_seconds=1.0),
                TaskResult(task_id="int_002", status=TaskStatus.COMPLETED, duration_seconds=2.0),
            ]
            mock_claude_cls.return_value = mock_claude

            yield {
                "gemini": mock_gemini,
                "claude": mock_claude,
            }

    def test_full_orchestration_flow(self, mock_deps):
        orchestrator = MinmoOrchestrator(
            working_directory="/test",
        )

        result = orchestrator.start_loop("Complete integration test")

        assert result["status"] == "completed"
        assert result["summary"]["total"] == 2
        assert result["summary"]["completed"] == 2
        assert "plan" in result
        assert "results" in result

        mock_deps["gemini"].plan.assert_called()
        assert mock_deps["claude"].execute.call_count == 2

    def test_orchestration_with_clarification(self, mock_deps):
        orchestrator = MinmoOrchestrator()

        result = orchestrator.start_loop("Ambiguous goal")

        mock_deps["gemini"].clarify_goal.assert_called()
        assert result["clarification"] is not None
