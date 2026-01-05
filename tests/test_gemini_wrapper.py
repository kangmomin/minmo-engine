"""
Tests for minmo.gemini_wrapper module (CLI-based).

Gemini CLI (@google/gemini-cli) 기반 래퍼 테스트
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import pytest

from minmo.gemini_wrapper import (
    ProjectType,
    ProjectAnalysis,
    ClarificationQuestion,
    TaskPlan,
    GeminiWrapper,
    COMMANDER_CONSTITUTION,
    InterviewFocus,
    InterviewQuestion,
    InterviewAnswer,
    FeatureSpec,
    PlanTask,
    PlanModeResult,
)


class TestProjectType:
    """Tests for ProjectType enum."""

    def test_enum_values(self):
        assert ProjectType.NEW.value == "new"
        assert ProjectType.EXISTING.value == "existing"
        assert ProjectType.UNKNOWN.value == "unknown"


class TestProjectAnalysis:
    """Tests for ProjectAnalysis dataclass."""

    def test_create_analysis(self):
        analysis = ProjectAnalysis(
            project_type=ProjectType.EXISTING,
            detected_languages=["Python", "JavaScript"],
            detected_frameworks=["FastAPI", "React"],
            detected_databases=["PostgreSQL"],
            structure_summary="A web application with backend and frontend",
            recommendations=["Add type hints"],
            confidence=0.85,
        )

        assert analysis.project_type == ProjectType.EXISTING
        assert "Python" in analysis.detected_languages
        assert analysis.confidence == 0.85

    def test_default_values(self):
        analysis = ProjectAnalysis(project_type=ProjectType.NEW)

        assert analysis.detected_languages == []
        assert analysis.detected_frameworks == []
        assert analysis.detected_databases == []
        assert analysis.structure_summary == ""
        assert analysis.recommendations == []
        assert analysis.confidence == 0.0


class TestClarificationQuestion:
    """Tests for ClarificationQuestion dataclass."""

    def test_create_question(self):
        question = ClarificationQuestion(
            question="Which database do you prefer?",
            options=["PostgreSQL", "MySQL", "SQLite"],
            context="We need a database for storing user data",
            required=True,
        )

        assert "database" in question.question
        assert len(question.options) == 3
        assert question.required is True

    def test_default_values(self):
        question = ClarificationQuestion(question="Simple question?")

        assert question.options == []
        assert question.context == ""
        assert question.required is True


class TestTaskPlan:
    """Tests for TaskPlan dataclass."""

    def test_create_plan(self):
        plan = TaskPlan(
            goal="Implement user authentication",
            tasks=[
                {"id": "task_001", "title": "Create User model"},
                {"id": "task_002", "title": "Implement login endpoint"},
            ],
            estimated_complexity="medium",
            prerequisites=["Database setup"],
            risks=["Session management complexity"],
        )

        assert plan.goal == "Implement user authentication"
        assert len(plan.tasks) == 2
        assert plan.estimated_complexity == "medium"

    def test_default_values(self):
        plan = TaskPlan(goal="Simple task")

        assert plan.tasks == []
        assert plan.estimated_complexity == "medium"
        assert plan.prerequisites == []
        assert plan.risks == []


class TestInterviewFocus:
    """Tests for InterviewFocus enum."""

    def test_enum_values(self):
        assert InterviewFocus.ARCHITECTURE.value == "architecture"
        assert InterviewFocus.DATA_MODEL.value == "data_model"
        assert InterviewFocus.EXCEPTION_HANDLING.value == "exception_handling"
        assert InterviewFocus.CONVENTION.value == "convention"
        assert InterviewFocus.INTEGRATION.value == "integration"
        assert InterviewFocus.TESTING.value == "testing"


class TestInterviewQuestion:
    """Tests for InterviewQuestion dataclass."""

    def test_create_question(self):
        question = InterviewQuestion(
            question="Which database do you prefer?",
            focus=InterviewFocus.DATA_MODEL,
            options=["PostgreSQL", "MySQL", "SQLite"],
            context="We need a database for storing user data",
        )

        assert "database" in question.question
        assert question.focus == InterviewFocus.DATA_MODEL
        assert len(question.options) == 3

    def test_default_values(self):
        question = InterviewQuestion(
            question="Simple question?",
            focus=InterviewFocus.ARCHITECTURE
        )

        assert question.options == []
        assert question.context == ""


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_create_spec(self):
        spec = FeatureSpec(
            feature_name="user_authentication",
            summary="User login and registration system",
            requirements=["Email login", "Password reset"],
            architecture_decisions=["Use JWT tokens"],
            data_model={"User": {"id": "int", "email": "str"}},
            error_handling=["Invalid credentials error"],
            conventions=["REST API naming"],
            constraints=["Must work offline"],
            out_of_scope=["Social login"],
        )

        assert spec.feature_name == "user_authentication"
        assert len(spec.requirements) == 2

    def test_default_values(self):
        spec = FeatureSpec(feature_name="test", summary="Test feature")

        assert spec.requirements == []
        assert spec.architecture_decisions == []
        assert spec.data_model == {}


class TestPlanTask:
    """Tests for PlanTask dataclass."""

    def test_create_task(self):
        task = PlanTask(
            id="task_001",
            title="Create User model",
            goal="Define User dataclass with all fields",
            files_to_modify=["models/user.py"],
            expected_logic="Create User class",
            dependencies=[],
            acceptance_criteria=["User class exists"],
        )

        assert task.id == "task_001"
        assert len(task.files_to_modify) == 1

    def test_default_values(self):
        task = PlanTask(id="task_002", title="Test task", goal="Do something")

        assert task.files_to_modify == []
        assert task.expected_logic == ""


class TestPlanModeResult:
    """Tests for PlanModeResult dataclass."""

    def test_create_result(self):
        spec = FeatureSpec(feature_name="test", summary="Test")
        tasks = [PlanTask(id="t1", title="Task 1", goal="Goal 1")]

        result = PlanModeResult(
            feature_spec=spec,
            tasks=tasks,
            interview_history=[],
            approved=True,
            spec_file_path="/specs/test.md",
        )

        assert result.feature_spec.feature_name == "test"
        assert len(result.tasks) == 1
        assert result.approved is True

    def test_default_values(self):
        result = PlanModeResult()

        assert result.feature_spec is None
        assert result.tasks == []
        assert result.approved is False


class TestGeminiWrapper:
    """Tests for GeminiWrapper class (CLI-based)."""

    @pytest.fixture
    def mock_scribe(self):
        """Mock the scribe_mcp functions."""
        with patch("minmo.gemini_wrapper.log_event") as mock_log, \
             patch("minmo.gemini_wrapper.init_database") as mock_init:
            mock_log.return_value = True
            mock_init.return_value = None
            yield {"log_event": mock_log, "init_database": mock_init}

    @pytest.fixture
    def mock_shutil(self):
        """Mock shutil.which to return gemini path."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/gemini"
            yield mock_which

    @pytest.fixture
    def wrapper(self, mock_scribe, mock_shutil):
        """Create a GeminiWrapper instance with mocked dependencies."""
        wrapper = GeminiWrapper(
            working_directory="/test/project",
            timeout_seconds=60,
            verbose=False
        )
        yield wrapper
        wrapper.close()

    def test_init_default_values(self, mock_scribe, mock_shutil):
        wrapper = GeminiWrapper()

        assert wrapper.working_directory == os.getcwd()
        assert wrapper.timeout_seconds == 120
        assert wrapper.verbose is False
        wrapper.close()

    def test_init_custom_values(self, mock_scribe, mock_shutil):
        wrapper = GeminiWrapper(
            working_directory="/custom/path",
            timeout_seconds=300,
            verbose=True
        )

        assert wrapper.working_directory == "/custom/path"
        assert wrapper.timeout_seconds == 300
        assert wrapper.verbose is True
        wrapper.close()

    def test_find_gemini_cli_from_env(self, mock_scribe):
        """Test finding gemini CLI from environment variable."""
        with patch.dict(os.environ, {"GEMINI_CLI_PATH": "/custom/gemini"}):
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = True
                wrapper = GeminiWrapper()
                assert wrapper.gemini_path == "/custom/gemini"
                wrapper.close()

    def test_find_gemini_cli_from_path(self, mock_scribe, mock_shutil):
        """Test finding gemini CLI from PATH."""
        wrapper = GeminiWrapper()
        assert wrapper.gemini_path == "/usr/local/bin/gemini"
        wrapper.close()

    def test_check_login_status_success(self, wrapper, mock_scribe):
        """Test successful login status check."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result = wrapper.check_login_status()
            assert result is True

    def test_check_login_status_failure(self, wrapper, mock_scribe):
        """Test failed login status check."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1)
            result = wrapper.check_login_status()
            assert result is False

    def test_parse_json_response_with_json_block(self, wrapper):
        """Test parsing JSON from markdown code block."""
        response = '''Here is the analysis:
```json
{"type": "test", "content": {"key": "value"}}
```
'''
        result = wrapper._parse_json_response(response)
        assert result["type"] == "test"
        assert result["content"]["key"] == "value"

    def test_parse_json_response_plain_json(self, wrapper):
        """Test parsing plain JSON."""
        response = '{"type": "plain", "content": {"data": 123}}'
        result = wrapper._parse_json_response(response)
        assert result["type"] == "plain"

    def test_parse_json_response_invalid_json(self, wrapper):
        """Test parsing invalid JSON returns text content."""
        response = "This is not JSON at all"
        result = wrapper._parse_json_response(response)
        assert result["type"] == "text"
        assert result["content"] == response

    def test_init_project_existing(self, wrapper, mock_scribe):
        """Test project initialization for existing project."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "analysis",
  "content": {
    "project_type": "existing",
    "detected_languages": ["Python"],
    "detected_frameworks": ["FastAPI"],
    "detected_databases": ["SQLite"],
    "structure_summary": "A backend project",
    "recommendations": [],
    "confidence": 0.9
  }
}
```'''

            file_list = ["main.py", "models.py", "routes.py"]
            result = wrapper.init_project("/test/project", file_list)

            assert result.project_type == ProjectType.EXISTING
            assert "Python" in result.detected_languages
            assert result.confidence == 0.9

    def test_init_project_new(self, wrapper, mock_scribe):
        """Test project initialization for new project."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "recommendation",
  "content": {
    "project_type": "new",
    "detected_languages": [],
    "detected_frameworks": [],
    "detected_databases": [],
    "recommendations": ["Use Python for backend"],
    "confidence": 0.5
  }
}
```'''

            result = wrapper.init_project("/new/project", None)

            assert result.project_type == ProjectType.NEW

    def test_clarify_goal_no_clarification_needed(self, wrapper, mock_scribe):
        """Test clarify goal when no clarification is needed."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "clarification",
  "content": {
    "needs_clarification": false,
    "understood_requirements": {
      "summary": "Create a login page",
      "scope": ["Frontend", "Backend"],
      "constraints": [],
      "assumptions": []
    }
  }
}
```'''

            result = wrapper.clarify_goal("Create a login page")

            assert result["original_goal"] == "Create a login page"
            assert result["final_requirements"] is not None

    def test_plan(self, wrapper, mock_scribe):
        """Test plan generation."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "plan",
  "content": {
    "goal": "Implement user login",
    "tasks": [
      {"id": "task_001", "title": "Create User model"},
      {"id": "task_002", "title": "Implement login endpoint"}
    ],
    "estimated_complexity": "medium",
    "prerequisites": [],
    "risks": []
  }
}
```'''

            result = wrapper.plan("Implement user login")

            assert result.goal == "Implement user login"
            assert len(result.tasks) == 2

    def test_analyze_code(self, wrapper, mock_scribe):
        """Test code analysis."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "analysis",
  "content": {
    "summary": "A utility function module",
    "purpose": "Helper functions for data processing",
    "key_functions": ["process_data", "validate_input"],
    "dependencies": ["json", "typing"],
    "issues": [],
    "suggestions": []
  }
}
```'''

            code = "def process_data(data): return data"
            result = wrapper.analyze_code(code, "/test/utils.py")

            assert result["summary"] == "A utility function module"

    def test_review_changes(self, wrapper, mock_scribe):
        """Test code review."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "review",
  "content": {
    "summary": "Added new feature",
    "approval": "approved",
    "comments": [{"type": "praise", "location": "main.py:10", "message": "Good"}],
    "security_concerns": [],
    "breaking_changes": []
  }
}
```'''

            diff = "+ def new_feature(): return True"
            result = wrapper.review_changes(diff)

            assert result["approval"] == "approved"

    def test_get_conversation_history(self, wrapper, mock_scribe):
        """Test conversation history retrieval."""
        wrapper._conversation_history = [
            {"role": "user", "content": "Hello"},
            {"role": "model", "content": "Hi there"}
        ]

        history = wrapper.get_conversation_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"

    def test_reset_conversation(self, wrapper, mock_scribe):
        """Test conversation reset."""
        wrapper._conversation_history = [{"role": "user", "content": "Test"}]
        wrapper.reset_conversation()

        assert wrapper._conversation_history == []

    def test_generate_interview_questions(self, wrapper, mock_scribe):
        """Test interview question generation."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "interview_questions",
  "content": {
    "questions": [
      {
        "question": "Which database do you prefer?",
        "focus": "data_model",
        "options": ["PostgreSQL", "MySQL"],
        "context": "Need to choose database"
      }
    ]
  }
}
```'''

            questions = wrapper.generate_interview_questions("Add database")

            assert len(questions) == 1
            assert questions[0].focus == InterviewFocus.DATA_MODEL

    def test_generate_feature_spec(self, wrapper, mock_scribe):
        """Test feature spec generation."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "feature_spec",
  "content": {
    "feature_name": "user_auth",
    "summary": "User authentication system",
    "requirements": ["Login", "Logout"],
    "architecture_decisions": ["Use JWT"],
    "data_model": {},
    "error_handling": [],
    "conventions": [],
    "constraints": [],
    "out_of_scope": []
  }
}
```'''

            question = InterviewQuestion(
                question="Which auth?",
                focus=InterviewFocus.ARCHITECTURE
            )
            answers = [InterviewAnswer(question=question, answer="JWT")]

            spec = wrapper.generate_feature_spec("Add auth", answers)

            assert spec.feature_name == "user_auth"

    def test_decompose_to_tasks(self, wrapper, mock_scribe):
        """Test task decomposition."""
        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "task_decomposition",
  "content": {
    "tasks": [
      {
        "id": "task_001",
        "title": "Create User model",
        "goal": "Define User class",
        "files_to_modify": ["models/user.py"],
        "expected_logic": "Create dataclass",
        "dependencies": [],
        "acceptance_criteria": ["Class exists"]
      }
    ]
  }
}
```'''

            spec = FeatureSpec(feature_name="test", summary="Test")
            tasks = wrapper.decompose_to_tasks(spec)

            assert len(tasks) == 1
            assert tasks[0].id == "task_001"

    def test_run_plan_mode(self, wrapper, mock_scribe):
        """Test complete plan mode flow."""
        with patch.object(wrapper, 'generate_interview_questions') as mock_q, \
             patch.object(wrapper, 'generate_feature_spec') as mock_spec, \
             patch.object(wrapper, 'decompose_to_tasks') as mock_tasks:

            mock_q.return_value = [
                InterviewQuestion(
                    question="Test?",
                    focus=InterviewFocus.ARCHITECTURE,
                    options=["A", "B"]
                )
            ]

            mock_spec.return_value = FeatureSpec(
                feature_name="test_feature",
                summary="Test"
            )

            mock_tasks.return_value = [
                PlanTask(id="t1", title="Task 1", goal="Goal 1")
            ]

            def on_question(q):
                return "A"

            def on_tasks_review(tasks):
                return True

            result = wrapper.run_plan_mode(
                user_goal="Test goal",
                on_question=on_question,
                on_tasks_review=on_tasks_review
            )

            assert result.feature_spec.feature_name == "test_feature"
            assert result.approved is True

    def test_context_manager(self, mock_scribe, mock_shutil):
        """Test wrapper works as context manager."""
        with GeminiWrapper() as wrapper:
            assert wrapper is not None

    def test_log_callback(self, mock_scribe, mock_shutil):
        """Test log callback is called."""
        logs = []

        def on_log(agent, content, metadata):
            logs.append({"agent": agent, "content": content})

        wrapper = GeminiWrapper(on_log=on_log)

        with patch.object(wrapper, '_send_message') as mock_send:
            mock_send.return_value = '{"type": "analysis", "content": {}}'
            wrapper.init_project("/test", [])

        wrapper.close()

        assert len(logs) > 0
        assert logs[0]["agent"] == "commander"


class TestCommanderConstitution:
    """Tests for COMMANDER_CONSTITUTION constant."""

    def test_constitution_exists(self):
        assert COMMANDER_CONSTITUTION is not None
        assert len(COMMANDER_CONSTITUTION) > 0

    def test_constitution_contains_key_principles(self):
        assert "최소주의" in COMMANDER_CONSTITUTION
        assert "명확성" in COMMANDER_CONSTITUTION
        assert "검증" in COMMANDER_CONSTITUTION
        assert "품질" in COMMANDER_CONSTITUTION

    def test_constitution_contains_json_format(self):
        assert "JSON" in COMMANDER_CONSTITUTION
        assert "type" in COMMANDER_CONSTITUTION

    def test_constitution_contains_forbidden_items(self):
        assert "금지 사항" in COMMANDER_CONSTITUTION
