"""
Tests for minmo.gemini_cli_wrapper module.

Gemini CLI (@google/gemini-cli) 래퍼에 대한 테스트
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path

import pytest

from minmo.gemini_cli_wrapper import (
    InterviewFocus,
    ProjectAnalysis,
    InterviewQuestion,
    InterviewAnswer,
    FeatureSpec,
    PlanTask,
    PlanModeResult,
    GeminiCLIWrapper,
    COMMANDER_CLI_CONSTITUTION,
)


class TestInterviewFocus:
    """Tests for InterviewFocus enum."""

    def test_enum_values(self):
        assert InterviewFocus.ARCHITECTURE.value == "architecture"
        assert InterviewFocus.DATA_MODEL.value == "data_model"
        assert InterviewFocus.EXCEPTION_HANDLING.value == "exception_handling"
        assert InterviewFocus.CONVENTION.value == "convention"
        assert InterviewFocus.INTEGRATION.value == "integration"
        assert InterviewFocus.TESTING.value == "testing"


class TestProjectAnalysis:
    """Tests for ProjectAnalysis dataclass."""

    def test_create_analysis(self):
        analysis = ProjectAnalysis(
            project_path="/test/project",
            languages=["Python", "JavaScript"],
            frameworks=["FastAPI", "React"],
            structure_summary="A web application with backend and frontend",
            key_files=["main.py", "app.tsx"],
            conventions=["PEP 8", "ESLint"],
            dependencies=["fastapi", "react"],
        )

        assert analysis.project_path == "/test/project"
        assert "Python" in analysis.languages
        assert "FastAPI" in analysis.frameworks
        assert len(analysis.key_files) == 2

    def test_default_values(self):
        analysis = ProjectAnalysis(project_path="/test")

        assert analysis.languages == []
        assert analysis.frameworks == []
        assert analysis.structure_summary == ""
        assert analysis.key_files == []
        assert analysis.conventions == []
        assert analysis.dependencies == []


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
        assert question.context != ""

    def test_default_values(self):
        question = InterviewQuestion(
            question="Simple question?",
            focus=InterviewFocus.ARCHITECTURE
        )

        assert question.options == []
        assert question.context == ""


class TestInterviewAnswer:
    """Tests for InterviewAnswer dataclass."""

    def test_create_answer(self):
        question = InterviewQuestion(
            question="Which auth method?",
            focus=InterviewFocus.ARCHITECTURE,
            options=["JWT", "Session"],
        )
        answer = InterviewAnswer(question=question, answer="JWT")

        assert answer.question.question == "Which auth method?"
        assert answer.answer == "JWT"


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
        assert "Use JWT tokens" in spec.architecture_decisions
        assert "User" in spec.data_model

    def test_default_values(self):
        spec = FeatureSpec(feature_name="test", summary="Test feature")

        assert spec.requirements == []
        assert spec.architecture_decisions == []
        assert spec.data_model == {}
        assert spec.error_handling == []
        assert spec.conventions == []
        assert spec.constraints == []
        assert spec.out_of_scope == []


class TestPlanTask:
    """Tests for PlanTask dataclass."""

    def test_create_task(self):
        task = PlanTask(
            id="task_001",
            title="Create User model",
            goal="Define User dataclass with all fields",
            files_to_modify=["models/user.py"],
            expected_logic="Create User class with id, email, password_hash",
            dependencies=[],
            acceptance_criteria=["User class exists", "All fields defined"],
        )

        assert task.id == "task_001"
        assert task.title == "Create User model"
        assert len(task.files_to_modify) == 1
        assert len(task.acceptance_criteria) == 2

    def test_default_values(self):
        task = PlanTask(id="task_002", title="Test task", goal="Do something")

        assert task.files_to_modify == []
        assert task.expected_logic == ""
        assert task.dependencies == []
        assert task.acceptance_criteria == []


class TestPlanModeResult:
    """Tests for PlanModeResult dataclass."""

    def test_create_result(self):
        analysis = ProjectAnalysis(project_path="/test")
        spec = FeatureSpec(feature_name="test", summary="Test")
        tasks = [PlanTask(id="t1", title="Task 1", goal="Goal 1")]

        result = PlanModeResult(
            project_analysis=analysis,
            feature_spec=spec,
            tasks=tasks,
            interview_history=[],
            approved=True,
            spec_file_path="/specs/test.md",
        )

        assert result.project_analysis is not None
        assert result.feature_spec.feature_name == "test"
        assert len(result.tasks) == 1
        assert result.approved is True

    def test_default_values(self):
        result = PlanModeResult()

        assert result.project_analysis is None
        assert result.feature_spec is None
        assert result.tasks == []
        assert result.interview_history == []
        assert result.approved is False
        assert result.spec_file_path == ""


class TestGeminiCLIWrapper:
    """Tests for GeminiCLIWrapper class."""

    @pytest.fixture
    def mock_scribe(self):
        """Mock the scribe_mcp functions."""
        with patch("minmo.gemini_cli_wrapper.log_event") as mock_log, \
             patch("minmo.gemini_cli_wrapper.init_database") as mock_init:
            mock_log.return_value = True
            mock_init.return_value = None
            yield {"log_event": mock_log, "init_database": mock_init}

    @pytest.fixture
    def mock_pexpect(self):
        """Mock pexpect module."""
        mock_process = MagicMock()
        mock_process.isalive.return_value = True
        mock_process.before = ""

        with patch("minmo.gemini_cli_wrapper.PexpectSpawn") as mock_spawn:
            mock_spawn.return_value = mock_process
            yield {
                "spawn": mock_spawn,
                "process": mock_process,
            }

    @pytest.fixture
    def mock_shutil(self):
        """Mock shutil.which to return gemini path."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/local/bin/gemini"
            yield mock_which

    @pytest.fixture
    def wrapper(self, mock_scribe, mock_shutil):
        """Create a GeminiCLIWrapper instance with mocked dependencies."""
        wrapper = GeminiCLIWrapper(
            working_directory="/test/project",
            timeout_seconds=60,
            verbose=False
        )
        yield wrapper
        wrapper.close()

    def test_init_default_values(self, mock_scribe, mock_shutil):
        wrapper = GeminiCLIWrapper()

        assert wrapper.working_directory == os.getcwd()
        assert wrapper.timeout_seconds == 120
        assert wrapper.verbose is False
        wrapper.close()

    def test_init_custom_values(self, mock_scribe, mock_shutil):
        wrapper = GeminiCLIWrapper(
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
                wrapper = GeminiCLIWrapper()
                assert wrapper.gemini_path == "/custom/gemini"
                wrapper.close()

    def test_find_gemini_cli_from_path(self, mock_scribe, mock_shutil):
        """Test finding gemini CLI from PATH."""
        wrapper = GeminiCLIWrapper()
        assert wrapper.gemini_path == "/usr/local/bin/gemini"
        wrapper.close()

    def test_find_gemini_cli_not_found(self, mock_scribe):
        """Test when gemini CLI is not found raises RuntimeError."""
        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            with patch("os.path.exists") as mock_exists:
                mock_exists.return_value = False
                with pytest.raises(RuntimeError) as excinfo:
                    GeminiCLIWrapper()
                # Should contain installation instructions
                assert "npm install -g @google/gemini-cli" in str(excinfo.value)
                assert "gemini login" in str(excinfo.value)

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

    def test_check_login_status_exception(self, wrapper, mock_scribe):
        """Test login status check with exception."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("Process failed")
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
        assert result["content"]["data"] == 123

    def test_parse_json_response_json_array(self, wrapper):
        """Test parsing JSON array."""
        response = '[{"id": 1}, {"id": 2}]'
        result = wrapper._parse_json_response(response)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_json_response_invalid_json(self, wrapper):
        """Test parsing invalid JSON returns text content."""
        response = "This is not JSON at all"
        result = wrapper._parse_json_response(response)
        assert result["type"] == "text"
        assert result["content"] == response

    def test_analyze_project(self, wrapper, mock_pexpect, mock_scribe):
        """Test project analysis."""
        mock_pexpect["process"].readline.side_effect = [
            '```json',
            '{"type": "analysis", "content": {"languages": ["Python"], "frameworks": ["FastAPI"], "structure_summary": "A backend project", "key_files": ["main.py"], "conventions": ["PEP 8"], "dependencies": ["fastapi"]}}',
            '```',
            '>',
        ]

        with patch.object(wrapper, '_send_command') as mock_send:
            mock_send.return_value = '''```json
{"type": "analysis", "content": {"languages": ["Python"], "frameworks": ["FastAPI"], "structure_summary": "A backend project", "key_files": ["main.py"], "conventions": ["PEP 8"], "dependencies": ["fastapi"]}}
```'''

            result = wrapper.analyze_project("/test/project")

            assert result.project_path == "/test/project"
            assert "Python" in result.languages
            assert "FastAPI" in result.frameworks

    def test_generate_interview_questions(self, wrapper, mock_pexpect, mock_scribe):
        """Test interview question generation."""
        with patch.object(wrapper, '_send_command') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "interview",
  "content": {
    "questions": [
      {
        "question": "Which database do you prefer?",
        "focus": "data_model",
        "options": ["PostgreSQL", "MySQL", "SQLite"],
        "context": "Need to choose database"
      },
      {
        "question": "How should errors be handled?",
        "focus": "exception_handling",
        "options": ["Return error codes", "Throw exceptions"],
        "context": "Error handling strategy"
      }
    ]
  }
}
```'''

            analysis = ProjectAnalysis(project_path="/test", languages=["Python"])
            questions = wrapper.generate_interview_questions(
                user_goal="Add user authentication",
                project_analysis=analysis
            )

            assert len(questions) == 2
            assert questions[0].focus == InterviewFocus.DATA_MODEL
            assert questions[1].focus == InterviewFocus.EXCEPTION_HANDLING

    def test_generate_interview_questions_invalid_focus(self, wrapper, mock_scribe):
        """Test handling invalid focus value."""
        with patch.object(wrapper, '_send_command') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "interview",
  "content": {
    "questions": [
      {
        "question": "Test question?",
        "focus": "invalid_focus",
        "options": [],
        "context": ""
      }
    ]
  }
}
```'''

            questions = wrapper.generate_interview_questions("Test goal")

            # Should fall back to ARCHITECTURE
            assert len(questions) == 1
            assert questions[0].focus == InterviewFocus.ARCHITECTURE

    def test_generate_feature_spec(self, wrapper, mock_scribe):
        """Test feature spec generation."""
        with patch.object(wrapper, '_send_command') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "feature_spec",
  "content": {
    "feature_name": "user_auth",
    "summary": "User authentication system",
    "requirements": ["Login", "Logout"],
    "architecture_decisions": ["Use JWT"],
    "data_model": {"User": {"id": "int"}},
    "error_handling": ["Invalid credentials"],
    "conventions": ["REST API"],
    "constraints": [],
    "out_of_scope": ["Social login"]
  }
}
```'''

            question = InterviewQuestion(
                question="Which auth?",
                focus=InterviewFocus.ARCHITECTURE
            )
            answers = [InterviewAnswer(question=question, answer="JWT")]

            spec = wrapper.generate_feature_spec(
                user_goal="Add auth",
                interview_answers=answers,
                project_analysis=None
            )

            assert spec.feature_name == "user_auth"
            assert "Login" in spec.requirements
            assert "Use JWT" in spec.architecture_decisions

    def test_decompose_to_tasks(self, wrapper, mock_scribe):
        """Test task decomposition."""
        with patch.object(wrapper, '_send_command') as mock_send:
            mock_send.return_value = '''```json
{
  "type": "task_list",
  "content": {
    "tasks": [
      {
        "id": "task_001",
        "title": "Create User model",
        "goal": "Define User class",
        "files_to_modify": ["models/user.py"],
        "expected_logic": "Create User dataclass",
        "dependencies": [],
        "acceptance_criteria": ["Class exists"]
      },
      {
        "id": "task_002",
        "title": "Create login endpoint",
        "goal": "POST /login",
        "files_to_modify": ["routes/auth.py"],
        "expected_logic": "Validate credentials",
        "dependencies": ["task_001"],
        "acceptance_criteria": ["Endpoint works"]
      }
    ]
  }
}
```'''

            spec = FeatureSpec(
                feature_name="user_auth",
                summary="Auth system",
                requirements=["Login"]
            )

            tasks = wrapper.decompose_to_tasks(spec)

            assert len(tasks) == 2
            assert tasks[0].id == "task_001"
            assert tasks[1].dependencies == ["task_001"]

    def test_run_plan_mode_complete_flow(self, wrapper, mock_scribe):
        """Test complete plan mode flow."""
        # Mock all internal methods
        with patch.object(wrapper, 'analyze_project') as mock_analyze, \
             patch.object(wrapper, 'generate_interview_questions') as mock_questions, \
             patch.object(wrapper, 'generate_feature_spec') as mock_spec, \
             patch.object(wrapper, 'decompose_to_tasks') as mock_tasks:

            mock_analyze.return_value = ProjectAnalysis(
                project_path="/test",
                languages=["Python"]
            )

            mock_questions.return_value = [
                InterviewQuestion(
                    question="Which database?",
                    focus=InterviewFocus.DATA_MODEL,
                    options=["PostgreSQL", "SQLite"]
                )
            ]

            mock_spec.return_value = FeatureSpec(
                feature_name="test_feature",
                summary="Test"
            )

            mock_tasks.return_value = [
                PlanTask(id="t1", title="Task 1", goal="Goal 1")
            ]

            # Callbacks
            def on_question(q):
                return "PostgreSQL"

            def on_spec_review(spec):
                return True

            def on_tasks_review(tasks):
                return True

            result = wrapper.run_plan_mode(
                user_goal="Add database",
                on_question=on_question,
                on_spec_review=on_spec_review,
                on_tasks_review=on_tasks_review
            )

            assert result.project_analysis is not None
            assert result.feature_spec.feature_name == "test_feature"
            assert len(result.tasks) == 1
            assert result.approved is True

    def test_run_plan_mode_spec_not_approved(self, wrapper, mock_scribe):
        """Test plan mode when spec is not approved."""
        with patch.object(wrapper, 'analyze_project') as mock_analyze, \
             patch.object(wrapper, 'generate_interview_questions') as mock_questions, \
             patch.object(wrapper, 'generate_feature_spec') as mock_spec:

            mock_analyze.return_value = ProjectAnalysis(project_path="/test")
            mock_questions.return_value = []
            mock_spec.return_value = FeatureSpec(feature_name="test", summary="Test")

            def on_spec_review(spec):
                return False  # Not approved

            result = wrapper.run_plan_mode(
                user_goal="Test",
                on_spec_review=on_spec_review
            )

            assert result.approved is False
            assert len(result.tasks) == 0

    def test_run_plan_mode_analysis_failure(self, wrapper, mock_scribe):
        """Test plan mode handles analysis failure gracefully."""
        with patch.object(wrapper, 'analyze_project') as mock_analyze, \
             patch.object(wrapper, 'generate_interview_questions') as mock_questions, \
             patch.object(wrapper, 'generate_feature_spec') as mock_spec, \
             patch.object(wrapper, 'decompose_to_tasks') as mock_tasks:

            mock_analyze.side_effect = Exception("Analysis failed")
            mock_questions.return_value = []
            mock_spec.return_value = FeatureSpec(feature_name="test", summary="Test")
            mock_tasks.return_value = []

            result = wrapper.run_plan_mode(user_goal="Test")

            # Should have fallback analysis
            assert result.project_analysis is not None
            assert result.project_analysis.project_path == wrapper.working_directory

    def test_context_manager(self, mock_scribe, mock_shutil):
        """Test wrapper works as context manager."""
        with GeminiCLIWrapper() as wrapper:
            assert wrapper is not None
        # close() should be called automatically

    def test_log_callback(self, mock_scribe, mock_shutil):
        """Test log callback is called."""
        logs = []

        def on_output(msg):
            logs.append(msg)

        wrapper = GeminiCLIWrapper(
            on_output=on_output,
            verbose=True
        )

        with patch.object(wrapper, '_send_command') as mock_send:
            mock_send.return_value = '{"type": "analysis", "content": {}}'
            wrapper.analyze_project()

        wrapper.close()

        # Should have logged some messages
        assert len(logs) > 0

    def test_error_callback(self, mock_scribe, mock_shutil, mock_pexpect):
        """Test error callback is called on failure."""
        errors = []

        def on_error(msg, metadata):
            errors.append({"msg": msg, "metadata": metadata})

        wrapper = GeminiCLIWrapper(on_error=on_error)

        # Force a failure by making spawn raise exception
        mock_pexpect["spawn"].side_effect = Exception("Spawn failed")

        result = wrapper._start_interactive_session()

        assert result is False
        assert len(errors) > 0

        wrapper.close()


class TestCommanderCLIConstitution:
    """Tests for COMMANDER_CLI_CONSTITUTION constant."""

    def test_constitution_exists(self):
        assert COMMANDER_CLI_CONSTITUTION is not None
        assert len(COMMANDER_CLI_CONSTITUTION) > 0

    def test_constitution_contains_key_principles(self):
        assert "최소주의" in COMMANDER_CLI_CONSTITUTION
        assert "명확성" in COMMANDER_CLI_CONSTITUTION
        assert "검증" in COMMANDER_CLI_CONSTITUTION

    def test_constitution_contains_json_format(self):
        assert "JSON" in COMMANDER_CLI_CONSTITUTION
        assert "type" in COMMANDER_CLI_CONSTITUTION

    def test_constitution_contains_role_info(self):
        assert "Commander" in COMMANDER_CLI_CONSTITUTION or "지휘관" in COMMANDER_CLI_CONSTITUTION
