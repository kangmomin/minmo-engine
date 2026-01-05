"""
Tests for Plan Mode feature.
"""

import json
from unittest.mock import MagicMock, patch
from dataclasses import asdict

import pytest

from minmo.gemini_wrapper import (
    GeminiWrapper,
    InterviewFocus,
    InterviewQuestion,
    InterviewAnswer,
    FeatureSpec,
    PlanTask,
    PlanModeResult,
)


class TestPlanModeDataClasses:
    """Tests for Plan Mode data classes."""

    def test_interview_focus_enum(self):
        assert InterviewFocus.ARCHITECTURE.value == "architecture"
        assert InterviewFocus.DATA_MODEL.value == "data_model"
        assert InterviewFocus.EXCEPTION_HANDLING.value == "exception_handling"
        assert InterviewFocus.CONVENTION.value == "convention"
        assert InterviewFocus.INTEGRATION.value == "integration"
        assert InterviewFocus.TESTING.value == "testing"

    def test_interview_question_creation(self):
        question = InterviewQuestion(
            question="How should we handle authentication?",
            focus=InterviewFocus.ARCHITECTURE,
            options=["JWT", "Session", "OAuth"],
            context="We need to decide on auth strategy",
            follow_up_hint="Consider security implications"
        )

        assert question.question == "How should we handle authentication?"
        assert question.focus == InterviewFocus.ARCHITECTURE
        assert len(question.options) == 3
        assert "JWT" in question.options

    def test_interview_question_defaults(self):
        question = InterviewQuestion(
            question="Test question",
            focus=InterviewFocus.TESTING
        )

        assert question.options == []
        assert question.context == ""
        assert question.follow_up_hint == ""

    def test_interview_answer(self):
        question = InterviewQuestion(
            question="Which DB?",
            focus=InterviewFocus.DATA_MODEL
        )
        answer = InterviewAnswer(question=question, answer="PostgreSQL")

        assert answer.question.question == "Which DB?"
        assert answer.answer == "PostgreSQL"

    def test_feature_spec_creation(self):
        spec = FeatureSpec(
            feature_name="user_auth",
            summary="User authentication feature",
            requirements=["Login", "Logout", "Register"],
            architecture_decisions=["Use JWT tokens"],
            data_model={"User": {"id": "int", "email": "str"}},
            error_handling=["Invalid credentials: return 401"],
            conventions=["snake_case for functions"],
            constraints=["Must work with existing DB"],
            out_of_scope=["Social login"]
        )

        assert spec.feature_name == "user_auth"
        assert len(spec.requirements) == 3
        assert "Login" in spec.requirements
        assert spec.data_model["User"]["id"] == "int"

    def test_feature_spec_defaults(self):
        spec = FeatureSpec(
            feature_name="test",
            summary="Test feature"
        )

        assert spec.requirements == []
        assert spec.architecture_decisions == []
        assert spec.data_model == {}

    def test_plan_task_creation(self):
        task = PlanTask(
            id="task_001",
            title="Implement login",
            goal="Create login endpoint",
            files_to_modify=["src/auth/login.py", "src/routes.py"],
            expected_logic="POST /login endpoint with JWT response",
            dependencies=[],
            acceptance_criteria=["Returns JWT on success", "Returns 401 on failure"]
        )

        assert task.id == "task_001"
        assert task.title == "Implement login"
        assert len(task.files_to_modify) == 2
        assert len(task.acceptance_criteria) == 2

    def test_plan_mode_result(self):
        spec = FeatureSpec(feature_name="test", summary="Test")
        task = PlanTask(id="task_001", title="Test task", goal="Test goal")

        result = PlanModeResult(
            feature_spec=spec,
            tasks=[task],
            interview_history=[],
            approved=True,
            spec_file_path="/path/to/spec.md"
        )

        assert result.feature_spec.feature_name == "test"
        assert len(result.tasks) == 1
        assert result.approved is True


class TestGeminiWrapperPlanMode:
    """Tests for GeminiWrapper Plan Mode methods."""

    @pytest.fixture
    def mock_gemini_model(self):
        """Mock Gemini model."""
        with patch("minmo.gemini_wrapper.genai") as mock_genai:
            mock_model = MagicMock()
            mock_chat = MagicMock()
            mock_model.start_chat.return_value = mock_chat
            mock_genai.GenerativeModel.return_value = mock_model

            yield {
                "genai": mock_genai,
                "model": mock_model,
                "chat": mock_chat
            }

    @pytest.fixture
    def mock_scribe(self):
        """Mock Scribe functions."""
        with patch("minmo.gemini_wrapper.log_event") as mock_log, \
             patch("minmo.gemini_wrapper.init_database"):
            mock_log.return_value = True
            yield mock_log

    def test_generate_interview_questions(self, mock_gemini_model, mock_scribe):
        """Test interview question generation."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "interview_questions",
            "content": {
                "questions": [
                    {
                        "question": "How should we structure the API?",
                        "focus": "architecture",
                        "options": ["REST", "GraphQL"],
                        "context": "API design decision",
                        "follow_up_hint": "Consider client needs"
                    },
                    {
                        "question": "What data format for storage?",
                        "focus": "data_model",
                        "options": ["JSON", "Relational"],
                        "context": "Storage decision",
                        "follow_up_hint": "Consider query patterns"
                    }
                ]
            }
        })
        mock_gemini_model["chat"].send_message.return_value = mock_response

        wrapper = GeminiWrapper(api_key="test-key")
        questions = wrapper.generate_interview_questions(
            user_goal="Create a REST API",
            project_context={"language": "Python"}
        )

        assert len(questions) == 2
        assert questions[0].focus == InterviewFocus.ARCHITECTURE
        assert "API" in questions[0].question

    def test_generate_feature_spec(self, mock_gemini_model, mock_scribe):
        """Test feature spec generation."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "feature_spec",
            "content": {
                "feature_name": "user_auth",
                "summary": "User authentication system",
                "requirements": ["Login", "Logout"],
                "architecture_decisions": ["Use JWT"],
                "data_model": {"User": {"id": "int"}},
                "error_handling": ["401 for invalid credentials"],
                "conventions": ["snake_case"],
                "constraints": ["Postgres only"],
                "out_of_scope": ["Social login"]
            }
        })
        mock_gemini_model["chat"].send_message.return_value = mock_response

        wrapper = GeminiWrapper(api_key="test-key")

        question = InterviewQuestion(
            question="Which auth method?",
            focus=InterviewFocus.ARCHITECTURE,
            options=["JWT", "Session"]
        )
        answers = [InterviewAnswer(question=question, answer="JWT")]

        spec = wrapper.generate_feature_spec(
            user_goal="Add authentication",
            interview_answers=answers
        )

        assert spec.feature_name == "user_auth"
        assert "Login" in spec.requirements
        assert "Use JWT" in spec.architecture_decisions

    def test_decompose_to_tasks(self, mock_gemini_model, mock_scribe):
        """Test task decomposition."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "task_decomposition",
            "content": {
                "tasks": [
                    {
                        "id": "task_001",
                        "title": "Create User model",
                        "goal": "Define User entity",
                        "files_to_modify": ["src/models/user.py"],
                        "expected_logic": "SQLAlchemy model with id, email, password_hash",
                        "dependencies": [],
                        "acceptance_criteria": ["Model created", "Migration generated"]
                    },
                    {
                        "id": "task_002",
                        "title": "Implement login endpoint",
                        "goal": "POST /login",
                        "files_to_modify": ["src/routes/auth.py"],
                        "expected_logic": "Validate credentials, return JWT",
                        "dependencies": ["task_001"],
                        "acceptance_criteria": ["Endpoint returns JWT"]
                    }
                ]
            }
        })
        mock_gemini_model["chat"].send_message.return_value = mock_response

        wrapper = GeminiWrapper(api_key="test-key")

        spec = FeatureSpec(
            feature_name="user_auth",
            summary="Auth system",
            requirements=["Login", "Logout"]
        )

        tasks = wrapper.decompose_to_tasks(spec)

        assert len(tasks) == 2
        assert tasks[0].id == "task_001"
        assert tasks[1].dependencies == ["task_001"]

    def test_validate_against_conventions(self, mock_gemini_model, mock_scribe):
        """Test convention validation."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "convention_validation",
            "content": {
                "approved": True,
                "violations": [],
                "warnings": ["Consider adding type hints"],
                "recommendations": ["Follow existing patterns"]
            }
        })
        mock_gemini_model["chat"].send_message.return_value = mock_response

        wrapper = GeminiWrapper(api_key="test-key")

        tasks = [
            PlanTask(
                id="task_001",
                title="Test task",
                goal="Test goal",
                files_to_modify=["src/test.py"]
            )
        ]

        result = wrapper.validate_against_conventions(
            tasks=tasks,
            existing_conventions=["snake_case for functions", "Type hints required"]
        )

        assert result["approved"] is True
        assert len(result["violations"]) == 0

    def test_validate_against_conventions_with_violations(self, mock_gemini_model, mock_scribe):
        """Test convention validation with violations."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "convention_validation",
            "content": {
                "approved": False,
                "violations": [
                    {
                        "task_id": "task_001",
                        "violation": "Uses camelCase",
                        "convention": "snake_case for functions",
                        "severity": "error",
                        "suggestion": "Rename to snake_case"
                    }
                ],
                "warnings": [],
                "recommendations": []
            }
        })
        mock_gemini_model["chat"].send_message.return_value = mock_response

        wrapper = GeminiWrapper(api_key="test-key")

        tasks = [
            PlanTask(id="task_001", title="Test", goal="Test")
        ]

        result = wrapper.validate_against_conventions(
            tasks=tasks,
            existing_conventions=["snake_case for functions"]
        )

        assert result["approved"] is False
        assert len(result["violations"]) == 1

    def test_run_plan_mode_full_flow(self, mock_gemini_model, mock_scribe):
        """Test complete plan mode flow."""
        # Setup mock responses for each step
        responses = [
            # 1. Interview questions
            json.dumps({
                "type": "interview_questions",
                "content": {
                    "questions": [
                        {
                            "question": "Which database?",
                            "focus": "data_model",
                            "options": ["PostgreSQL", "MySQL"],
                            "context": "DB choice",
                            "follow_up_hint": ""
                        }
                    ]
                }
            }),
            # 2. Feature spec
            json.dumps({
                "type": "feature_spec",
                "content": {
                    "feature_name": "test_feature",
                    "summary": "Test feature",
                    "requirements": ["Req1"],
                    "architecture_decisions": [],
                    "data_model": {},
                    "error_handling": [],
                    "conventions": [],
                    "constraints": [],
                    "out_of_scope": []
                }
            }),
            # 3. Task decomposition
            json.dumps({
                "type": "task_decomposition",
                "content": {
                    "tasks": [
                        {
                            "id": "task_001",
                            "title": "Task 1",
                            "goal": "Goal 1",
                            "files_to_modify": ["file.py"],
                            "expected_logic": "Logic",
                            "dependencies": [],
                            "acceptance_criteria": ["Done"]
                        }
                    ]
                }
            }),
            # 4. Convention validation
            json.dumps({
                "type": "convention_validation",
                "content": {
                    "approved": True,
                    "violations": [],
                    "warnings": [],
                    "recommendations": []
                }
            })
        ]

        response_iter = iter(responses)

        def get_next_response(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.text = next(response_iter)
            return mock_resp

        mock_gemini_model["chat"].send_message.side_effect = get_next_response

        wrapper = GeminiWrapper(api_key="test-key")

        # Mock callbacks
        on_question = MagicMock(return_value="PostgreSQL")
        on_spec_review = MagicMock(return_value=True)
        on_tasks_review = MagicMock(return_value=True)

        result = wrapper.run_plan_mode(
            user_goal="Create a test feature",
            existing_conventions=["snake_case"],
            on_question=on_question,
            on_spec_review=on_spec_review,
            on_tasks_review=on_tasks_review
        )

        assert result.approved is True
        assert result.feature_spec.feature_name == "test_feature"
        assert len(result.tasks) == 1
        on_question.assert_called()
        on_spec_review.assert_called()
        on_tasks_review.assert_called()

    def test_run_plan_mode_spec_rejected(self, mock_gemini_model, mock_scribe):
        """Test plan mode when spec is rejected."""
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "interview_questions",
            "content": {"questions": []}
        })
        mock_gemini_model["chat"].send_message.return_value = mock_response

        # Second call for feature spec
        spec_response = MagicMock()
        spec_response.text = json.dumps({
            "type": "feature_spec",
            "content": {
                "feature_name": "test",
                "summary": "Test"
            }
        })

        mock_gemini_model["chat"].send_message.side_effect = [mock_response, spec_response]

        wrapper = GeminiWrapper(api_key="test-key")

        # Reject the spec
        on_spec_review = MagicMock(return_value=False)

        result = wrapper.run_plan_mode(
            user_goal="Test",
            on_spec_review=on_spec_review
        )

        assert result.approved is False
        assert result.tasks == []


class TestScribePlanModeTools:
    """Tests for Scribe MCP Plan Mode tools."""

    @pytest.fixture
    def mock_indexer(self):
        """Mock CodeIndexer."""
        with patch("minmo.scribe_mcp.get_code_indexer") as mock_get:
            mock_indexer = MagicMock()
            mock_indexer.get_project_overview.return_value = {
                "project_path": "/test/project",
                "stats": {
                    "total_files": 10,
                    "languages": {"Python": 8, "JavaScript": 2}
                },
                "main_classes": [
                    {"name": "UserService", "file": "src/services/user.py", "docstring": "User service"}
                ],
                "main_functions": [
                    {
                        "name": "get_user",
                        "file": "src/services/user.py",
                        "signature": "def get_user(user_id: int) -> User:",
                        "docstring": "Get user by ID"
                    },
                    {
                        "name": "validate_email",
                        "file": "src/utils/validators.py",
                        "signature": "def validate_email(email)",
                        "docstring": None
                    }
                ]
            }
            mock_get.return_value = mock_indexer
            yield mock_indexer

    def test_get_conventions(self, mock_indexer):
        """Test convention analysis logic."""
        # Access the underlying function from the MCP tool wrapper
        import minmo.scribe_mcp as scribe_module

        # Get the function directly from the module's globals before decoration
        # or test via the tool's fn attribute
        from minmo.scribe_mcp import get_conventions

        # FastMCP wraps the function - access the underlying fn
        if hasattr(get_conventions, 'fn'):
            result = get_conventions.fn()
        else:
            # If it's a direct function (testing environment)
            result = get_conventions()

        assert result["success"] is True
        assert "conventions" in result
        assert "naming_patterns" in result

    def test_get_architecture_info(self, mock_indexer):
        """Test architecture info analysis."""
        from minmo.scribe_mcp import get_architecture_info

        if hasattr(get_architecture_info, 'fn'):
            result = get_architecture_info.fn()
        else:
            result = get_architecture_info()

        assert result["success"] is True
        assert "directories" in result
        assert "layers" in result

    def test_save_feature_spec(self, tmp_path, monkeypatch):
        """Test feature spec file saving."""
        from pathlib import Path as RealPath

        # Monkeypatch Path.cwd to return tmp_path
        monkeypatch.setattr("minmo.scribe_mcp.Path.cwd", lambda: tmp_path)

        from minmo.scribe_mcp import save_feature_spec

        if hasattr(save_feature_spec, 'fn'):
            result = save_feature_spec.fn(
                feature_name="test_feature",
                content="# Test Feature\n\nTest content"
            )
        else:
            result = save_feature_spec(
                feature_name="test_feature",
                content="# Test Feature\n\nTest content"
            )

        assert result["success"] is True
        assert "file_path" in result

        # Verify file was created
        spec_file = tmp_path / "info" / "specs" / "test_feature.md"
        assert spec_file.exists()
        assert "Test Feature" in spec_file.read_text()

    def test_load_feature_spec(self, tmp_path, monkeypatch):
        """Test feature spec loading."""
        # Create a test spec file
        spec_dir = tmp_path / "info" / "specs"
        spec_dir.mkdir(parents=True)
        spec_file = spec_dir / "test_feature.md"
        spec_file.write_text("# Test Feature\n\nContent here")

        # Monkeypatch Path.cwd
        monkeypatch.setattr("minmo.scribe_mcp.Path.cwd", lambda: tmp_path)

        from minmo.scribe_mcp import load_feature_spec

        if hasattr(load_feature_spec, 'fn'):
            result = load_feature_spec.fn("test_feature")
        else:
            result = load_feature_spec("test_feature")

        assert result["success"] is True
        assert "content" in result
        assert "Test Feature" in result["content"]

    def test_load_feature_spec_not_found(self, tmp_path, monkeypatch):
        """Test loading non-existent feature spec."""
        monkeypatch.setattr("minmo.scribe_mcp.Path.cwd", lambda: tmp_path)

        from minmo.scribe_mcp import load_feature_spec

        if hasattr(load_feature_spec, 'fn'):
            result = load_feature_spec.fn("nonexistent")
        else:
            result = load_feature_spec("nonexistent")

        assert result["success"] is False
        assert "error" in result


class TestCLIPlanCommand:
    """Tests for CLI plan command."""

    def test_generate_spec_markdown(self):
        """Test markdown generation from spec."""
        from minmo.cli import _generate_spec_markdown

        spec = FeatureSpec(
            feature_name="user_auth",
            summary="User authentication",
            requirements=["Login", "Logout"],
            architecture_decisions=["JWT tokens"],
            error_handling=["401 for invalid creds"],
            conventions=["snake_case"],
            out_of_scope=["Social login"]
        )

        task = PlanTask(
            id="task_001",
            title="Implement login",
            goal="Create login endpoint",
            files_to_modify=["auth.py"],
            expected_logic="Validate and return JWT",
            acceptance_criteria=["Returns JWT"]
        )

        result = PlanModeResult(
            feature_spec=spec,
            tasks=[task],
            interview_history=[],
            approved=True
        )

        markdown = _generate_spec_markdown(spec, result)

        assert "# user_auth" in markdown
        assert "## 요약" in markdown
        assert "User authentication" in markdown
        assert "## 요구사항" in markdown
        assert "- Login" in markdown
        assert "## 태스크 목록" in markdown
        assert "### task_001: Implement login" in markdown
