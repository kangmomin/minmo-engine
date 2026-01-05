"""
Tests for minmo.gemini_wrapper module.
"""

import os
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from minmo.gemini_wrapper import (
    ProjectType,
    ProjectAnalysis,
    ClarificationQuestion,
    TaskPlan,
    GeminiWrapper,
    COMMANDER_CONSTITUTION,
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


class TestGeminiWrapper:
    """Tests for GeminiWrapper class."""

    @pytest.fixture
    def mock_genai(self):
        """Mock the google.generativeai module."""
        with patch("minmo.gemini_wrapper.genai") as mock:
            mock_model = MagicMock()
            mock_chat = MagicMock()
            mock_response = MagicMock()
            mock_response.text = '{"type": "analysis", "content": {}, "requires_confirmation": false, "next_action": "wait_for_input"}'

            mock_chat.send_message.return_value = mock_response
            mock_chat.history = []
            mock_model.start_chat.return_value = mock_chat

            mock.GenerativeModel.return_value = mock_model
            mock.GenerationConfig.return_value = {}

            yield mock

    @pytest.fixture
    def mock_scribe(self):
        """Mock the scribe_mcp functions."""
        with patch("minmo.gemini_wrapper.log_event") as mock_log, \
             patch("minmo.gemini_wrapper.init_database") as mock_init:
            mock_log.return_value = True
            mock_init.return_value = None
            yield {"log_event": mock_log, "init_database": mock_init}

    @pytest.fixture
    def wrapper(self, mock_genai, mock_scribe):
        """Create a GeminiWrapper instance with mocked dependencies."""
        os.environ["GEMINI_API_KEY"] = "test-api-key"
        wrapper = GeminiWrapper()
        yield wrapper
        os.environ.pop("GEMINI_API_KEY", None)

    def test_init_without_api_key_raises_error(self, mock_genai, mock_scribe):
        os.environ.pop("GEMINI_API_KEY", None)

        with pytest.raises(ValueError) as excinfo:
            GeminiWrapper()

        assert "API 키가 필요합니다" in str(excinfo.value)

    def test_init_with_api_key_param(self, mock_genai, mock_scribe):
        wrapper = GeminiWrapper(api_key="direct-api-key")
        assert wrapper.api_key == "direct-api-key"

    def test_init_with_env_api_key(self, mock_genai, mock_scribe):
        os.environ["GEMINI_API_KEY"] = "env-api-key"
        wrapper = GeminiWrapper()
        assert wrapper.api_key == "env-api-key"
        os.environ.pop("GEMINI_API_KEY", None)

    def test_init_with_custom_model(self, mock_genai, mock_scribe):
        os.environ["GEMINI_API_KEY"] = "test-key"
        wrapper = GeminiWrapper(model_name="gemini-1.5-pro")
        assert wrapper.model_name == "gemini-1.5-pro"
        os.environ.pop("GEMINI_API_KEY", None)

    def test_init_with_custom_temperature(self, mock_genai, mock_scribe):
        os.environ["GEMINI_API_KEY"] = "test-key"
        wrapper = GeminiWrapper(temperature=0.7)
        assert wrapper.temperature == 0.7
        os.environ.pop("GEMINI_API_KEY", None)

    def test_parse_json_response_with_json_block(self, wrapper):
        response = '''Here is the analysis:
```json
{"type": "test", "content": {"key": "value"}}
```
'''
        result = wrapper._parse_json_response(response)
        assert result["type"] == "test"
        assert result["content"]["key"] == "value"

    def test_parse_json_response_with_code_block(self, wrapper):
        response = '''Result:
```
{"type": "test", "content": {}}
```
'''
        result = wrapper._parse_json_response(response)
        assert result["type"] == "test"

    def test_parse_json_response_plain_json(self, wrapper):
        response = '{"type": "plain", "content": {}}'
        result = wrapper._parse_json_response(response)
        assert result["type"] == "plain"

    def test_parse_json_response_invalid_json(self, wrapper):
        response = "This is not JSON at all"
        result = wrapper._parse_json_response(response)
        assert result["type"] == "text"
        assert result["content"] == response

    def test_init_project_existing(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "analysis",
            "content": {
                "project_type": "existing",
                "detected_languages": ["Python"],
                "detected_frameworks": ["FastAPI"],
                "detected_databases": ["SQLite"],
                "structure_summary": "A backend project",
                "recommendations": [],
                "confidence": 0.9,
            },
            "requires_confirmation": True,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        file_list = ["main.py", "models.py", "routes.py"]
        result = wrapper.init_project("/test/project", file_list)

        assert result.project_type == ProjectType.EXISTING
        assert "Python" in result.detected_languages
        assert result.confidence == 0.9

    def test_init_project_new(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "recommendation",
            "content": {
                "project_type": "new",
                "detected_languages": [],
                "detected_frameworks": [],
                "detected_databases": [],
                "recommendations": ["Use Python for backend"],
                "confidence": 0.5,
            },
            "requires_confirmation": True,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        result = wrapper.init_project("/new/project", None)

        assert result.project_type == ProjectType.NEW

    def test_clarify_goal_no_clarification_needed(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "clarification",
            "content": {
                "needs_clarification": False,
                "understood_requirements": {
                    "summary": "Create a login page",
                    "scope": ["Frontend", "Backend"],
                    "constraints": [],
                    "assumptions": []
                }
            },
            "requires_confirmation": True,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        result = wrapper.clarify_goal("Create a login page")

        assert result["original_goal"] == "Create a login page"
        assert result["final_requirements"] is not None
        assert result["final_requirements"]["summary"] == "Create a login page"

    def test_clarify_goal_with_questions(self, wrapper, mock_genai):
        mock_responses = [
            MagicMock(text=json.dumps({
                "type": "clarification",
                "content": {
                    "needs_clarification": True,
                    "questions": [{
                        "question": "Which authentication method?",
                        "options": ["OAuth", "JWT", "Session"],
                        "context": "Need to know auth approach",
                        "required": True
                    }]
                },
                "next_action": "clarify"
            })),
            MagicMock(text=json.dumps({
                "type": "clarification",
                "content": {
                    "needs_clarification": False,
                    "understood_requirements": {
                        "summary": "JWT authentication",
                        "scope": ["Backend auth"],
                        "constraints": [],
                        "assumptions": []
                    }
                },
                "next_action": "wait_for_input"
            }))
        ]
        wrapper.chat.send_message.side_effect = mock_responses

        answers_given = []

        def on_question(q):
            answers_given.append(q.question)
            return "JWT"

        result = wrapper.clarify_goal("Add authentication", on_question=on_question)

        assert len(result["clarifications"]) == 1
        assert "Which authentication method?" in answers_given
        assert result["final_requirements"]["summary"] == "JWT authentication"

    def test_plan(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "plan",
            "content": {
                "goal": "Implement user login",
                "tasks": [
                    {
                        "id": "task_001",
                        "title": "Create User model",
                        "description": "Define User dataclass",
                        "type": "implementation",
                        "priority": "high",
                        "dependencies": []
                    },
                    {
                        "id": "task_002",
                        "title": "Implement login endpoint",
                        "description": "POST /login",
                        "type": "implementation",
                        "priority": "high",
                        "dependencies": ["task_001"]
                    }
                ],
                "estimated_complexity": "medium",
                "prerequisites": [],
                "risks": []
            },
            "requires_confirmation": True,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        result = wrapper.plan("Implement user login")

        assert result.goal == "Implement user login"
        assert len(result.tasks) == 2
        assert result.tasks[0]["id"] == "task_001"
        assert result.estimated_complexity == "medium"

    def test_plan_with_project_analysis(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "plan",
            "content": {
                "goal": "Add feature",
                "tasks": [],
                "estimated_complexity": "low",
                "prerequisites": [],
                "risks": []
            },
            "requires_confirmation": True,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        analysis = ProjectAnalysis(
            project_type=ProjectType.EXISTING,
            detected_languages=["Python"],
            detected_frameworks=["FastAPI"],
        )

        result = wrapper.plan("Add feature", project_analysis=analysis)

        assert result is not None
        call_args = wrapper.chat.send_message.call_args[0][0]
        assert "Python" in call_args
        assert "FastAPI" in call_args

    def test_analyze_code(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "analysis",
            "content": {
                "summary": "A utility function module",
                "purpose": "Helper functions for data processing",
                "key_functions": ["process_data", "validate_input"],
                "dependencies": ["json", "typing"],
                "issues": [],
                "suggestions": []
            },
            "requires_confirmation": False,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        code = """
def process_data(data: dict) -> dict:
    return {k: v.strip() for k, v in data.items()}
"""
        result = wrapper.analyze_code(code, "/test/utils.py")

        assert result["summary"] == "A utility function module"
        assert "process_data" in result["key_functions"]

    def test_analyze_code_with_question(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "analysis",
            "content": {
                "summary": "Analysis with specific question",
                "purpose": "Answered",
                "key_functions": [],
                "dependencies": [],
                "issues": ["Found the issue"],
                "suggestions": []
            },
            "requires_confirmation": False,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        result = wrapper.analyze_code(
            "def buggy(): pass",
            "/test/file.py",
            question="Why is this function not working?"
        )

        assert "Found the issue" in result["issues"]

    def test_review_changes(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "review",
            "content": {
                "summary": "Added new feature",
                "approval": "approved",
                "comments": [
                    {
                        "type": "praise",
                        "location": "main.py:10",
                        "message": "Good implementation"
                    }
                ],
                "security_concerns": [],
                "breaking_changes": []
            },
            "requires_confirmation": False,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        diff = """
+ def new_feature():
+     return True
"""
        result = wrapper.review_changes(diff, context="Adding new feature")

        assert result["approval"] == "approved"
        assert len(result["comments"]) == 1

    def test_review_changes_with_issues(self, wrapper, mock_genai):
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "type": "review",
            "content": {
                "summary": "Potential security issue",
                "approval": "changes_requested",
                "comments": [
                    {
                        "type": "issue",
                        "location": "auth.py:25",
                        "message": "SQL injection vulnerability"
                    }
                ],
                "security_concerns": ["SQL injection"],
                "breaking_changes": []
            },
            "requires_confirmation": False,
            "next_action": "wait_for_input"
        })
        wrapper.chat.send_message.return_value = mock_response

        diff = "- cursor.execute(f'SELECT * FROM users WHERE id={user_id}')"
        result = wrapper.review_changes(diff)

        assert result["approval"] == "changes_requested"
        assert "SQL injection" in result["security_concerns"]

    def test_get_conversation_history(self, wrapper, mock_genai):
        mock_msg1 = MagicMock()
        mock_msg1.role = "user"
        mock_msg1.parts = [MagicMock(text="Hello")]

        mock_msg2 = MagicMock()
        mock_msg2.role = "model"
        mock_msg2.parts = [MagicMock(text="Hi there")]

        wrapper.chat.history = [mock_msg1, mock_msg2]

        history = wrapper.get_conversation_history()

        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "model"

    def test_reset_conversation(self, wrapper, mock_genai):
        wrapper.chat.history = [MagicMock()]
        wrapper.reset_conversation()

        wrapper.model.start_chat.assert_called()

    def test_log_callback(self, mock_genai, mock_scribe):
        os.environ["GEMINI_API_KEY"] = "test-key"
        logs = []

        def on_log(agent, content, metadata):
            logs.append({"agent": agent, "content": content})

        wrapper = GeminiWrapper(on_log=on_log)

        mock_response = MagicMock()
        mock_response.text = '{"type": "test", "content": {}}'
        wrapper.chat.send_message.return_value = mock_response

        wrapper.init_project("/test", [])

        assert len(logs) > 0
        assert logs[0]["agent"] == "commander"

        os.environ.pop("GEMINI_API_KEY", None)


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
