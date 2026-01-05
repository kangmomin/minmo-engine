"""
Minmo-Engine: MCP 통합 자동화 프레임워크

Claude Code와 Gemini를 연동한 멀티 에이전트 오케스트레이션 시스템
"""

__version__ = "1.0.3"
__author__ = "Minmo Team"

from minmo.orchestrator import MinmoOrchestrator, ScribeMCP
from minmo.gemini_wrapper import (
    GeminiWrapper,
    ProjectAnalysis,
    ProjectType,
    TaskPlan,
    ClarificationQuestion,
    COMMANDER_CONSTITUTION,
    # Plan Mode
    InterviewFocus,
    InterviewQuestion,
    InterviewAnswer,
    FeatureSpec,
    PlanTask,
    PlanModeResult,
)
from minmo.claude_wrapper import (
    ClaudeCodeWrapper,
    TaskResult,
    TaskStatus,
    ExecutionContext,
    WORKER_CONSTITUTION,
)
from minmo.indexer import (
    CodeIndexer,
    CodeSymbol,
    FileIndex,
    ImportInfo,
    SearchResult,
    IndexStorage,
    SQLiteFTS5Storage,
    VectorStorage,
    PythonASTParser,
)
from minmo.gemini_cli_wrapper import (
    GeminiCLIWrapper,
    ProjectAnalysis as CLIProjectAnalysis,
    InterviewQuestion as CLIInterviewQuestion,
    InterviewAnswer as CLIInterviewAnswer,
    FeatureSpec as CLIFeatureSpec,
    PlanTask as CLIPlanTask,
    PlanModeResult as CLIPlanModeResult,
    COMMANDER_CLI_CONSTITUTION,
)

__all__ = [
    # Orchestrator
    "MinmoOrchestrator",
    "ScribeMCP",
    # Commander (Gemini)
    "GeminiWrapper",
    "ProjectAnalysis",
    "ProjectType",
    "TaskPlan",
    "ClarificationQuestion",
    "COMMANDER_CONSTITUTION",
    # Plan Mode
    "InterviewFocus",
    "InterviewQuestion",
    "InterviewAnswer",
    "FeatureSpec",
    "PlanTask",
    "PlanModeResult",
    # Worker (Claude Code)
    "ClaudeCodeWrapper",
    "TaskResult",
    "TaskStatus",
    "ExecutionContext",
    "WORKER_CONSTITUTION",
    # Indexer (RAG)
    "CodeIndexer",
    "CodeSymbol",
    "FileIndex",
    "ImportInfo",
    "SearchResult",
    "IndexStorage",
    "SQLiteFTS5Storage",
    "VectorStorage",
    "PythonASTParser",
    # Commander CLI (Gemini CLI)
    "GeminiCLIWrapper",
    "CLIProjectAnalysis",
    "CLIInterviewQuestion",
    "CLIInterviewAnswer",
    "CLIFeatureSpec",
    "CLIPlanTask",
    "CLIPlanModeResult",
    "COMMANDER_CLI_CONSTITUTION",
]
