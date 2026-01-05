"""
Minmo-Engine: MCP 통합 자동화 프레임워크

Claude Code와 Gemini를 연동한 멀티 에이전트 오케스트레이션 시스템
"""

__version__ = "1.0.0"
__author__ = "Minmo Team"

from minmo.orchestrator import MinmoOrchestrator, ScribeMCP
from minmo.gemini_wrapper import (
    GeminiWrapper,
    ProjectAnalysis,
    ProjectType,
    TaskPlan,
    ClarificationQuestion,
    COMMANDER_CONSTITUTION,
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
]
