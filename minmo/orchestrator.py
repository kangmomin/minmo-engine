"""
Minmo Orchestrator - 지휘관(Gemini)과 작업자(Claude Code)를 조율하는 엔진
"""

import json
from typing import Any, Protocol, Callable
from dataclasses import asdict

from minmo.gemini_wrapper import GeminiWrapper, TaskPlan, ProjectAnalysis
from minmo.claude_wrapper import ClaudeCodeWrapper, TaskResult, TaskStatus
from minmo.scribe_mcp import log_event, get_state, update_todo


class ScribeProtocol(Protocol):
    """Scribe MCP 서버 인터페이스"""
    def save_plan(self, plan: Any) -> None: ...
    def get_next_task(self) -> Any: ...
    def log_result(self, result: Any) -> None: ...


class CommanderProtocol(Protocol):
    """지휘관 (Gemini) 인터페이스"""
    def plan(self, goal: str) -> TaskPlan: ...


class WorkerProtocol(Protocol):
    """작업자 (Claude Code) 인터페이스"""
    def execute(self, task: Any) -> Any: ...


class ScribeMCP:
    """Scribe MCP 클라이언트 래퍼 - Redis/SQLite 연동"""

    def __init__(self):
        self._plan: TaskPlan | None = None
        self._tasks: list[dict[str, Any]] = []
        self._current_index = 0

    def save_plan(self, plan: TaskPlan | dict | list) -> None:
        """계획을 저장하고 Scribe에 기록합니다."""
        self._plan = plan

        if isinstance(plan, TaskPlan):
            self._tasks = plan.tasks
            plan_data = asdict(plan)
        elif isinstance(plan, dict) and "tasks" in plan:
            self._tasks = plan["tasks"]
            plan_data = plan
        elif isinstance(plan, list):
            self._tasks = plan
            plan_data = {"tasks": plan}
        else:
            self._tasks = []
            plan_data = {}

        self._current_index = 0

        # Scribe에 계획 기록
        try:
            log_event(
                agent="orchestrator",
                content=f"작업 계획 저장: {len(self._tasks)}개 태스크",
                metadata=json.dumps(plan_data, ensure_ascii=False, default=str)
            )

            # Redis에 각 태스크 등록
            for task in self._tasks:
                task_id = task.get("id", f"task_{self._tasks.index(task)}")
                update_todo(task_id, "pending")
        except Exception:
            pass  # 로깅 실패해도 계속 진행

    def get_next_task(self) -> dict[str, Any] | None:
        """다음 작업을 반환하고 상태를 업데이트합니다."""
        if self._current_index < len(self._tasks):
            task = self._tasks[self._current_index]
            task_id = task.get("id", f"task_{self._current_index}")

            # 상태 업데이트
            try:
                update_todo(task_id, "in_progress")
                log_event(
                    agent="orchestrator",
                    content=f"작업 시작: {task.get('title', task_id)}",
                    metadata=json.dumps({"task_id": task_id}, ensure_ascii=False)
                )
            except Exception:
                pass

            self._current_index += 1
            return task
        return None

    def log_result(self, result: Any) -> None:
        """작업 결과를 Scribe에 기록합니다."""
        try:
            task_id = None
            if isinstance(result, dict):
                task_id = result.get("task_id") or result.get("task", {}).get("id")

            log_event(
                agent="orchestrator",
                content=f"작업 완료: {task_id or 'unknown'}",
                metadata=json.dumps(result, ensure_ascii=False, default=str)
            )

            if task_id:
                status = result.get("status", "completed")
                update_todo(task_id, status)
        except Exception:
            pass


class MinmoOrchestrator:
    """Minmo 오케스트레이터 - 전체 작업 흐름 관리"""

    def __init__(
        self,
        working_directory: str | None = None,
        on_output: Callable[[str], None] | None = None,
        on_error: Callable[[str, dict], None] | None = None,
        verbose: bool = False
    ):
        """
        오케스트레이터 초기화

        Args:
            working_directory: 작업 디렉토리
            on_output: 출력 콜백
            on_error: 에러 콜백
            verbose: 상세 출력 여부
        """
        self.scribe = ScribeMCP()
        self.commander: GeminiWrapper | None = None
        self.worker: ClaudeCodeWrapper | None = None
        self._is_complete = False
        self._working_directory = working_directory
        self._on_output = on_output
        self._on_error = on_error
        self._verbose = verbose

    def _ensure_commander(self) -> GeminiWrapper:
        """Commander(Gemini)가 초기화되었는지 확인하고 반환합니다."""
        if self.commander is None:
            self.commander = GeminiWrapper(
                working_directory=self._working_directory,
                verbose=self._verbose
            )
        return self.commander

    def _ensure_worker(self) -> ClaudeCodeWrapper:
        """Worker(Claude Code)가 초기화되었는지 확인하고 반환합니다."""
        if self.worker is None:
            self.worker = ClaudeCodeWrapper(
                working_directory=self._working_directory,
                on_output=self._on_output,
                on_error=self._on_error,
                commander_callback=self._commander_feedback,
                verbose=self._verbose
            )
        return self.worker

    def _commander_feedback(self, action: str, data: dict) -> dict[str, Any]:
        """
        지휘관(Gemini)에게 피드백을 요청합니다.

        Args:
            action: 요청 액션 (analyze_error 등)
            data: 요청 데이터

        Returns:
            지휘관의 응답
        """
        commander = self._ensure_commander()

        if action == "analyze_error":
            task = data.get("task", {})
            error = data.get("error", "")
            retry_count = data.get("retry_count", 0)

            # Gemini에게 에러 분석 요청
            try:
                analysis = commander.analyze_code(
                    code=error,
                    file_path="error_log",
                    question=f"이 에러를 분석하고 태스크 '{task.get('title', 'unknown')}'를 수정해야 할지 판단해주세요."
                )

                # 분석 결과에 따라 액션 결정
                if "skip" in str(analysis).lower() or retry_count >= 3:
                    return {"action": "skip", "reason": "최대 재시도 초과 또는 스킵 권장"}

                # 수정된 태스크 생성
                modified_task = task.copy()
                modified_task["description"] = (
                    f"{task.get('description', '')}\n\n"
                    f"[이전 시도 에러]\n{error[:500]}\n\n"
                    f"[지휘관 분석]\n{analysis.get('summary', '분석 결과 없음')}"
                )

                return {
                    "action": "retry_with_modification",
                    "modified_task": modified_task,
                    "analysis": analysis
                }

            except Exception as e:
                log_event(
                    agent="orchestrator",
                    content=f"지휘관 피드백 요청 실패: {e}",
                    metadata=json.dumps({"error": str(e)})
                )
                return {"action": "retry", "reason": "피드백 요청 실패, 원래 태스크로 재시도"}

        return {"action": "continue"}

    def is_complete(self) -> bool:
        """모든 작업이 완료되었는지 확인합니다."""
        return self._is_complete

    def init_project(self, project_path: str, file_list: list[str] | None = None) -> ProjectAnalysis:
        """
        프로젝트를 초기화하고 분석합니다.

        Args:
            project_path: 프로젝트 경로
            file_list: 파일 목록

        Returns:
            프로젝트 분석 결과
        """
        commander = self._ensure_commander()
        return commander.init_project(project_path, file_list)

    def clarify_goal(self, user_goal: str, on_question=None) -> dict[str, Any]:
        """
        모호한 요구사항을 명확화합니다.

        Args:
            user_goal: 사용자 목표
            on_question: 질문 콜백 함수

        Returns:
            명확화된 요구사항
        """
        commander = self._ensure_commander()
        return commander.clarify_goal(user_goal, on_question=on_question)

    def start_loop(
        self,
        user_goal: str,
        skip_clarification: bool = False,
        project_analysis: ProjectAnalysis | None = None,
        stop_on_failure: bool = False
    ) -> dict[str, Any]:
        """
        오케스트레이션 루프를 시작합니다.

        Args:
            user_goal: 사용자의 목표
            skip_clarification: 요구사항 명확화 건너뛰기
            project_analysis: 프로젝트 분석 결과
            stop_on_failure: 실패 시 중단 여부

        Returns:
            실행 결과
        """
        commander = self._ensure_commander()
        worker = self._ensure_worker()
        results: list[TaskResult] = []

        log_event(
            agent="orchestrator",
            content=f"오케스트레이션 시작: {user_goal[:100]}",
            metadata=json.dumps({"goal": user_goal})
        )

        # 0. 요구사항 명확화 (선택적)
        clarified = None
        if not skip_clarification:
            try:
                clarified = commander.clarify_goal(user_goal)
                if clarified.get("final_requirements"):
                    user_goal = clarified["final_requirements"].get("summary", user_goal)
            except Exception:
                pass  # 명확화 실패 시 원래 목표 사용

        # 1. 지휘관이 기획 (BMad Analyst)
        plan = commander.plan(user_goal, project_analysis=project_analysis)
        self.scribe.save_plan(plan)

        # 2. 작업자가 실행 (BMad Dev)
        failed_tasks = []

        while not self.is_complete():
            task = self.scribe.get_next_task()

            if task is None:
                self._is_complete = True
                break

            # Claude Code로 태스크 실행
            result = worker.execute(task)
            results.append(result)

            # 결과 기록
            self.scribe.log_result({
                "task_id": result.task_id,
                "status": result.status.value,
                "output": result.output[:1000] if result.output else "",
                "error": result.error,
                "files_modified": result.files_modified,
                "duration_seconds": result.duration_seconds
            })

            # 실패 처리
            if result.status == TaskStatus.FAILED:
                failed_tasks.append(result)

                if stop_on_failure:
                    log_event(
                        agent="orchestrator",
                        content=f"작업 중단: 태스크 {result.task_id} 실패",
                        metadata=json.dumps({"error": result.error})
                    )
                    self._is_complete = True
                    break

        # 결과 요약
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == TaskStatus.FAILED)

        final_status = "completed" if failed == 0 else ("partial" if completed > 0 else "failed")

        log_event(
            agent="orchestrator",
            content=f"오케스트레이션 완료: {completed}/{len(results)} 성공",
            metadata=json.dumps({
                "completed": completed,
                "failed": failed,
                "total": len(results),
                "status": final_status
            })
        )

        return {
            "goal": user_goal,
            "clarification": clarified,
            "plan": asdict(plan) if isinstance(plan, TaskPlan) else plan,
            "results": [
                {
                    "task_id": r.task_id,
                    "status": r.status.value,
                    "files_modified": r.files_modified,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error
                }
                for r in results
            ],
            "summary": {
                "total": len(results),
                "completed": completed,
                "failed": failed
            },
            "status": final_status
        }

    def analyze_code(self, code: str, file_path: str, question: str | None = None) -> dict[str, Any]:
        """코드를 분석합니다."""
        commander = self._ensure_commander()
        return commander.analyze_code(code, file_path, question)

    def review_changes(self, diff: str, context: str | None = None) -> dict[str, Any]:
        """코드 변경사항을 리뷰합니다."""
        commander = self._ensure_commander()
        return commander.review_changes(diff, context)
