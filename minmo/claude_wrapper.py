"""
Claude Code Wrapper - Minmo-Engine의 실행 부대 (Worker)
pexpect를 사용하여 Claude Code CLI를 제어하고 작업을 실행
"""

import os
import sys
import json
import re
import time
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from minmo.scribe_mcp import log_event, update_todo, init_database

# 플랫폼별 pexpect 임포트
if sys.platform == "win32":
    from pexpect.popen_spawn import PopenSpawn as PexpectSpawn
    PEXPECT_ENCODING = "utf-8"
else:
    import pexpect
    PexpectSpawn = pexpect.spawn
    PEXPECT_ENCODING = "utf-8"


# ============================================================
# 헌법 (Constitution) - Worker용 프롬프트 프리픽스
# ============================================================
WORKER_CONSTITUTION = """[MINMO WORKER 원칙 - 반드시 준수]
1. 주석 최소화: 코드가 스스로 설명하게 하라. 복잡한 로직에만 최소한의 주석.
2. 요구사항만 구현: 요청받지 않은 기능, 리팩토링, "개선"은 절대 금지.
3. 추측 금지: 불확실하면 멈추고 질문하라. 가정하지 마라.
4. 과잉 설계 금지: "확장성", "나중을 위해" 같은 이유로 복잡도를 높이지 마라.
5. 에러 핸들링은 경계에서만: 내부 코드는 실패하게 두라.

[작업 지시]
"""


# ============================================================
# 데이터 클래스
# ============================================================
class TaskStatus(Enum):
    """태스크 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskResult:
    """태스크 실행 결과"""
    task_id: str
    status: TaskStatus
    output: str = ""
    error: str | None = None
    files_modified: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    retry_count: int = 0


@dataclass
class ExecutionContext:
    """실행 컨텍스트"""
    working_directory: str
    mcp_config_path: str | None = None
    timeout_seconds: int = 300
    max_retries: int = 3
    verbose: bool = False


# ============================================================
# Claude Code Wrapper 클래스
# ============================================================
class ClaudeCodeWrapper:
    """
    Claude Code CLI 래퍼 - 작업자 역할

    pexpect를 사용하여 Claude Code를 제어하고,
    GeminiWrapper가 세운 TaskPlan을 실행합니다.
    """

    def __init__(
        self,
        working_directory: str | None = None,
        mcp_config_path: str | None = None,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        on_output: Callable[[str], None] | None = None,
        on_error: Callable[[str, dict], None] | None = None,
        commander_callback: Callable[[str, dict], dict] | None = None,
        verbose: bool = False
    ):
        """
        ClaudeCodeWrapper 초기화

        Args:
            working_directory: 작업 디렉토리 (기본: 현재 디렉토리)
            mcp_config_path: MCP 설정 파일 경로 (기본: 자동 생성)
            timeout_seconds: 명령 타임아웃 (초)
            max_retries: 최대 재시도 횟수
            on_output: 출력 콜백 함수
            on_error: 에러 콜백 함수
            commander_callback: 지휘관(Gemini) 피드백 콜백
            verbose: 상세 출력 여부
        """
        self.working_directory = working_directory or os.getcwd()
        self.mcp_config_path = mcp_config_path
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.on_output = on_output
        self.on_error = on_error
        self.commander_callback = commander_callback
        self.verbose = verbose

        # Claude CLI 경로 확인
        self.claude_path = self._find_claude_cli()

        # MCP 설정 자동 생성
        if self.mcp_config_path is None:
            self.mcp_config_path = self._create_mcp_config()

        # 프로세스 핸들
        self._process: PexpectSpawn | None = None

        # Scribe DB 초기화
        try:
            init_database()
        except Exception:
            pass

    def _find_claude_cli(self) -> str:
        """Claude CLI 실행 파일을 찾습니다."""
        # 환경변수에서 먼저 확인
        claude_path = os.environ.get("CLAUDE_CLI_PATH")
        if claude_path and os.path.exists(claude_path):
            return claude_path

        # PATH에서 검색
        claude_cmd = "claude.exe" if sys.platform == "win32" else "claude"
        found = shutil.which(claude_cmd)
        if found:
            return found

        # 일반적인 설치 경로 확인
        common_paths = []
        if sys.platform == "win32":
            common_paths = [
                os.path.expandvars(r"%LOCALAPPDATA%\Programs\claude\claude.exe"),
                os.path.expandvars(r"%APPDATA%\npm\claude.cmd"),
                r"C:\Program Files\Claude\claude.exe",
            ]
        else:
            common_paths = [
                "/usr/local/bin/claude",
                os.path.expanduser("~/.local/bin/claude"),
                "/opt/claude/bin/claude",
            ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        # 찾지 못한 경우 기본값 반환 (실행 시 에러 발생)
        return claude_cmd

    def _create_mcp_config(self) -> str:
        """Scribe MCP 서버용 설정 파일을 생성합니다."""
        config_dir = Path(self.working_directory) / ".claude"
        config_dir.mkdir(exist_ok=True)

        config_path = config_dir / "mcp_config.json"

        # Scribe MCP 서버 경로
        scribe_path = Path(__file__).parent / "scribe_mcp.py"

        config = {
            "mcpServers": {
                "minmo-scribe": {
                    "command": sys.executable,
                    "args": [str(scribe_path)],
                    "env": {
                        "PYTHONPATH": str(Path(__file__).parent.parent)
                    }
                }
            }
        }

        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))

        self._log("MCP 설정 파일 생성", {"path": str(config_path)})

        return str(config_path)

    def _log(self, content: str, metadata: dict | None = None) -> None:
        """Scribe를 통해 로그를 기록합니다."""
        try:
            log_event(
                agent="worker",
                content=content,
                metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None
            )
        except Exception:
            pass

        if self.verbose and self.on_output:
            self.on_output(f"[Worker] {content}")

    def _build_prompt(self, task: dict[str, Any]) -> str:
        """태스크에 대한 프롬프트를 생성합니다."""
        task_title = task.get("title", "작업")
        task_description = task.get("description", "")
        task_type = task.get("type", "implementation")
        files_affected = task.get("files_affected", [])

        prompt_parts = [WORKER_CONSTITUTION]

        prompt_parts.append(f"## 태스크: {task_title}\n")

        if task_description:
            prompt_parts.append(f"### 설명\n{task_description}\n")

        if task_type:
            prompt_parts.append(f"### 유형: {task_type}\n")

        if files_affected:
            prompt_parts.append(f"### 대상 파일\n")
            for f in files_affected:
                prompt_parts.append(f"- {f}\n")

        prompt_parts.append("\n### 실행 지침")
        prompt_parts.append("- 위 태스크만 수행하세요.")
        prompt_parts.append("- 완료되면 변경 사항을 요약해주세요.")
        prompt_parts.append("- 문제가 발생하면 즉시 보고하세요.")

        return "\n".join(prompt_parts)

    def _parse_output(self, output: str) -> dict[str, Any]:
        """Claude Code 출력을 파싱합니다."""
        result = {
            "raw_output": output,
            "files_modified": [],
            "errors": [],
            "warnings": [],
            "summary": ""
        }

        # 파일 수정 감지 패턴
        file_patterns = [
            r"(?:Created|Modified|Updated|Wrote|Edited)\s+[`'\"]?([^\s`'\"]+)[`'\"]?",
            r"(?:Writing to|Saving)\s+[`'\"]?([^\s`'\"]+)[`'\"]?",
            r"File:\s*([^\s]+)",
        ]

        for pattern in file_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            result["files_modified"].extend(matches)

        # 에러 감지 패턴
        error_patterns = [
            r"(?:Error|ERROR|Failed|FAILED):\s*(.+)",
            r"(?:Exception|Traceback).*?:\s*(.+)",
            r"(?:Cannot|Could not|Unable to)\s+(.+)",
        ]

        for pattern in error_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE | re.MULTILINE)
            result["errors"].extend(matches)

        # 경고 감지
        warning_patterns = [
            r"(?:Warning|WARN):\s*(.+)",
            r"(?:Deprecated|deprecated):\s*(.+)",
        ]

        for pattern in warning_patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            result["warnings"].extend(matches)

        # 중복 제거
        result["files_modified"] = list(set(result["files_modified"]))
        result["errors"] = list(set(result["errors"]))
        result["warnings"] = list(set(result["warnings"]))

        # 요약 추출 (마지막 몇 줄)
        lines = output.strip().split("\n")
        if lines:
            result["summary"] = "\n".join(lines[-5:])

        return result

    def _run_claude_command(self, prompt: str, task_id: str) -> tuple[str, bool]:
        """
        Claude Code CLI를 실행합니다.

        Args:
            prompt: 실행할 프롬프트
            task_id: 태스크 ID

        Returns:
            (출력 문자열, 성공 여부)
        """
        # 명령어 구성
        cmd_args = [self.claude_path]

        # MCP 설정 추가
        if self.mcp_config_path and os.path.exists(self.mcp_config_path):
            cmd_args.extend(["--mcp-config", self.mcp_config_path])

        # 비대화형 모드 + 프롬프트
        cmd_args.extend(["-p", prompt])

        # 출력 형식
        cmd_args.extend(["--output-format", "text"])

        self._log(f"Claude Code 실행: {task_id}", {
            "command": " ".join(cmd_args[:3]) + " ...",
            "working_dir": self.working_directory
        })

        output_buffer = []
        success = True

        try:
            if sys.platform == "win32":
                # Windows: PopenSpawn 사용
                process = PopenSpawn(
                    " ".join(f'"{arg}"' if " " in arg else arg for arg in cmd_args),
                    encoding=PEXPECT_ENCODING,
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory
                )
            else:
                # Unix: pexpect.spawn 사용
                process = pexpect.spawn(
                    cmd_args[0],
                    cmd_args[1:],
                    encoding=PEXPECT_ENCODING,
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory
                )

            self._process = process

            # 출력 읽기 (실시간)
            while True:
                try:
                    # EOF 또는 타임아웃까지 읽기
                    if sys.platform == "win32":
                        line = process.readline()
                    else:
                        process.expect(["\n", pexpect.EOF, pexpect.TIMEOUT], timeout=1)
                        line = process.before

                    if not line:
                        # EOF 확인
                        if sys.platform == "win32":
                            if not process.isalive():
                                break
                        else:
                            if process.eof():
                                break
                        continue

                    line_str = line.strip() if isinstance(line, str) else line.decode().strip()

                    if line_str:
                        output_buffer.append(line_str)

                        # 실시간 출력 콜백
                        if self.on_output:
                            self.on_output(line_str)

                        # 주기적으로 Scribe에 기록
                        if len(output_buffer) % 10 == 0:
                            self._log(f"진행 중: {task_id}", {
                                "lines_processed": len(output_buffer)
                            })

                except Exception as e:
                    if "EOF" in str(e) or "timeout" in str(e).lower():
                        break
                    # 다른 예외는 무시하고 계속
                    continue

            # 프로세스 종료 대기
            if sys.platform == "win32":
                process.wait()
                exit_code = process.exitstatus or 0
            else:
                process.close()
                exit_code = process.exitstatus or 0

            if exit_code != 0:
                success = False
                self._log(f"Claude Code 종료 (코드: {exit_code})", {"task_id": task_id})

        except Exception as e:
            success = False
            error_msg = f"Claude Code 실행 오류: {str(e)}"
            output_buffer.append(error_msg)
            self._log(error_msg, {"error": str(e), "task_id": task_id})

            if self.on_error:
                self.on_error(error_msg, {"task_id": task_id, "exception": type(e).__name__})

        finally:
            self._process = None

        return "\n".join(output_buffer), success

    def _handle_error(
        self,
        task: dict[str, Any],
        error_output: str,
        retry_count: int
    ) -> dict[str, Any] | None:
        """
        에러를 처리하고 필요시 지휘관에게 피드백을 요청합니다.

        Args:
            task: 실패한 태스크
            error_output: 에러 출력
            retry_count: 현재 재시도 횟수

        Returns:
            수정된 태스크 (None이면 재시도 중단)
        """
        task_id = task.get("id", "unknown")

        self._log(f"에러 처리: {task_id}", {
            "retry_count": retry_count,
            "error_preview": error_output[:500]
        })

        # 지휘관 콜백이 있으면 피드백 요청
        if self.commander_callback:
            try:
                feedback_request = {
                    "type": "error_feedback",
                    "task": task,
                    "error": error_output,
                    "retry_count": retry_count,
                    "request": "이 에러를 분석하고 태스크를 수정하거나 대안을 제시해주세요."
                }

                response = self.commander_callback("analyze_error", feedback_request)

                if response.get("action") == "retry_with_modification":
                    modified_task = response.get("modified_task", task)
                    self._log("지휘관이 태스크 수정", {"task_id": task_id})
                    return modified_task

                elif response.get("action") == "skip":
                    self._log("지휘관이 태스크 스킵 지시", {"task_id": task_id})
                    return None

                elif response.get("action") == "abort":
                    self._log("지휘관이 작업 중단 지시", {"task_id": task_id})
                    raise RuntimeError(f"작업 중단: {response.get('reason', '지휘관 지시')}")

            except Exception as e:
                self._log(f"지휘관 피드백 요청 실패: {e}", {"task_id": task_id})

        # 기본 동작: 원래 태스크로 재시도
        return task

    def execute(self, task: dict[str, Any]) -> TaskResult:
        """
        단일 태스크를 실행합니다.

        Args:
            task: 실행할 태스크

        Returns:
            TaskResult: 실행 결과
        """
        task_id = task.get("id", f"task_{datetime.now().strftime('%H%M%S')}")
        start_time = time.time()

        self._log(f"태스크 시작: {task_id}", {"title": task.get("title")})

        # 상태 업데이트
        try:
            update_todo(task_id, "in_progress")
        except Exception:
            pass

        current_task = task
        retry_count = 0
        last_output = ""
        last_error = None

        while retry_count <= self.max_retries:
            # 프롬프트 생성
            prompt = self._build_prompt(current_task)

            # Claude Code 실행
            output, success = self._run_claude_command(prompt, task_id)
            last_output = output

            # 출력 파싱
            parsed = self._parse_output(output)

            if success and not parsed["errors"]:
                # 성공
                duration = time.time() - start_time

                try:
                    update_todo(task_id, "completed")
                except Exception:
                    pass

                self._log(f"태스크 완료: {task_id}", {
                    "duration_seconds": duration,
                    "files_modified": parsed["files_modified"]
                })

                return TaskResult(
                    task_id=task_id,
                    status=TaskStatus.COMPLETED,
                    output=output,
                    files_modified=parsed["files_modified"],
                    duration_seconds=duration,
                    retry_count=retry_count
                )

            # 실패 처리
            last_error = "\n".join(parsed["errors"]) if parsed["errors"] else "Unknown error"

            if self.on_error:
                self.on_error(last_error, {"task_id": task_id, "retry": retry_count})

            retry_count += 1

            if retry_count <= self.max_retries:
                self._log(f"재시도 {retry_count}/{self.max_retries}: {task_id}")

                # 에러 처리 및 태스크 수정 시도
                modified_task = self._handle_error(current_task, output, retry_count)

                if modified_task is None:
                    # 스킵 지시
                    break

                current_task = modified_task
                time.sleep(1)  # 재시도 전 잠시 대기

        # 최종 실패
        duration = time.time() - start_time

        try:
            update_todo(task_id, "failed")
        except Exception:
            pass

        self._log(f"태스크 실패: {task_id}", {
            "duration_seconds": duration,
            "retry_count": retry_count,
            "error": last_error
        })

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            output=last_output,
            error=last_error,
            duration_seconds=duration,
            retry_count=retry_count
        )

    def execute_plan(
        self,
        tasks: list[dict[str, Any]],
        stop_on_failure: bool = True
    ) -> list[TaskResult]:
        """
        여러 태스크로 구성된 계획을 실행합니다.

        Args:
            tasks: 실행할 태스크 목록
            stop_on_failure: 실패 시 중단 여부

        Returns:
            TaskResult 목록
        """
        results = []

        self._log(f"계획 실행 시작: {len(tasks)}개 태스크")

        for i, task in enumerate(tasks):
            task_id = task.get("id", f"task_{i}")

            self._log(f"태스크 {i + 1}/{len(tasks)} 실행: {task_id}")

            result = self.execute(task)
            results.append(result)

            if result.status == TaskStatus.FAILED and stop_on_failure:
                self._log(f"계획 중단: 태스크 {task_id} 실패")
                break

        # 요약
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        failed = sum(1 for r in results if r.status == TaskStatus.FAILED)

        self._log(f"계획 실행 완료", {
            "total": len(results),
            "completed": completed,
            "failed": failed
        })

        return results

    def cancel(self) -> bool:
        """현재 실행 중인 작업을 취소합니다."""
        if self._process is not None:
            try:
                if sys.platform == "win32":
                    self._process.kill(signal=9)
                else:
                    self._process.terminate(force=True)
                self._log("작업 취소됨")
                return True
            except Exception as e:
                self._log(f"취소 실패: {e}")
                return False
        return False

    def is_running(self) -> bool:
        """현재 작업 실행 중인지 확인합니다."""
        if self._process is None:
            return False

        if sys.platform == "win32":
            return self._process.isalive()
        else:
            return self._process.isalive()

    def get_claude_version(self) -> str | None:
        """Claude CLI 버전을 반환합니다."""
        try:
            result = subprocess.run(
                [self.claude_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.stdout.strip()
        except Exception:
            return None
