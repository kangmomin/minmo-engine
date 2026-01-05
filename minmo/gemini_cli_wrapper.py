"""
Gemini CLI Wrapper - Minmo-Engine의 지휘관 (Commander)
pexpect를 사용하여 @google/gemini-cli를 제어하고 계획을 수립
"""

import os
import sys
import json
import re
import time
import shutil
from pathlib import Path
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from minmo.scribe_mcp import log_event, init_database

# 플랫폼별 pexpect 임포트
if sys.platform == "win32":
    from pexpect.popen_spawn import PopenSpawn as PexpectSpawn
    PEXPECT_ENCODING = "utf-8"
else:
    import pexpect
    PexpectSpawn = pexpect.spawn
    PEXPECT_ENCODING = "utf-8"


# ============================================================
# 헌법 (Constitution) - Commander용 시스템 프롬프트
# ============================================================
COMMANDER_CLI_CONSTITUTION = """[MINMO COMMANDER 원칙 - 반드시 준수]

당신은 Minmo-Engine의 **지휘관(Commander)**입니다.
개발 작업을 계획하고, 요구사항을 분석하며, 작업자에게 명확한 지시를 내립니다.

## 핵심 원칙 (절대 위반 금지)

### 1. 최소주의 원칙
- **요청받은 것만 개발하라.** 추가 기능, "있으면 좋을" 기능은 절대 제안하지 마라.
- **추측하지 마라.** 불확실하면 반드시 질문하라.
- **과도한 주석 금지.** 코드가 스스로 설명하게 하라.

### 2. 명확성 원칙
- **모호한 요구사항은 즉시 명확화하라.** 가정하지 말고 질문하라.
- **구체적인 작업 단위로 분해하라.** "로그인 기능 구현"이 아니라 세부 단계로.
- **기술 선택에는 근거를 제시하라.**

### 3. 검증 원칙
- **기존 코드를 먼저 분석하라.** 중복 구현을 피하라.
- **변경 전 영향 범위를 파악하라.** 사이드 이펙트를 예측하라.
- **사용자 확인 없이 큰 변경을 하지 마라.**

## 응답 형식
항상 JSON 형식으로 응답하세요:
```json
{
  "type": "plan|interview|analysis|task_list",
  "content": { ... }
}
```
"""


# ============================================================
# 데이터 클래스
# ============================================================
class InterviewFocus(Enum):
    """인터뷰 질문 초점 영역"""
    ARCHITECTURE = "architecture"
    DATA_MODEL = "data_model"
    EXCEPTION_HANDLING = "exception_handling"
    CONVENTION = "convention"
    INTEGRATION = "integration"
    TESTING = "testing"


@dataclass
class ProjectAnalysis:
    """프로젝트 분석 결과"""
    project_path: str
    languages: list[str] = field(default_factory=list)
    frameworks: list[str] = field(default_factory=list)
    structure_summary: str = ""
    key_files: list[str] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class InterviewQuestion:
    """인터뷰 질문"""
    question: str
    focus: InterviewFocus
    options: list[str] = field(default_factory=list)
    context: str = ""


@dataclass
class InterviewAnswer:
    """인터뷰 답변"""
    question: InterviewQuestion
    answer: str


@dataclass
class FeatureSpec:
    """기능 기획서"""
    feature_name: str
    summary: str
    requirements: list[str] = field(default_factory=list)
    architecture_decisions: list[str] = field(default_factory=list)
    data_model: dict[str, Any] = field(default_factory=dict)
    error_handling: list[str] = field(default_factory=list)
    conventions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)


@dataclass
class PlanTask:
    """분해된 태스크"""
    id: str
    title: str
    goal: str
    files_to_modify: list[str] = field(default_factory=list)
    expected_logic: str = ""
    dependencies: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)


@dataclass
class PlanModeResult:
    """Plan Mode 결과"""
    project_analysis: ProjectAnalysis | None = None
    feature_spec: FeatureSpec | None = None
    tasks: list[PlanTask] = field(default_factory=list)
    interview_history: list[InterviewAnswer] = field(default_factory=list)
    approved: bool = False
    spec_file_path: str = ""


# ============================================================
# Gemini CLI Wrapper 클래스
# ============================================================
class GeminiCLIWrapper:
    """
    Gemini CLI (@google/gemini-cli) 래퍼 - 지휘관 역할

    pexpect를 사용하여 Gemini CLI를 제어하고,
    프로젝트 분석, 인터뷰, 계획 수립을 수행합니다.
    """

    # Gemini CLI 프롬프트 패턴
    PROMPT_PATTERN = r"[>❯►][\s]*$"
    READY_PATTERNS = [
        r"[>❯►]\s*$",
        r"gemini>\s*$",
        r"\?\s*$",  # 질문 프롬프트
    ]

    def __init__(
        self,
        working_directory: str | None = None,
        timeout_seconds: int = 120,
        on_output: Callable[[str], None] | None = None,
        on_error: Callable[[str, dict], None] | None = None,
        verbose: bool = False
    ):
        """
        GeminiCLIWrapper 초기화

        Args:
            working_directory: 작업 디렉토리 (기본: 현재 디렉토리)
            timeout_seconds: 명령 타임아웃 (초)
            on_output: 출력 콜백 함수
            on_error: 에러 콜백 함수
            verbose: 상세 출력 여부
        """
        self.working_directory = working_directory or os.getcwd()
        self.timeout_seconds = timeout_seconds
        self.on_output = on_output
        self.on_error = on_error
        self.verbose = verbose

        # Gemini CLI 경로 확인
        self.gemini_path = self._find_gemini_cli()

        # 프로세스 핸들
        self._process: PexpectSpawn | None = None
        self._is_started = False

        # Scribe DB 초기화
        try:
            init_database()
        except Exception:
            pass

    def _find_gemini_cli(self) -> str:
        """
        Gemini CLI 실행 파일을 찾습니다.

        Raises:
            RuntimeError: Gemini CLI가 설치되어 있지 않은 경우
        """
        # 환경변수에서 먼저 확인
        gemini_path = os.environ.get("GEMINI_CLI_PATH")
        if gemini_path and os.path.exists(gemini_path):
            return gemini_path

        # PATH에서 검색
        gemini_cmd = "gemini.cmd" if sys.platform == "win32" else "gemini"
        found = shutil.which(gemini_cmd)
        if found:
            return found

        # npm global 경로 확인
        if sys.platform == "win32":
            npm_paths = [
                os.path.expandvars(r"%APPDATA%\npm\gemini.cmd"),
                os.path.expandvars(r"%LOCALAPPDATA%\npm\gemini.cmd"),
            ]
        else:
            npm_paths = [
                "/usr/local/bin/gemini",
                os.path.expanduser("~/.npm-global/bin/gemini"),
                os.path.expanduser("~/node_modules/.bin/gemini"),
            ]

        for path in npm_paths:
            if os.path.exists(path):
                return path

        # CLI를 찾지 못한 경우 친절한 안내 메시지와 함께 예외 발생
        raise RuntimeError(
            "\n"
            "Gemini CLI를 찾을 수 없습니다.\n"
            "\n"
            "다음 명령어로 Gemini CLI를 설치해주세요:\n"
            "  npm install -g @google/gemini-cli\n"
            "\n"
            "설치 후 로그인해주세요:\n"
            "  gemini login\n"
            "\n"
            "또는 GEMINI_CLI_PATH 환경변수로 경로를 직접 지정할 수 있습니다."
        )

    def _log(self, content: str, metadata: dict | None = None) -> None:
        """Scribe를 통해 로그를 기록합니다."""
        try:
            log_event(
                agent="commander",
                content=content,
                metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None
            )
        except Exception:
            pass

        if self.verbose and self.on_output:
            self.on_output(f"[Commander] {content}")

    def check_login_status(self) -> bool:
        """Gemini CLI 로그인 상태를 확인합니다."""
        try:
            import subprocess
            result = subprocess.run(
                [self.gemini_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False

    def _start_interactive_session(self) -> bool:
        """Gemini CLI 대화형 세션을 시작합니다."""
        if self._is_started and self._process:
            return True

        self._log("Gemini CLI 세션 시작", {"path": self.gemini_path})

        try:
            if sys.platform == "win32":
                self._process = PexpectSpawn(
                    f'"{self.gemini_path}"',
                    encoding=PEXPECT_ENCODING,
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory
                )
            else:
                self._process = pexpect.spawn(
                    self.gemini_path,
                    encoding=PEXPECT_ENCODING,
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory
                )

            # 초기 프롬프트 대기
            time.sleep(2)
            self._is_started = True
            self._log("Gemini CLI 세션 시작됨")
            return True

        except Exception as e:
            self._log(f"Gemini CLI 시작 실패: {str(e)}")
            if self.on_error:
                self.on_error(f"Gemini CLI 시작 실패: {str(e)}", {"error": str(e)})
            return False

    def _send_command(self, command: str, wait_for_complete: bool = True) -> str:
        """Gemini CLI에 명령을 보내고 응답을 받습니다."""
        if not self._process:
            if not self._start_interactive_session():
                return ""

        output_buffer = []

        try:
            # 명령 전송
            self._process.sendline(command)
            time.sleep(0.5)

            # 응답 수집
            start_time = time.time()
            while True:
                elapsed = time.time() - start_time
                if elapsed > self.timeout_seconds:
                    break

                try:
                    if sys.platform == "win32":
                        line = self._process.readline()
                        if line:
                            line = line.strip()
                            if line and line != command:
                                output_buffer.append(line)
                                if self.on_output and self.verbose:
                                    self.on_output(line)

                        if not self._process.isalive():
                            break

                        # 프롬프트 패턴 확인
                        if output_buffer:
                            last_line = output_buffer[-1]
                            if re.search(self.PROMPT_PATTERN, last_line):
                                output_buffer.pop()  # 프롬프트 제거
                                break
                    else:
                        index = self._process.expect(
                            self.READY_PATTERNS + [pexpect.TIMEOUT, pexpect.EOF],
                            timeout=2
                        )
                        if self._process.before:
                            text = self._process.before.strip()
                            if text and text != command:
                                output_buffer.append(text)
                        if index < len(self.READY_PATTERNS):
                            break
                        elif index == len(self.READY_PATTERNS):  # TIMEOUT
                            continue
                        else:  # EOF
                            break

                except Exception as e:
                    if "EOF" in str(e) or "timeout" in str(e).lower():
                        if wait_for_complete:
                            continue
                        break
                    break

            return "\n".join(output_buffer)

        except Exception as e:
            self._log(f"명령 실행 오류: {str(e)}")
            return ""

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """응답에서 JSON을 추출하고 파싱합니다."""
        # JSON 블록 추출
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 순수 JSON 시도
        json_patterns = [
            r"\{[\s\S]*\}",
            r"\[[\s\S]*\]"
        ]
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        # 파싱 실패 시 텍스트 응답 반환
        return {"type": "text", "content": text}

    # ============================================================
    # 핵심 메서드
    # ============================================================

    def analyze_project(self, project_path: str | None = None) -> ProjectAnalysis:
        """
        프로젝트 구조를 분석합니다.

        Args:
            project_path: 분석할 프로젝트 경로 (기본: 작업 디렉토리)

        Returns:
            ProjectAnalysis: 프로젝트 분석 결과
        """
        path = project_path or self.working_directory
        self._log("프로젝트 분석 시작", {"path": path})

        prompt = f"""다음 디렉토리의 프로젝트를 분석해주세요: {path}

다음 정보를 JSON 형식으로 응답해주세요:
```json
{{
  "type": "analysis",
  "content": {{
    "languages": ["사용된 프로그래밍 언어"],
    "frameworks": ["사용된 프레임워크"],
    "structure_summary": "프로젝트 구조 요약",
    "key_files": ["중요한 파일 경로"],
    "conventions": ["감지된 코딩 컨벤션"],
    "dependencies": ["주요 의존성"]
  }}
}}
```
"""

        response = self._send_command(prompt)
        parsed = self._parse_json_response(response)

        content = parsed.get("content", {})

        analysis = ProjectAnalysis(
            project_path=path,
            languages=content.get("languages", []),
            frameworks=content.get("frameworks", []),
            structure_summary=content.get("structure_summary", ""),
            key_files=content.get("key_files", []),
            conventions=content.get("conventions", []),
            dependencies=content.get("dependencies", [])
        )

        self._log("프로젝트 분석 완료", {
            "languages": analysis.languages,
            "frameworks": analysis.frameworks
        })

        return analysis

    def generate_interview_questions(
        self,
        user_goal: str,
        project_analysis: ProjectAnalysis | None = None
    ) -> list[InterviewQuestion]:
        """
        사용자 목표를 분석하여 인터뷰 질문을 생성합니다.

        Args:
            user_goal: 사용자의 요구사항
            project_analysis: 프로젝트 분석 결과

        Returns:
            인터뷰 질문 목록 (3-7개)
        """
        self._log("인터뷰 질문 생성", {"goal": user_goal})

        context = ""
        if project_analysis:
            context = f"""
프로젝트 정보:
- 언어: {', '.join(project_analysis.languages)}
- 프레임워크: {', '.join(project_analysis.frameworks)}
- 구조: {project_analysis.structure_summary}
- 컨벤션: {', '.join(project_analysis.conventions)}
"""

        prompt = f"""{COMMANDER_CLI_CONSTITUTION}

사용자 요구사항: {user_goal}

{context}

위 요구사항을 구현하기 전에 명확히 해야 할 질문을 3-7개 생성해주세요.
각 질문은 다음 영역 중 하나에 초점을 맞춥니다:
- architecture: 아키텍처, 구조
- data_model: 데이터 구조, 스키마
- exception_handling: 에러 처리
- convention: 코딩 컨벤션
- integration: 외부 시스템 연동
- testing: 테스트 전략

JSON 형식으로 응답:
```json
{{
  "type": "interview",
  "content": {{
    "questions": [
      {{
        "question": "질문 내용",
        "focus": "architecture",
        "options": ["옵션1", "옵션2", "직접 입력"],
        "context": "질문 배경"
      }}
    ]
  }}
}}
```
"""

        response = self._send_command(prompt)
        parsed = self._parse_json_response(response)

        content = parsed.get("content", {})
        raw_questions = content.get("questions", [])

        questions = []
        for q in raw_questions:
            try:
                focus = InterviewFocus(q.get("focus", "architecture"))
            except ValueError:
                focus = InterviewFocus.ARCHITECTURE

            questions.append(InterviewQuestion(
                question=q.get("question", ""),
                focus=focus,
                options=q.get("options", []),
                context=q.get("context", "")
            ))

        self._log("인터뷰 질문 생성 완료", {"count": len(questions)})
        return questions

    def generate_feature_spec(
        self,
        user_goal: str,
        interview_answers: list[InterviewAnswer],
        project_analysis: ProjectAnalysis | None = None
    ) -> FeatureSpec:
        """
        인터뷰 답변을 기반으로 기획서를 생성합니다.

        Args:
            user_goal: 사용자 목표
            interview_answers: 인터뷰 답변 목록
            project_analysis: 프로젝트 분석 결과

        Returns:
            FeatureSpec: 기획서
        """
        self._log("기획서 생성 시작", {"goal": user_goal})

        qa_text = "\n".join([
            f"Q: {ia.question.question}\nA: {ia.answer}"
            for ia in interview_answers
        ])

        context = ""
        if project_analysis:
            context = f"프로젝트: {', '.join(project_analysis.languages)} / {', '.join(project_analysis.frameworks)}"

        prompt = f"""인터뷰 결과를 바탕으로 기능 기획서를 작성해주세요.

요구사항: {user_goal}

{context}

인터뷰 내용:
{qa_text}

JSON 형식으로 기획서 작성:
```json
{{
  "type": "feature_spec",
  "content": {{
    "feature_name": "기능명_snake_case",
    "summary": "기능 요약",
    "requirements": ["요구사항 1", "요구사항 2"],
    "architecture_decisions": ["아키텍처 결정사항"],
    "data_model": {{"엔티티": {{"필드": "타입"}}}},
    "error_handling": ["에러 처리 방안"],
    "conventions": ["준수할 컨벤션"],
    "constraints": ["제약 조건"],
    "out_of_scope": ["범위 외 항목"]
  }}
}}
```
"""

        response = self._send_command(prompt)
        parsed = self._parse_json_response(response)

        content = parsed.get("content", {})

        spec = FeatureSpec(
            feature_name=content.get("feature_name", "unnamed_feature"),
            summary=content.get("summary", ""),
            requirements=content.get("requirements", []),
            architecture_decisions=content.get("architecture_decisions", []),
            data_model=content.get("data_model", {}),
            error_handling=content.get("error_handling", []),
            conventions=content.get("conventions", []),
            constraints=content.get("constraints", []),
            out_of_scope=content.get("out_of_scope", [])
        )

        self._log("기획서 생성 완료", {"feature": spec.feature_name})
        return spec

    def decompose_to_tasks(
        self,
        spec: FeatureSpec,
        project_analysis: ProjectAnalysis | None = None
    ) -> list[PlanTask]:
        """
        기획서를 구체적인 태스크로 분해합니다.

        Args:
            spec: 기능 기획서
            project_analysis: 프로젝트 분석 결과

        Returns:
            태스크 목록
        """
        self._log("태스크 분해 시작", {"feature": spec.feature_name})

        spec_summary = f"""
기능: {spec.feature_name}
요약: {spec.summary}
요구사항: {', '.join(spec.requirements)}
아키텍처: {', '.join(spec.architecture_decisions)}
"""

        prompt = f"""다음 기획서를 구체적인 작업 태스크로 분해해주세요.

{spec_summary}

각 태스크는 반드시 다음을 포함해야 합니다:
- 목표 (goal)
- 수정할 파일 (files_to_modify)
- 예상 로직 (expected_logic)
- 완료 조건 (acceptance_criteria)

JSON 형식:
```json
{{
  "type": "task_list",
  "content": {{
    "tasks": [
      {{
        "id": "task_001",
        "title": "작업 제목",
        "goal": "구체적인 목표",
        "files_to_modify": ["파일1.py", "파일2.py"],
        "expected_logic": "구현할 로직 설명",
        "dependencies": [],
        "acceptance_criteria": ["완료 조건 1", "완료 조건 2"]
      }}
    ]
  }}
}}
```
"""

        response = self._send_command(prompt)
        parsed = self._parse_json_response(response)

        content = parsed.get("content", {})
        raw_tasks = content.get("tasks", [])

        tasks = []
        for t in raw_tasks:
            tasks.append(PlanTask(
                id=t.get("id", f"task_{len(tasks)+1:03d}"),
                title=t.get("title", ""),
                goal=t.get("goal", ""),
                files_to_modify=t.get("files_to_modify", []),
                expected_logic=t.get("expected_logic", ""),
                dependencies=t.get("dependencies", []),
                acceptance_criteria=t.get("acceptance_criteria", [])
            ))

        self._log("태스크 분해 완료", {"count": len(tasks)})
        return tasks

    def run_plan_mode(
        self,
        user_goal: str,
        on_question: Callable[[InterviewQuestion], str] | None = None,
        on_spec_review: Callable[[FeatureSpec], bool] | None = None,
        on_tasks_review: Callable[[list[PlanTask]], bool] | None = None
    ) -> PlanModeResult:
        """
        Plan Mode 전체 흐름을 실행합니다.

        1. 프로젝트 분석
        2. 인터뷰 질문 생성 및 수집
        3. 기획서 생성
        4. 태스크 분해
        5. 사용자 승인

        Args:
            user_goal: 사용자 요구사항
            on_question: 질문 콜백 (질문 → 답변)
            on_spec_review: 기획서 리뷰 콜백
            on_tasks_review: 태스크 리뷰 콜백

        Returns:
            PlanModeResult: Plan Mode 결과
        """
        self._log("Plan Mode 시작", {"goal": user_goal})

        result = PlanModeResult()

        # 1. 프로젝트 분석
        try:
            result.project_analysis = self.analyze_project()
        except Exception as e:
            self._log(f"프로젝트 분석 실패: {str(e)}")
            result.project_analysis = ProjectAnalysis(project_path=self.working_directory)

        # 2. 인터뷰 질문 생성
        questions = self.generate_interview_questions(
            user_goal=user_goal,
            project_analysis=result.project_analysis
        )

        # 3. 인터뷰 수행
        for q in questions:
            if on_question:
                answer = on_question(q)
            else:
                answer = q.options[0] if q.options else "확인"

            result.interview_history.append(InterviewAnswer(question=q, answer=answer))

        # 4. 기획서 생성
        result.feature_spec = self.generate_feature_spec(
            user_goal=user_goal,
            interview_answers=result.interview_history,
            project_analysis=result.project_analysis
        )

        # 5. 기획서 리뷰
        if on_spec_review:
            spec_approved = on_spec_review(result.feature_spec)
            if not spec_approved:
                self._log("기획서 미승인")
                return result

        # 6. 태스크 분해
        result.tasks = self.decompose_to_tasks(
            spec=result.feature_spec,
            project_analysis=result.project_analysis
        )

        # 7. 태스크 리뷰 및 승인
        if on_tasks_review:
            result.approved = on_tasks_review(result.tasks)
        else:
            result.approved = False

        self._log("Plan Mode 완료", {
            "approved": result.approved,
            "task_count": len(result.tasks)
        })

        return result

    def close(self):
        """Gemini CLI 세션을 종료합니다."""
        if self._process:
            try:
                self._process.sendline("/exit")
                time.sleep(0.5)
                if sys.platform == "win32":
                    self._process.kill(9)
                else:
                    self._process.close()
            except Exception:
                pass
            finally:
                self._process = None
                self._is_started = False

        self._log("Gemini CLI 세션 종료")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
