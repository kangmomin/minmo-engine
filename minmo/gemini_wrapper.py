"""
Gemini Wrapper - Minmo-Engine의 지휘관 (Commander)
Gemini CLI (@google/gemini-cli)를 pexpect로 제어하여 프로젝트 분석, 계획 수립을 담당
"""

import os
import sys
import json
import re
import time
import shutil
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from minmo.scribe_mcp import log_event, get_state, init_database

# 플랫폼별 pexpect 임포트
if sys.platform == "win32":
    from pexpect.popen_spawn import PopenSpawn as PexpectSpawn
else:
    import pexpect
    PexpectSpawn = pexpect.spawn


# ============================================================
# 헌법 (Constitution) - System Prompt
# ============================================================
COMMANDER_CONSTITUTION = """
# Minmo Commander 헌법 (Constitution)

당신은 Minmo-Engine의 **지휘관(Commander)**입니다.
개발 작업을 계획하고, 요구사항을 분석하며, 작업자에게 명확한 지시를 내립니다.

## 핵심 원칙 (절대 위반 금지)

### 1. 최소주의 원칙
- **요청받은 것만 개발하라.** 추가 기능, "있으면 좋을" 기능은 절대 제안하지 마라.
- **추측하지 마라.** 불확실하면 반드시 질문하라.
- **과도한 주석 금지.** 코드가 스스로 설명하게 하라. 복잡한 로직에만 최소한의 주석.

### 2. 명확성 원칙
- **모호한 요구사항은 즉시 명확화하라.** 가정하지 말고 질문하라.
- **구체적인 작업 단위로 분해하라.** "로그인 기능 구현"이 아니라 세부 단계로.
- **기술 선택에는 근거를 제시하라.** 비교 테이블로 장단점을 명시.

### 3. 검증 원칙
- **기존 코드를 먼저 분석하라.** 중복 구현을 피하라.
- **변경 전 영향 범위를 파악하라.** 사이드 이펙트를 예측하라.
- **사용자 확인 없이 큰 변경을 하지 마라.**

### 4. 품질 원칙
- **에러 핸들링은 경계에서만.** 내부 코드는 실패하게 두라.
- **추상화는 3번 반복될 때만.** 미리 일반화하지 마라.
- **타입 안정성을 유지하라.** Any 남용 금지.

## 응답 형식

모든 분석/계획 응답은 다음 JSON 형식을 따릅니다:

```json
{
  "type": "plan|clarification|analysis|recommendation",
  "content": { ... },
  "requires_confirmation": true|false,
  "next_action": "execute|wait_for_input|clarify"
}
```

## 금지 사항

1. 사용자가 요청하지 않은 리팩토링 제안
2. "나중에 확장성을 위해" 같은 과잉 설계
3. 불필요한 디자인 패턴 적용
4. 주관적 코드 스타일 강요
5. 테스트 없이 "작동할 것이다" 단정
"""


# ============================================================
# 데이터 클래스
# ============================================================
class ProjectType(Enum):
    """프로젝트 유형"""
    NEW = "new"
    EXISTING = "existing"
    UNKNOWN = "unknown"


@dataclass
class ProjectAnalysis:
    """프로젝트 분석 결과"""
    project_type: ProjectType
    detected_languages: list[str] = field(default_factory=list)
    detected_frameworks: list[str] = field(default_factory=list)
    detected_databases: list[str] = field(default_factory=list)
    structure_summary: str = ""
    recommendations: list[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ClarificationQuestion:
    """명확화 질문"""
    question: str
    options: list[str] = field(default_factory=list)
    context: str = ""
    required: bool = True


@dataclass
class TaskPlan:
    """작업 계획"""
    goal: str
    tasks: list[dict[str, Any]] = field(default_factory=list)
    estimated_complexity: str = "medium"
    prerequisites: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)


# ============================================================
# Plan Mode 데이터 클래스
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
class InterviewQuestion:
    """Speckit 스타일 심층 인터뷰 질문"""
    question: str
    focus: InterviewFocus
    options: list[str] = field(default_factory=list)
    context: str = ""
    follow_up_hint: str = ""


@dataclass
class InterviewAnswer:
    """인터뷰 답변"""
    question: InterviewQuestion
    answer: str


@dataclass
class FeatureSpec:
    """기획서 (Feature Specification)"""
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
    """상세 작업 (Plan Mode Task)"""
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
    feature_spec: FeatureSpec | None = None
    tasks: list[PlanTask] = field(default_factory=list)
    interview_history: list[InterviewAnswer] = field(default_factory=list)
    approved: bool = False
    spec_file_path: str = ""


# ============================================================
# Gemini Wrapper 클래스 (CLI 기반)
# ============================================================
class GeminiWrapper:
    """
    Gemini CLI 래퍼 - 지휘관 역할

    @google/gemini-cli를 pexpect로 제어하여 프로젝트 분석,
    요구사항 명확화, 작업 계획 수립을 담당합니다.

    gemini login 세션을 사용하므로 API 키가 필요 없습니다.
    """

    # Gemini CLI 프롬프트 패턴
    PROMPT_PATTERNS = [
        r"[>❯►]\s*$",
        r"gemini>\s*$",
        r"\?\s*$",
    ]
    READY_PATTERN = r"[>❯►]\s*$"

    def __init__(
        self,
        working_directory: str | None = None,
        timeout_seconds: int = 120,
        on_log: Callable[[str, str, dict | None], None] | None = None,
        verbose: bool = False
    ):
        """
        GeminiWrapper 초기화 (Gemini CLI 기반)

        Args:
            working_directory: 작업 디렉토리 (기본: 현재 디렉토리)
            timeout_seconds: 명령 타임아웃 (초)
            on_log: 로그 콜백 함수 (agent, content, metadata)
            verbose: 상세 출력 여부
        """
        self.working_directory = working_directory or os.getcwd()
        self.timeout_seconds = timeout_seconds
        self.on_log = on_log
        self.verbose = verbose

        # Gemini CLI 경로 확인
        self.gemini_path = self._find_gemini_cli()

        # 프로세스 핸들
        self._process: PexpectSpawn | None = None
        self._is_started = False

        # 대화 기록
        self._conversation_history: list[dict[str, str]] = []

        # Scribe 연동을 위한 DB 초기화
        try:
            init_database()
        except Exception:
            pass

    def _find_gemini_cli(self) -> str:
        """Gemini CLI 실행 파일을 찾습니다."""
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

        return gemini_cmd

    def _log(self, content: str, metadata: dict | None = None) -> None:
        """Scribe를 통해 로그를 기록합니다."""
        try:
            log_event(
                agent="commander",
                content=content,
                metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None
            )
            if self.on_log:
                self.on_log("commander", content, metadata)
        except Exception as e:
            if self.on_log:
                self.on_log("commander", f"[로그 실패] {content}", {"error": str(e)})

    def _start_session(self) -> bool:
        """Gemini CLI 대화형 세션을 시작합니다."""
        if self._is_started and self._process:
            return True

        self._log("Gemini CLI 세션 시작", {"path": self.gemini_path})

        try:
            # 환경변수 설정 (UTF-8 인코딩)
            env = os.environ.copy()
            env["LANG"] = "C.UTF-8"
            env["LC_ALL"] = "C.UTF-8"
            env["PYTHONIOENCODING"] = "utf-8"

            if sys.platform == "win32":
                self._process = PexpectSpawn(
                    f'"{self.gemini_path}"',
                    encoding="utf-8",
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory,
                    env=env
                )
            else:
                self._process = pexpect.spawn(
                    self.gemini_path,
                    encoding="utf-8",
                    timeout=self.timeout_seconds,
                    cwd=self.working_directory,
                    env=env
                )

            # 초기 프롬프트 대기
            time.sleep(2)
            self._is_started = True
            self._log("Gemini CLI 세션 시작됨")
            return True

        except Exception as e:
            self._log(f"Gemini CLI 시작 실패: {str(e)}", {"error": str(e)})
            return False

    def _send_message(self, message: str) -> str:
        """Gemini CLI에 메시지를 보내고 응답을 받습니다."""
        if not self._process:
            if not self._start_session():
                return ""

        self._conversation_history.append({"role": "user", "content": message})
        output_buffer = []

        try:
            # 명령 전송 (UTF-8 인코딩)
            self._process.sendline(message.encode("utf-8").decode("utf-8"))
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
                            if line and line != message:
                                output_buffer.append(line)
                                if self.verbose:
                                    print(f"[Gemini] {line}")

                        if not self._process.isalive():
                            break

                        # 프롬프트 패턴 확인
                        if output_buffer:
                            last_line = output_buffer[-1]
                            if re.search(self.READY_PATTERN, last_line):
                                output_buffer.pop()
                                break
                    else:
                        index = self._process.expect(
                            self.PROMPT_PATTERNS + [pexpect.TIMEOUT, pexpect.EOF],
                            timeout=2
                        )
                        if self._process.before:
                            text = self._process.before.strip()
                            if text and text != message:
                                output_buffer.append(text)
                        if index < len(self.PROMPT_PATTERNS):
                            break
                        elif index == len(self.PROMPT_PATTERNS):
                            continue
                        else:
                            break

                except Exception as e:
                    if "EOF" in str(e) or "timeout" in str(e).lower():
                        continue
                    break

            response = "\n".join(output_buffer)
            self._conversation_history.append({"role": "model", "content": response})
            return response

        except Exception as e:
            self._log(f"명령 실행 오류: {str(e)}")
            return ""

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """JSON 응답을 파싱합니다."""
        # JSON 블록 추출 시도
        json_match = re.search(r"```json\s*([\s\S]*?)\s*```", response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 일반 코드 블록
        if "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                try:
                    return json.loads(response[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # 순수 JSON 시도
        json_patterns = [r"\{[\s\S]*\}", r"\[[\s\S]*\]"]
        for pattern in json_patterns:
            match = re.search(pattern, response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue

        return {
            "type": "text",
            "content": response,
            "requires_confirmation": False,
            "next_action": "wait_for_input"
        }

    def check_login_status(self) -> bool:
        """Gemini CLI 로그인 상태를 확인합니다."""
        try:
            import subprocess
            env = os.environ.copy()
            env["LANG"] = "C.UTF-8"

            result = subprocess.run(
                [self.gemini_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                env=env,
                encoding="utf-8"
            )
            return result.returncode == 0
        except Exception:
            return False

    # ============================================================
    # 핵심 메서드
    # ============================================================

    def init_project(self, project_path: str, file_list: list[str] | None = None) -> ProjectAnalysis:
        """
        프로젝트를 초기화하고 분석합니다.

        Args:
            project_path: 프로젝트 경로
            file_list: 프로젝트 내 파일 목록 (없으면 신규로 간주)

        Returns:
            ProjectAnalysis: 프로젝트 분석 결과
        """
        self._log("프로젝트 초기화 시작", {"path": project_path})

        is_existing = file_list and len(file_list) > 0

        if is_existing:
            prompt = f"""{COMMANDER_CONSTITUTION}

다음 프로젝트를 분석해주세요.

프로젝트 경로: {project_path}
파일 목록:
{json.dumps(file_list[:100], indent=2, ensure_ascii=False)}
{"(... 외 " + str(len(file_list) - 100) + "개 파일)" if len(file_list) > 100 else ""}

다음 JSON 형식으로 응답해주세요:
```json
{{
  "type": "analysis",
  "content": {{
    "project_type": "existing",
    "detected_languages": ["언어1", "언어2"],
    "detected_frameworks": ["프레임워크1"],
    "detected_databases": ["DB1"],
    "structure_summary": "프로젝트 구조 요약",
    "recommendations": ["권장사항"],
    "confidence": 0.0 ~ 1.0
  }}
}}
```
"""
        else:
            prompt = f"""{COMMANDER_CONSTITUTION}

새 프로젝트를 시작합니다.

프로젝트 경로: {project_path}

다음 JSON 형식으로 기술 스택 제안을 해주세요:
```json
{{
  "type": "recommendation",
  "content": {{
    "project_type": "new",
    "detected_languages": [],
    "detected_frameworks": [],
    "detected_databases": [],
    "recommendations": ["추천사항"],
    "confidence": 0.5
  }}
}}
```
"""

        response = self._send_message(prompt)
        parsed = self._parse_json_response(response)
        content = parsed.get("content", {})

        analysis = ProjectAnalysis(
            project_type=ProjectType.EXISTING if is_existing else ProjectType.NEW,
            detected_languages=content.get("detected_languages", []),
            detected_frameworks=content.get("detected_frameworks", []),
            detected_databases=content.get("detected_databases", []),
            structure_summary=content.get("structure_summary", ""),
            recommendations=content.get("recommendations", []),
            confidence=content.get("confidence", 0.5)
        )

        self._log("프로젝트 분석 완료", {
            "project_type": analysis.project_type.value,
            "languages": analysis.detected_languages,
            "frameworks": analysis.detected_frameworks
        })

        return analysis

    def clarify_goal(
        self,
        user_goal: str,
        context: dict[str, Any] | None = None,
        on_question: Callable[[ClarificationQuestion], str] | None = None,
        max_rounds: int = 5
    ) -> dict[str, Any]:
        """
        모호한 요구사항에 대해 역질문을 던져 명확화합니다.

        Args:
            user_goal: 사용자의 목표/요구사항
            context: 추가 컨텍스트
            on_question: 질문 콜백 함수
            max_rounds: 최대 질문 라운드 수

        Returns:
            명확화된 요구사항
        """
        self._log("요구사항 명확화 시작", {"goal": user_goal})

        clarified = {
            "original_goal": user_goal,
            "clarifications": [],
            "final_requirements": None
        }

        prompt = f"""{COMMANDER_CONSTITUTION}

사용자의 요구사항을 분석하고, 불명확한 부분이 있으면 질문해주세요.

**사용자 요구사항:** {user_goal}

**컨텍스트:**
{json.dumps(context, indent=2, ensure_ascii=False) if context else "없음"}

다음 JSON 형식으로 응답해주세요:

질문이 필요한 경우:
```json
{{
  "type": "clarification",
  "content": {{
    "needs_clarification": true,
    "questions": [
      {{
        "question": "질문 내용",
        "options": ["옵션1", "옵션2", "직접 입력"],
        "context": "이 질문을 하는 이유",
        "required": true
      }}
    ]
  }}
}}
```

요구사항이 명확한 경우:
```json
{{
  "type": "clarification",
  "content": {{
    "needs_clarification": false,
    "understood_requirements": {{
      "summary": "요구사항 요약",
      "scope": ["범위1", "범위2"],
      "constraints": ["제약사항"],
      "assumptions": ["가정사항"]
    }}
  }}
}}
```
"""

        for round_num in range(max_rounds):
            response = self._send_message(prompt)
            parsed = self._parse_json_response(response)
            content = parsed.get("content", {})

            if not content.get("needs_clarification", False):
                clarified["final_requirements"] = content.get("understood_requirements", {})
                break

            questions = content.get("questions", [])
            answers = []

            for q_data in questions:
                question = ClarificationQuestion(
                    question=q_data.get("question", ""),
                    options=q_data.get("options", []),
                    context=q_data.get("context", ""),
                    required=q_data.get("required", True)
                )

                if on_question:
                    answer = on_question(question)
                else:
                    answer = question.options[0] if question.options else "확인"

                answers.append({"question": question.question, "answer": answer})

            clarified["clarifications"].append({
                "round": round_num + 1,
                "questions": questions,
                "answers": answers
            })

            prompt = f"""
사용자의 답변을 바탕으로 추가 질문이 필요한지 판단해주세요.

**원래 요구사항:** {user_goal}
**이전 질문과 답변:**
{json.dumps(answers, indent=2, ensure_ascii=False)}

추가 질문이 필요하면 위와 같은 형식으로, 명확해졌으면 understood_requirements를 반환해주세요.
"""

        self._log("요구사항 명확화 완료", {
            "rounds": len(clarified["clarifications"]),
            "final": clarified["final_requirements"]
        })

        return clarified

    def plan(
        self,
        goal: str,
        context: dict[str, Any] | None = None,
        project_analysis: ProjectAnalysis | None = None
    ) -> TaskPlan:
        """
        목표를 분석하고 작업 계획을 수립합니다.

        Args:
            goal: 사용자의 목표
            context: 추가 컨텍스트
            project_analysis: 프로젝트 분석 결과

        Returns:
            TaskPlan: 작업 계획
        """
        self._log("작업 계획 수립 시작", {"goal": goal})

        project_info = ""
        if project_analysis:
            project_info = f"""
**프로젝트 정보:**
- 유형: {project_analysis.project_type.value}
- 언어: {', '.join(project_analysis.detected_languages) or '미정'}
- 프레임워크: {', '.join(project_analysis.detected_frameworks) or '미정'}
- 구조: {project_analysis.structure_summary or '분석 전'}
"""

        prompt = f"""{COMMANDER_CONSTITUTION}

다음 목표에 대한 작업 계획을 수립해주세요.

**목표:** {goal}

{project_info}

**컨텍스트:**
{json.dumps(context, indent=2, ensure_ascii=False) if context else "없음"}

다음 JSON 형식으로 응답해주세요:
```json
{{
  "type": "plan",
  "content": {{
    "goal": "목표 요약",
    "tasks": [
      {{
        "id": "task_001",
        "title": "작업 제목",
        "description": "구체적인 작업 설명",
        "type": "analysis|implementation|test|documentation",
        "priority": "high|medium|low",
        "dependencies": ["선행 작업 ID"],
        "files_affected": ["예상 수정 파일"]
      }}
    ],
    "estimated_complexity": "low|medium|high|very_high",
    "prerequisites": ["전제 조건"],
    "risks": ["잠재적 위험"]
  }}
}}
```
"""

        response = self._send_message(prompt)
        parsed = self._parse_json_response(response)
        content = parsed.get("content", {})

        plan = TaskPlan(
            goal=content.get("goal", goal),
            tasks=content.get("tasks", []),
            estimated_complexity=content.get("estimated_complexity", "medium"),
            prerequisites=content.get("prerequisites", []),
            risks=content.get("risks", [])
        )

        self._log("작업 계획 수립 완료", {
            "task_count": len(plan.tasks),
            "complexity": plan.estimated_complexity
        })

        return plan

    def analyze_code(self, code: str, file_path: str, question: str | None = None) -> dict[str, Any]:
        """
        코드를 분석합니다.

        Args:
            code: 분석할 코드
            file_path: 파일 경로
            question: 특정 질문

        Returns:
            분석 결과
        """
        self._log("코드 분석 시작", {"file": file_path})

        prompt = f"""
다음 코드를 분석해주세요.

**파일:** {file_path}

```
{code[:10000]}
{"... (코드가 잘렸습니다)" if len(code) > 10000 else ""}
```

{f"**질문:** {question}" if question else ""}

다음 JSON 형식으로 응답해주세요:
```json
{{
  "type": "analysis",
  "content": {{
    "summary": "코드 요약",
    "purpose": "코드의 목적",
    "key_functions": ["주요 함수/클래스"],
    "dependencies": ["의존성"],
    "issues": ["잠재적 문제점"],
    "suggestions": ["개선 제안"]
  }}
}}
```
"""

        response = self._send_message(prompt)
        parsed = self._parse_json_response(response)

        self._log("코드 분석 완료", {"file": file_path})

        return parsed.get("content", {})

    def review_changes(self, diff: str, context: str | None = None) -> dict[str, Any]:
        """
        코드 변경사항을 리뷰합니다.

        Args:
            diff: 변경 diff
            context: 변경 컨텍스트

        Returns:
            리뷰 결과
        """
        self._log("변경사항 리뷰 시작")

        prompt = f"""
다음 코드 변경사항을 리뷰해주세요.

{f"**컨텍스트:** {context}" if context else ""}

**변경 내용:**
```diff
{diff[:15000]}
{"... (diff가 잘렸습니다)" if len(diff) > 15000 else ""}
```

다음 JSON 형식으로 응답해주세요:
```json
{{
  "type": "review",
  "content": {{
    "summary": "변경 요약",
    "approval": "approved|changes_requested|needs_discussion",
    "comments": [
      {{
        "type": "issue|suggestion|praise",
        "location": "파일:라인",
        "message": "코멘트 내용"
      }}
    ],
    "security_concerns": ["보안 우려사항"],
    "breaking_changes": ["호환성 문제"]
  }}
}}
```
"""

        response = self._send_message(prompt)
        parsed = self._parse_json_response(response)

        self._log("변경사항 리뷰 완료", parsed.get("content", {}))

        return parsed.get("content", {})

    def get_conversation_history(self) -> list[dict[str, str]]:
        """대화 기록을 반환합니다."""
        return self._conversation_history.copy()

    def reset_conversation(self) -> None:
        """대화를 초기화합니다."""
        self._conversation_history = []
        if self._process:
            try:
                self._process.sendline("/reset")
            except Exception:
                pass
        self._log("대화 초기화")

    # ============================================================
    # Plan Mode 메서드
    # ============================================================

    def generate_interview_questions(
        self,
        user_goal: str,
        project_context: dict[str, Any] | None = None,
        existing_conventions: list[str] | None = None
    ) -> list[InterviewQuestion]:
        """
        사용자 목표를 분석하여 Speckit 스타일 심층 질문을 생성합니다.

        Args:
            user_goal: 사용자의 요구사항/목표
            project_context: 프로젝트 컨텍스트
            existing_conventions: 기존 프로젝트 컨벤션 목록

        Returns:
            3-7개의 심층 질문 목록
        """
        self._log("인터뷰 질문 생성 시작", {"goal": user_goal})

        conventions_info = ""
        if existing_conventions:
            conventions_info = f"""
**기존 프로젝트 컨벤션:**
{chr(10).join(f"- {c}" for c in existing_conventions)}
"""

        prompt = f"""{COMMANDER_CONSTITUTION}

사용자의 요구사항을 분석하고, 구현 전 반드시 명확화해야 할 심층 질문을 생성해주세요.

**사용자 요구사항:** {user_goal}

**프로젝트 컨텍스트:**
{json.dumps(project_context, indent=2, ensure_ascii=False) if project_context else "없음"}

{conventions_info}

**질문 생성 규칙:**
1. 최소 3개, 최대 7개의 질문을 생성합니다.
2. 각 질문은 다음 영역 중 하나에 초점을 맞춥니다:
   - architecture: 전체 구조, 컴포넌트 관계
   - data_model: 데이터 구조, 스키마
   - exception_handling: 에러 처리, 복구 전략
   - convention: 기존 컨벤션 준수, 네이밍
   - integration: 외부 시스템 연동
   - testing: 테스트 전략
3. 각 질문에는 선택 가능한 옵션을 제공합니다 (2-4개).

다음 JSON 형식으로 응답해주세요:
```json
{{
  "type": "interview_questions",
  "content": {{
    "questions": [
      {{
        "question": "구체적인 질문 내용",
        "focus": "architecture|data_model|exception_handling|convention|integration|testing",
        "options": ["옵션1", "옵션2", "직접 입력"],
        "context": "이 질문을 하는 이유",
        "follow_up_hint": "추가 확인 사항"
      }}
    ]
  }}
}}
```
"""

        response = self._send_message(prompt)
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
                context=q.get("context", ""),
                follow_up_hint=q.get("follow_up_hint", "")
            ))

        self._log("인터뷰 질문 생성 완료", {"count": len(questions)})
        return questions

    def generate_feature_spec(
        self,
        user_goal: str,
        interview_answers: list[InterviewAnswer],
        project_context: dict[str, Any] | None = None
    ) -> FeatureSpec:
        """
        인터뷰 답변을 기반으로 기획서를 생성합니다.

        Args:
            user_goal: 원래 사용자 목표
            interview_answers: 인터뷰 질문과 답변 목록
            project_context: 프로젝트 컨텍스트

        Returns:
            FeatureSpec: 기획서
        """
        self._log("기획서 생성 시작", {"goal": user_goal})

        qa_list = [
            {
                "question": ia.question.question,
                "focus": ia.question.focus.value,
                "answer": ia.answer
            }
            for ia in interview_answers
        ]

        prompt = f"""
인터뷰 결과를 바탕으로 기능 기획서를 작성해주세요.

**원래 요구사항:** {user_goal}

**인터뷰 질문과 답변:**
{json.dumps(qa_list, indent=2, ensure_ascii=False)}

**프로젝트 컨텍스트:**
{json.dumps(project_context, indent=2, ensure_ascii=False) if project_context else "없음"}

다음 JSON 형식으로 기획서를 작성해주세요:
```json
{{
  "type": "feature_spec",
  "content": {{
    "feature_name": "기능명 (영문, snake_case)",
    "summary": "기능 요약 (1-2문장)",
    "requirements": ["구체적인 요구사항"],
    "architecture_decisions": ["아키텍처 결정 사항"],
    "data_model": {{"entities": [], "relationships": []}},
    "error_handling": ["에러 시나리오: 처리 방법"],
    "conventions": ["준수할 컨벤션"],
    "constraints": ["제약 조건"],
    "out_of_scope": ["범위 외 항목"]
  }}
}}
```
"""

        response = self._send_message(prompt)
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
        project_context: dict[str, Any] | None = None
    ) -> list[PlanTask]:
        """
        기획서를 구체적인 Task 리스트로 분해합니다.

        Args:
            spec: 기능 기획서
            project_context: 프로젝트 컨텍스트

        Returns:
            PlanTask 목록
        """
        self._log("태스크 분해 시작", {"feature": spec.feature_name})

        spec_dict = {
            "feature_name": spec.feature_name,
            "summary": spec.summary,
            "requirements": spec.requirements,
            "architecture_decisions": spec.architecture_decisions,
            "data_model": spec.data_model,
            "error_handling": spec.error_handling,
            "conventions": spec.conventions,
            "constraints": spec.constraints
        }

        prompt = f"""
기획서를 구체적인 작업 태스크로 분해해주세요.

**기획서:**
{json.dumps(spec_dict, indent=2, ensure_ascii=False)}

**프로젝트 컨텍스트:**
{json.dumps(project_context, indent=2, ensure_ascii=False) if project_context else "없음"}

다음 JSON 형식으로 태스크를 분해해주세요:
```json
{{
  "type": "task_decomposition",
  "content": {{
    "tasks": [
      {{
        "id": "task_001",
        "title": "작업 제목",
        "goal": "구체적인 목표",
        "files_to_modify": ["수정할 파일"],
        "expected_logic": "구현 로직 설명",
        "dependencies": ["선행 작업 ID"],
        "acceptance_criteria": ["완료 조건"]
      }}
    ]
  }}
}}
```
"""

        response = self._send_message(prompt)
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

    def validate_against_conventions(
        self,
        tasks: list[PlanTask],
        existing_conventions: list[str],
        architecture_info: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        태스크가 기존 컨벤션과 아키텍처를 위반하지 않는지 검증합니다.

        Args:
            tasks: 검증할 태스크 목록
            existing_conventions: 기존 컨벤션 목록
            architecture_info: 기존 아키텍처 정보

        Returns:
            검증 결과
        """
        self._log("컨벤션 검증 시작", {"task_count": len(tasks)})

        tasks_summary = [
            {
                "id": t.id,
                "title": t.title,
                "files": t.files_to_modify,
                "logic": t.expected_logic
            }
            for t in tasks
        ]

        prompt = f"""
제안된 태스크들이 기존 프로젝트 컨벤션과 아키텍처를 준수하는지 검증해주세요.

**태스크 목록:**
{json.dumps(tasks_summary, indent=2, ensure_ascii=False)}

**기존 컨벤션:**
{chr(10).join(f"- {c}" for c in existing_conventions) if existing_conventions else "없음"}

**아키텍처 정보:**
{json.dumps(architecture_info, indent=2, ensure_ascii=False) if architecture_info else "없음"}

다음 JSON 형식으로 검증 결과를 반환해주세요:
```json
{{
  "type": "convention_validation",
  "content": {{
    "approved": true|false,
    "violations": [
      {{
        "task_id": "위반 태스크 ID",
        "violation": "위반 내용",
        "convention": "위반한 컨벤션",
        "severity": "error|warning",
        "suggestion": "수정 제안"
      }}
    ],
    "warnings": ["경고 메시지"],
    "recommendations": ["권장 사항"]
  }}
}}
```
"""

        response = self._send_message(prompt)
        parsed = self._parse_json_response(response)

        content = parsed.get("content", {})

        self._log("컨벤션 검증 완료", {
            "approved": content.get("approved", False),
            "violations": len(content.get("violations", []))
        })

        return content

    def run_plan_mode(
        self,
        user_goal: str,
        project_context: dict[str, Any] | None = None,
        existing_conventions: list[str] | None = None,
        architecture_info: dict[str, Any] | None = None,
        on_question: Callable[[InterviewQuestion], str] | None = None,
        on_spec_review: Callable[[FeatureSpec], bool] | None = None,
        on_tasks_review: Callable[[list[PlanTask]], bool] | None = None
    ) -> PlanModeResult:
        """
        Plan Mode 전체 흐름을 실행합니다.

        1. 심층 인터뷰 질문 생성 및 수집
        2. 기획서 생성
        3. 태스크 분해
        4. 컨벤션 검증
        5. 사용자 승인

        Args:
            user_goal: 사용자 요구사항
            project_context: 프로젝트 컨텍스트
            existing_conventions: 기존 컨벤션 목록
            architecture_info: 아키텍처 정보
            on_question: 질문 콜백
            on_spec_review: 기획서 리뷰 콜백
            on_tasks_review: 태스크 리뷰 콜백

        Returns:
            PlanModeResult: Plan Mode 결과
        """
        self._log("Plan Mode 시작", {"goal": user_goal})

        # 1. 인터뷰 질문 생성
        questions = self.generate_interview_questions(
            user_goal=user_goal,
            project_context=project_context,
            existing_conventions=existing_conventions
        )

        # 2. 인터뷰 수행
        interview_answers: list[InterviewAnswer] = []
        for q in questions:
            if on_question:
                answer = on_question(q)
            else:
                answer = q.options[0] if q.options else "확인"

            interview_answers.append(InterviewAnswer(question=q, answer=answer))

        # 3. 기획서 생성
        spec = self.generate_feature_spec(
            user_goal=user_goal,
            interview_answers=interview_answers,
            project_context=project_context
        )

        # 4. 기획서 리뷰
        spec_approved = True
        if on_spec_review:
            spec_approved = on_spec_review(spec)

        if not spec_approved:
            self._log("기획서 미승인으로 Plan Mode 종료")
            return PlanModeResult(
                feature_spec=spec,
                interview_history=interview_answers,
                approved=False
            )

        # 5. 태스크 분해
        tasks = self.decompose_to_tasks(
            spec=spec,
            project_context=project_context
        )

        # 6. 컨벤션 검증
        if existing_conventions:
            validation = self.validate_against_conventions(
                tasks=tasks,
                existing_conventions=existing_conventions,
                architecture_info=architecture_info
            )

            if not validation.get("approved", True):
                self._log("컨벤션 위반으로 Plan Mode 종료", validation)
                return PlanModeResult(
                    feature_spec=spec,
                    tasks=tasks,
                    interview_history=interview_answers,
                    approved=False
                )

        # 7. 태스크 리뷰
        tasks_approved = True
        if on_tasks_review:
            tasks_approved = on_tasks_review(tasks)

        result = PlanModeResult(
            feature_spec=spec,
            tasks=tasks,
            interview_history=interview_answers,
            approved=tasks_approved
        )

        self._log("Plan Mode 완료", {
            "approved": tasks_approved,
            "task_count": len(tasks)
        })

        return result

    def close(self) -> None:
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
