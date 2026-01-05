"""
Gemini Wrapper - Minmo-Engine의 지휘관 (Commander)
Gemini API를 사용하여 프로젝트 분석, 계획 수립, 요구사항 명확화를 담당
"""

import os
import json
from typing import Any, Callable
from dataclasses import dataclass, field
from enum import Enum

import google.generativeai as genai

from minmo.scribe_mcp import log_event, get_state, init_database


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
# Gemini Wrapper 클래스
# ============================================================
class GeminiWrapper:
    """
    Gemini API 래퍼 - 지휘관 역할

    프로젝트 분석, 요구사항 명확화, 작업 계획 수립을 담당합니다.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-1.5-flash",
        temperature: float = 0.3,
        on_log: Callable[[str, str, dict | None], None] | None = None
    ):
        """
        GeminiWrapper 초기화

        Args:
            api_key: Gemini API 키 (없으면 환경변수 GEMINI_API_KEY 사용)
            model_name: 사용할 모델 (gemini-1.5-pro, gemini-1.5-flash)
            temperature: 생성 온도 (0.0 ~ 1.0, 낮을수록 결정적)
            on_log: 로그 콜백 함수 (agent, content, metadata)
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API 키가 필요합니다. "
                "환경변수 GEMINI_API_KEY를 설정하거나 api_key 파라미터를 전달하세요."
            )

        self.model_name = model_name
        self.temperature = temperature
        self.on_log = on_log

        # Gemini 설정
        genai.configure(api_key=self.api_key)

        # 모델 초기화
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
            ),
            system_instruction=COMMANDER_CONSTITUTION
        )

        # 대화 기록
        self.chat = self.model.start_chat(history=[])

        # Scribe 연동을 위한 DB 초기화
        try:
            init_database()
        except Exception:
            pass  # DB 초기화 실패해도 계속 진행

    def _log(self, content: str, metadata: dict | None = None) -> None:
        """Scribe를 통해 로그를 기록합니다."""
        try:
            result = log_event(
                agent="commander",
                content=content,
                metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None
            )
            if self.on_log:
                self.on_log("commander", content, metadata)
        except Exception as e:
            # 로깅 실패해도 계속 진행
            if self.on_log:
                self.on_log("commander", f"[로그 실패] {content}", {"error": str(e)})

    def _send_message(self, message: str) -> str:
        """Gemini에 메시지를 보내고 응답을 받습니다."""
        try:
            response = self.chat.send_message(message)
            return response.text
        except Exception as e:
            self._log(f"Gemini API 오류: {str(e)}", {"error_type": type(e).__name__})
            raise

    def _parse_json_response(self, response: str) -> dict[str, Any]:
        """JSON 응답을 파싱합니다."""
        # JSON 블록 추출 시도
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 텍스트 응답 반환
            return {
                "type": "text",
                "content": response,
                "requires_confirmation": False,
                "next_action": "wait_for_input"
            }

    # ============================================================
    # 핵심 메서드
    # ============================================================

    def init_project(self, project_path: str, file_list: list[str] | None = None) -> ProjectAnalysis:
        """
        프로젝트를 초기화하고 분석합니다.

        - 기존 프로젝트: 코드 분석 후 구조 파악, 컨펌 요청
        - 신규 프로젝트: 언어/툴/DB 비교 테이블 제안

        Args:
            project_path: 프로젝트 경로
            file_list: 프로젝트 내 파일 목록 (없으면 신규로 간주)

        Returns:
            ProjectAnalysis: 프로젝트 분석 결과
        """
        self._log("프로젝트 초기화 시작", {"path": project_path})

        # 기존 프로젝트 여부 판단
        is_existing = file_list and len(file_list) > 0

        if is_existing:
            # 기존 프로젝트 분석
            prompt = f"""
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
    "key_files": ["중요 파일 경로"],
    "recommendations": ["권장사항"],
    "confidence": 0.0 ~ 1.0
  }},
  "requires_confirmation": true,
  "next_action": "wait_for_input"
}}
```
"""
        else:
            # 신규 프로젝트 제안
            prompt = f"""
새 프로젝트를 시작합니다.

프로젝트 경로: {project_path}

다음 JSON 형식으로 기술 스택 비교 테이블을 제안해주세요:
```json
{{
  "type": "recommendation",
  "content": {{
    "project_type": "new",
    "language_comparison": [
      {{"name": "Python", "pros": ["장점"], "cons": ["단점"], "use_case": "적합한 경우"}},
      {{"name": "TypeScript", "pros": ["장점"], "cons": ["단점"], "use_case": "적합한 경우"}}
    ],
    "framework_comparison": [
      {{"name": "FastAPI", "pros": ["장점"], "cons": ["단점"], "use_case": "적합한 경우"}}
    ],
    "database_comparison": [
      {{"name": "PostgreSQL", "pros": ["장점"], "cons": ["단점"], "use_case": "적합한 경우"}}
    ],
    "recommended_stack": {{
      "language": "추천 언어",
      "framework": "추천 프레임워크",
      "database": "추천 DB",
      "reasoning": "추천 이유"
    }}
  }},
  "requires_confirmation": true,
  "next_action": "wait_for_input"
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
            context: 추가 컨텍스트 (프로젝트 정보 등)
            on_question: 질문 콜백 함수 (질문을 받고 답변을 반환)
            max_rounds: 최대 질문 라운드 수

        Returns:
            명확화된 요구사항
        """
        self._log("요구사항 명확화 시작", {"goal": user_goal})

        clarified_requirements = {
            "original_goal": user_goal,
            "clarifications": [],
            "final_requirements": None
        }

        prompt = f"""
사용자의 요구사항을 분석하고, 불명확한 부분이 있으면 질문해주세요.

**사용자 요구사항:** {user_goal}

**컨텍스트:**
{json.dumps(context, indent=2, ensure_ascii=False) if context else "없음"}

다음 JSON 형식으로 응답해주세요:

1. 질문이 필요한 경우:
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
  }},
  "requires_confirmation": false,
  "next_action": "clarify"
}}
```

2. 요구사항이 명확한 경우:
```json
{{
  "type": "clarification",
  "content": {{
    "needs_clarification": false,
    "understood_requirements": {{
      "summary": "요구사항 요약",
      "scope": ["범위1", "범위2"],
      "constraints": ["제약사항"],
      "assumptions": ["가정사항 (확인 필요)"]
    }}
  }},
  "requires_confirmation": true,
  "next_action": "wait_for_input"
}}
```
"""

        for round_num in range(max_rounds):
            response = self._send_message(prompt)
            parsed = self._parse_json_response(response)

            content = parsed.get("content", {})
            needs_clarification = content.get("needs_clarification", False)

            if not needs_clarification:
                # 명확화 완료
                clarified_requirements["final_requirements"] = content.get("understood_requirements", {})
                break

            # 질문 처리
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
                    # 콜백이 없으면 첫 번째 옵션 선택 (테스트용)
                    answer = question.options[0] if question.options else "확인"

                answers.append({
                    "question": question.question,
                    "answer": answer
                })

            clarified_requirements["clarifications"].append({
                "round": round_num + 1,
                "questions": questions,
                "answers": answers
            })

            # 다음 라운드를 위한 프롬프트 업데이트
            prompt = f"""
사용자의 답변을 바탕으로 추가 질문이 필요한지 판단해주세요.

**원래 요구사항:** {user_goal}

**이전 질문과 답변:**
{json.dumps(answers, indent=2, ensure_ascii=False)}

추가 질문이 필요하면 위와 같은 형식으로, 명확해졌으면 understood_requirements를 반환해주세요.
"""

        self._log("요구사항 명확화 완료", {
            "rounds": len(clarified_requirements["clarifications"]),
            "final": clarified_requirements["final_requirements"]
        })

        return clarified_requirements

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

        prompt = f"""
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
        "estimated_lines": 50,
        "files_affected": ["예상 수정 파일"]
      }}
    ],
    "estimated_complexity": "low|medium|high|very_high",
    "prerequisites": ["전제 조건"],
    "risks": ["잠재적 위험"],
    "success_criteria": ["완료 기준"]
  }},
  "requires_confirmation": true,
  "next_action": "wait_for_input"
}}
```

**주의사항:**
- 각 작업은 독립적으로 실행 가능해야 합니다.
- 작업 크기는 코드 50-200줄 수준으로 분해하세요.
- 불필요한 작업을 추가하지 마세요.
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
            question: 특정 질문 (없으면 일반 분석)

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
    "suggestions": ["개선 제안 (요청 시에만)"]
  }},
  "requires_confirmation": false,
  "next_action": "wait_for_input"
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
  }},
  "requires_confirmation": false,
  "next_action": "wait_for_input"
}}
```

**리뷰 기준:**
- 요청된 기능만 구현되었는지
- 불필요한 변경이 없는지
- 보안 취약점이 없는지
- 기존 코드와 일관성이 있는지
"""

        response = self._send_message(prompt)
        parsed = self._parse_json_response(response)

        self._log("변경사항 리뷰 완료", parsed.get("content", {}))

        return parsed.get("content", {})

    def get_conversation_history(self) -> list[dict[str, str]]:
        """대화 기록을 반환합니다."""
        history = []
        for msg in self.chat.history:
            history.append({
                "role": msg.role,
                "content": msg.parts[0].text if msg.parts else ""
            })
        return history

    def reset_conversation(self) -> None:
        """대화를 초기화합니다."""
        self.chat = self.model.start_chat(history=[])
        self._log("대화 초기화")
