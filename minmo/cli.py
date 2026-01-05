"""
Minmo CLI - 세련된 터미널 인터페이스
"""

import sys
import subprocess
import signal
import time
import argparse
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.table import Table
from rich.tree import Tree
from rich import box

from minmo import __version__
from minmo.orchestrator import MinmoOrchestrator
from minmo.indexer import CodeIndexer
from minmo.gemini_wrapper import (
    GeminiWrapper,
    InterviewQuestion,
    FeatureSpec,
    PlanTask,
    PlanModeResult
)
from minmo.scribe_mcp import get_conventions, get_architecture_info, save_feature_spec

console = Console()

# ASCII 아트 로고
LOGO = """
 ███╗   ███╗██╗███╗   ██╗███╗   ███╗ ██████╗
 ████╗ ████║██║████╗  ██║████╗ ████║██╔═══██╗
 ██╔████╔██║██║██╔██╗ ██║██╔████╔██║██║   ██║
 ██║╚██╔╝██║██║██║╚██╗██║██║╚██╔╝██║██║   ██║
 ██║ ╚═╝ ██║██║██║ ╚████║██║ ╚═╝ ██║╚██████╔╝
 ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
"""


class ScribeMCPServer:
    """Scribe MCP 서버 프로세스 관리"""

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.script_path = Path(__file__).parent / "scribe_mcp.py"

    def start(self) -> bool:
        """MCP 서버를 백그라운드에서 시작합니다."""
        if self.process is not None:
            return True

        try:
            self.process = subprocess.Popen(
                [sys.executable, str(self.script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            # 서버 시작 대기
            time.sleep(0.5)

            if self.process.poll() is None:
                return True
            else:
                return False
        except Exception as e:
            console.print(f"[red]MCP 서버 시작 실패: {e}[/red]")
            return False

    def stop(self):
        """MCP 서버를 종료합니다."""
        if self.process is not None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None


def print_header():
    """헤더를 출력합니다."""
    logo_text = Text(LOGO, style="bold cyan")
    console.print(logo_text)
    console.print(
        Panel(
            f"[bold white]MCP 통합 자동화 프레임워크[/bold white]\n"
            f"[dim]Version {__version__}[/dim]",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2)
        )
    )
    console.print()


def print_status_table(goal: str, status: str = "대기 중"):
    """상태 테이블을 출력합니다."""
    table = Table(box=box.ROUNDED, border_style="dim")
    table.add_column("항목", style="cyan", width=12)
    table.add_column("내용", style="white")

    table.add_row("목표", goal)
    table.add_row("상태", f"[yellow]{status}[/yellow]")

    console.print(table)
    console.print()


def run_orchestrator(goal: str):
    """오케스트레이터를 실행합니다."""
    mcp_server = ScribeMCPServer()

    try:
        # MCP 서버 시작
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Scribe MCP 서버 시작 중...", total=None)

            if mcp_server.start():
                progress.update(task, description="[green]Scribe MCP 서버 준비 완료[/green]")
                time.sleep(0.3)
            else:
                console.print("[yellow]경고: MCP 서버를 시작할 수 없습니다. 일부 기능이 제한됩니다.[/yellow]")

        print_status_table(goal, "실행 중")

        # 오케스트레이터 실행
        orchestrator = MinmoOrchestrator()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("작업 계획 수립 중...", total=None)

            result = orchestrator.start_loop(goal)

            progress.update(task, description="[green]작업 완료[/green]")

        # 결과 출력
        console.print()
        console.print(Panel(
            f"[green]목표 달성 완료[/green]\n\n"
            f"[dim]실행된 작업: {len(result.get('results', []))}개[/dim]",
            title="결과",
            box=box.ROUNDED,
            border_style="green"
        ))

    except KeyboardInterrupt:
        console.print("\n[yellow]사용자에 의해 중단되었습니다.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]오류 발생: {e}[/red]")
    finally:
        mcp_server.stop()


def interactive_mode():
    """대화형 모드를 시작합니다."""
    print_header()

    console.print("[cyan]대화형 모드[/cyan] - 목표를 입력하세요 (종료: Ctrl+C 또는 'exit')\n")

    while True:
        try:
            goal = console.input("[bold cyan]minmo>[/bold cyan] ").strip()

            if not goal:
                continue

            if goal.lower() in ('exit', 'quit', 'q'):
                console.print("[dim]종료합니다.[/dim]")
                break

            console.print()
            run_orchestrator(goal)
            console.print()

        except KeyboardInterrupt:
            console.print("\n[dim]종료합니다.[/dim]")
            break
        except EOFError:
            break


# ============================================================
# 인덱싱 명령어
# ============================================================
def cmd_index(args):
    """프로젝트를 인덱싱합니다."""
    project_path = args.path or "."

    console.print(Panel(
        f"[cyan]프로젝트 인덱싱[/cyan]\n경로: {Path(project_path).resolve()}",
        box=box.ROUNDED
    ))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("인덱싱 중...", total=None)

            indexer = CodeIndexer(project_path)
            stats = indexer.index_all(force=args.force)

            progress.update(task, description="[green]인덱싱 완료[/green]")

        # 결과 테이블
        table = Table(box=box.ROUNDED, border_style="green")
        table.add_column("항목", style="cyan")
        table.add_column("수량", style="white", justify="right")

        table.add_row("인덱싱됨", str(stats["indexed"]))
        table.add_row("스킵됨", str(stats["skipped"]))
        table.add_row("실패", str(stats["failed"]))
        table.add_row("총계", str(stats["indexed"] + stats["skipped"] + stats["failed"]))

        console.print(table)

        if args.watch:
            console.print("\n[yellow]파일 변경 감시 중... (Ctrl+C로 종료)[/yellow]")
            indexer.start_watching()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                indexer.stop_watching()
                console.print("\n[dim]감시 종료[/dim]")

    except Exception as e:
        console.print(f"[red]인덱싱 오류: {e}[/red]")


def cmd_search(args):
    """코드베이스에서 심볼을 검색합니다."""
    query = args.query
    project_path = args.path or "."

    try:
        indexer = CodeIndexer(project_path)
        results = indexer.search(query, limit=args.limit)

        if not results:
            console.print(f"[yellow]'{query}'에 대한 검색 결과가 없습니다.[/yellow]")
            console.print("[dim]먼저 'minmo index'를 실행해주세요.[/dim]")
            return

        console.print(Panel(
            f"[cyan]검색 결과[/cyan]: '{query}' ({len(results)}개)",
            box=box.ROUNDED
        ))

        for i, result in enumerate(results, 1):
            symbol = result.symbol

            # 심볼 타입에 따른 색상
            type_color = {
                "class": "magenta",
                "function": "green",
                "method": "blue"
            }.get(symbol.symbol_type, "white")

            # 트리 형태로 표시
            tree = Tree(f"[{type_color}]{symbol.symbol_type}[/{type_color}] [bold]{symbol.name}[/bold]")
            tree.add(f"[dim]파일:[/dim] {symbol.file_path}:{symbol.line_start}")

            if symbol.signature:
                tree.add(f"[dim]시그니처:[/dim] {symbol.signature}")

            if symbol.parent:
                tree.add(f"[dim]부모:[/dim] {symbol.parent}")

            if symbol.docstring:
                doc_preview = symbol.docstring[:100].replace("\n", " ")
                if len(symbol.docstring) > 100:
                    doc_preview += "..."
                tree.add(f"[dim]문서:[/dim] {doc_preview}")

            console.print(tree)
            if i < len(results):
                console.print()

    except Exception as e:
        console.print(f"[red]검색 오류: {e}[/red]")


def cmd_overview(args):
    """프로젝트 개요를 출력합니다."""
    project_path = args.path or "."

    try:
        indexer = CodeIndexer(project_path)
        overview = indexer.get_project_overview()

        console.print(Panel(
            f"[cyan]프로젝트 개요[/cyan]\n{overview['project_path']}",
            box=box.ROUNDED
        ))

        # 통계 테이블
        stats = overview.get("stats", {})
        if stats:
            table = Table(title="통계", box=box.ROUNDED)
            table.add_column("항목", style="cyan")
            table.add_column("수량", justify="right")

            table.add_row("파일", str(stats.get("files", 0)))
            table.add_row("클래스", str(stats.get("classes", 0)))
            table.add_row("함수", str(stats.get("functions", 0)))
            table.add_row("메서드", str(stats.get("methods", 0)))

            console.print(table)
            console.print()

        # 주요 클래스
        if overview.get("main_classes"):
            console.print("[bold]주요 클래스:[/bold]")
            for cls in overview["main_classes"][:10]:
                console.print(f"  [magenta]class[/magenta] {cls['name']} - [dim]{cls['file']}[/dim]")
            console.print()

        # 주요 함수
        if overview.get("main_functions"):
            console.print("[bold]주요 함수:[/bold]")
            for func in overview["main_functions"][:10]:
                console.print(f"  [green]def[/green] {func['name']} - [dim]{func['file']}[/dim]")

    except Exception as e:
        console.print(f"[red]개요 조회 오류: {e}[/red]")


# ============================================================
# Plan Mode 명령어
# ============================================================

def _ask_interview_question(question: InterviewQuestion) -> str:
    """인터뷰 질문을 사용자에게 보여주고 답변을 받습니다."""
    # 질문 패널 표시
    focus_labels = {
        "architecture": "아키텍처",
        "data_model": "데이터 모델",
        "exception_handling": "예외 처리",
        "convention": "컨벤션",
        "integration": "통합",
        "testing": "테스트"
    }

    focus_label = focus_labels.get(question.focus.value, question.focus.value)

    console.print(Panel(
        f"[bold]{question.question}[/bold]\n\n"
        f"[dim]{question.context}[/dim]",
        title=f"[cyan]질문[/cyan] [{focus_label}]",
        box=box.ROUNDED,
        border_style="cyan"
    ))

    # 옵션 표시
    if question.options:
        console.print("\n[bold]선택 가능한 옵션:[/bold]")
        for i, opt in enumerate(question.options, 1):
            console.print(f"  [cyan]{i}.[/cyan] {opt}")
        console.print(f"  [cyan]0.[/cyan] 직접 입력")
        console.print()

        # 선택 받기
        while True:
            try:
                choice = console.input("[bold cyan]선택 (번호 또는 텍스트)>[/bold cyan] ").strip()

                if choice.isdigit():
                    idx = int(choice)
                    if idx == 0:
                        answer = console.input("[bold cyan]답변>[/bold cyan] ").strip()
                        break
                    elif 1 <= idx <= len(question.options):
                        answer = question.options[idx - 1]
                        break
                    else:
                        console.print("[red]유효하지 않은 번호입니다.[/red]")
                else:
                    # 텍스트로 직접 입력
                    answer = choice
                    break
            except KeyboardInterrupt:
                raise
    else:
        answer = console.input("[bold cyan]답변>[/bold cyan] ").strip()

    console.print()
    return answer


def _review_spec(spec: FeatureSpec) -> bool:
    """기획서를 사용자에게 보여주고 승인을 받습니다."""
    console.print(Panel(
        f"[bold]{spec.feature_name}[/bold]\n\n{spec.summary}",
        title="[green]기획서 생성 완료[/green]",
        box=box.ROUNDED,
        border_style="green"
    ))

    # 요구사항
    if spec.requirements:
        console.print("\n[bold]요구사항:[/bold]")
        for req in spec.requirements:
            console.print(f"  • {req}")

    # 아키텍처 결정사항
    if spec.architecture_decisions:
        console.print("\n[bold]아키텍처 결정:[/bold]")
        for dec in spec.architecture_decisions:
            console.print(f"  • {dec}")

    # 에러 핸들링
    if spec.error_handling:
        console.print("\n[bold]에러 처리:[/bold]")
        for err in spec.error_handling:
            console.print(f"  • {err}")

    # 컨벤션
    if spec.conventions:
        console.print("\n[bold]준수할 컨벤션:[/bold]")
        for conv in spec.conventions:
            console.print(f"  • {conv}")

    # 제외 범위
    if spec.out_of_scope:
        console.print("\n[bold]범위 외 (미구현):[/bold]")
        for oos in spec.out_of_scope:
            console.print(f"  • [dim]{oos}[/dim]")

    console.print()

    # 승인 여부 확인
    while True:
        choice = console.input("[bold]기획서를 승인하시겠습니까? [/bold][cyan](y/n/e=수정)[/cyan] ").strip().lower()
        if choice in ('y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False
        elif choice in ('e', 'edit'):
            console.print("[yellow]기획서 수정은 아직 지원되지 않습니다. 다시 시작해주세요.[/yellow]")
            return False


def _review_tasks(tasks: list[PlanTask]) -> bool:
    """태스크 목록을 사용자에게 보여주고 승인을 받습니다."""
    console.print(Panel(
        f"[bold]총 {len(tasks)}개 태스크[/bold]",
        title="[cyan]작업 분해 결과[/cyan]",
        box=box.ROUNDED,
        border_style="cyan"
    ))

    for i, task in enumerate(tasks, 1):
        tree = Tree(f"[bold cyan]{i}. {task.title}[/bold cyan] ({task.id})")
        tree.add(f"[bold]목표:[/bold] {task.goal}")

        if task.files_to_modify:
            files_branch = tree.add("[bold]수정 파일:[/bold]")
            for f in task.files_to_modify:
                files_branch.add(f"[dim]{f}[/dim]")

        if task.expected_logic:
            logic_preview = task.expected_logic[:200]
            if len(task.expected_logic) > 200:
                logic_preview += "..."
            tree.add(f"[bold]예상 로직:[/bold] {logic_preview}")

        if task.dependencies:
            tree.add(f"[bold]선행 작업:[/bold] {', '.join(task.dependencies)}")

        if task.acceptance_criteria:
            criteria_branch = tree.add("[bold]완료 조건:[/bold]")
            for c in task.acceptance_criteria:
                criteria_branch.add(f"[dim]✓ {c}[/dim]")

        console.print(tree)
        console.print()

    # 승인 여부 확인
    console.print(Panel(
        "[bold yellow]중요:[/bold yellow] 승인하면 작업이 시작됩니다.\n"
        "거부하면 기획 단계를 종료하고 수정 후 다시 시작할 수 있습니다.",
        box=box.ROUNDED,
        border_style="yellow"
    ))

    while True:
        choice = console.input("[bold]모든 태스크를 승인하시겠습니까? [/bold][cyan](y/n)[/cyan] ").strip().lower()
        if choice in ('y', 'yes'):
            return True
        elif choice in ('n', 'no'):
            return False


def _generate_spec_markdown(spec: FeatureSpec, result: PlanModeResult) -> str:
    """기획서를 마크다운 문자열로 변환합니다."""
    lines = [
        f"# {spec.feature_name}",
        "",
        f"## 요약",
        spec.summary,
        "",
    ]

    if spec.requirements:
        lines.append("## 요구사항")
        for req in spec.requirements:
            lines.append(f"- {req}")
        lines.append("")

    if spec.architecture_decisions:
        lines.append("## 아키텍처 결정")
        for dec in spec.architecture_decisions:
            lines.append(f"- {dec}")
        lines.append("")

    if spec.data_model:
        lines.append("## 데이터 모델")
        lines.append("```json")
        import json
        lines.append(json.dumps(spec.data_model, indent=2, ensure_ascii=False))
        lines.append("```")
        lines.append("")

    if spec.error_handling:
        lines.append("## 에러 처리")
        for err in spec.error_handling:
            lines.append(f"- {err}")
        lines.append("")

    if spec.conventions:
        lines.append("## 준수 컨벤션")
        for conv in spec.conventions:
            lines.append(f"- {conv}")
        lines.append("")

    if spec.constraints:
        lines.append("## 제약 조건")
        for con in spec.constraints:
            lines.append(f"- {con}")
        lines.append("")

    if spec.out_of_scope:
        lines.append("## 범위 외")
        for oos in spec.out_of_scope:
            lines.append(f"- {oos}")
        lines.append("")

    if result.tasks:
        lines.append("## 태스크 목록")
        for task in result.tasks:
            lines.append(f"### {task.id}: {task.title}")
            lines.append(f"- **목표:** {task.goal}")
            if task.files_to_modify:
                lines.append(f"- **파일:** {', '.join(task.files_to_modify)}")
            if task.expected_logic:
                lines.append(f"- **로직:** {task.expected_logic}")
            if task.acceptance_criteria:
                lines.append("- **완료 조건:**")
                for c in task.acceptance_criteria:
                    lines.append(f"  - {c}")
            lines.append("")

    return "\n".join(lines)


def cmd_plan(args):
    """Plan Mode - 기획/설계 단계를 실행합니다."""
    goal = args.goal

    console.print(Panel(
        f"[bold cyan]Plan Mode[/bold cyan]\n\n"
        f"[dim]요구사항을 분석하고 기획서를 생성합니다.[/dim]\n"
        f"[dim]승인 전까지 실제 작업은 시작되지 않습니다.[/dim]",
        box=box.DOUBLE,
        border_style="cyan"
    ))
    console.print()

    console.print(f"[bold]목표:[/bold] {goal}")
    console.print()

    try:
        # 1. 프로젝트 컨텍스트 수집
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("프로젝트 분석 중...", total=None)

            project_context = {}
            existing_conventions = []
            architecture_info = None

            # 컨벤션 조회
            try:
                conv_result = get_conventions()
                if conv_result.get("success"):
                    existing_conventions = conv_result.get("conventions", [])
                    project_context["naming_patterns"] = conv_result.get("naming_patterns", {})
            except Exception:
                pass

            # 아키텍처 정보 조회
            try:
                arch_result = get_architecture_info()
                if arch_result.get("success"):
                    architecture_info = arch_result
                    project_context["directories"] = arch_result.get("directories", [])
                    project_context["layers"] = arch_result.get("layers", [])
            except Exception:
                pass

            progress.update(task, description="[green]프로젝트 분석 완료[/green]")

        if existing_conventions:
            console.print("[bold]감지된 컨벤션:[/bold]")
            for conv in existing_conventions:
                console.print(f"  • [dim]{conv}[/dim]")
            console.print()

        # 2. GeminiWrapper 초기화
        try:
            commander = GeminiWrapper()
        except ValueError as e:
            console.print(f"[red]오류: {e}[/red]")
            console.print("[dim]GEMINI_API_KEY 환경변수를 설정해주세요.[/dim]")
            return

        # 3. Plan Mode 실행
        console.print(Panel(
            "[bold]인터뷰를 시작합니다[/bold]\n\n"
            "[dim]아키텍처, 데이터 모델, 예외 처리, 컨벤션에 대한 질문에 답해주세요.[/dim]",
            box=box.ROUNDED,
            border_style="yellow"
        ))
        console.print()

        result = commander.run_plan_mode(
            user_goal=goal,
            project_context=project_context,
            existing_conventions=existing_conventions,
            architecture_info=architecture_info,
            on_question=_ask_interview_question,
            on_spec_review=_review_spec,
            on_tasks_review=_review_tasks
        )

        # 4. 결과 처리
        if result.approved:
            console.print(Panel(
                "[bold green]Plan Mode 완료 - 승인됨[/bold green]\n\n"
                f"기획서: {result.feature_spec.feature_name}\n"
                f"태스크: {len(result.tasks)}개",
                box=box.DOUBLE,
                border_style="green"
            ))

            # 기획서 저장
            spec_content = _generate_spec_markdown(result.feature_spec, result)
            save_result = save_feature_spec(
                feature_name=result.feature_spec.feature_name,
                content=spec_content
            )

            if save_result.get("success"):
                console.print(f"\n[green]기획서 저장 완료:[/green] {save_result['file_path']}")
                result.spec_file_path = save_result['file_path']
            else:
                console.print(f"[yellow]기획서 저장 실패: {save_result.get('error')}[/yellow]")

            # 실행 여부 확인
            console.print()
            execute = console.input("[bold]지금 바로 실행하시겠습니까? [/bold][cyan](y/n)[/cyan] ").strip().lower()

            if execute in ('y', 'yes'):
                console.print("\n[cyan]작업 실행을 시작합니다...[/cyan]\n")
                # TODO: 오케스트레이터로 태스크 실행
                console.print("[yellow]태스크 실행 기능은 아직 구현 중입니다.[/yellow]")
            else:
                console.print("\n[dim]'minmo run' 명령으로 나중에 실행할 수 있습니다.[/dim]")

        else:
            console.print(Panel(
                "[bold yellow]Plan Mode 종료 - 미승인[/bold yellow]\n\n"
                "[dim]기획서가 승인되지 않았습니다.[/dim]\n"
                "[dim]요구사항을 수정하고 다시 시도해주세요.[/dim]",
                box=box.ROUNDED,
                border_style="yellow"
            ))

    except KeyboardInterrupt:
        console.print("\n[yellow]사용자에 의해 중단되었습니다.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]오류 발생: {e}[/red]")
        import traceback
        if args.verbose:
            traceback.print_exc()


def main():
    """메인 진입점"""
    parser = argparse.ArgumentParser(
        prog="minmo",
        description="Minmo-Engine: MCP 통합 자동화 프레임워크"
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"minmo {__version__}"
    )

    subparsers = parser.add_subparsers(dest="command", help="사용 가능한 명령어")

    # run 명령어 (기본)
    run_parser = subparsers.add_parser("run", help="목표를 실행합니다")
    run_parser.add_argument("goal", nargs="?", help="실행할 목표")
    run_parser.add_argument("-i", "--interactive", action="store_true", help="대화형 모드")

    # index 명령어
    index_parser = subparsers.add_parser("index", help="프로젝트를 인덱싱합니다")
    index_parser.add_argument("-p", "--path", default=".", help="프로젝트 경로")
    index_parser.add_argument("-f", "--force", action="store_true", help="전체 재인덱싱")
    index_parser.add_argument("-w", "--watch", action="store_true", help="파일 변경 감시")

    # search 명령어
    search_parser = subparsers.add_parser("search", help="코드베이스에서 검색합니다")
    search_parser.add_argument("query", help="검색 쿼리")
    search_parser.add_argument("-p", "--path", default=".", help="프로젝트 경로")
    search_parser.add_argument("-l", "--limit", type=int, default=10, help="최대 결과 수")

    # overview 명령어
    overview_parser = subparsers.add_parser("overview", help="프로젝트 개요를 표시합니다")
    overview_parser.add_argument("-p", "--path", default=".", help="프로젝트 경로")

    # plan 명령어
    plan_parser = subparsers.add_parser("plan", help="Plan Mode - 기획/설계 단계를 실행합니다")
    plan_parser.add_argument("goal", help="구현할 기능 요구사항")
    plan_parser.add_argument("-v", "--verbose", action="store_true", help="상세 출력")

    args = parser.parse_args()

    # Ctrl+C 핸들링
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "overview":
        cmd_overview(args)
    elif args.command == "plan":
        print_header()
        cmd_plan(args)
    elif args.command == "run":
        if args.goal and not args.interactive:
            print_header()
            run_orchestrator(args.goal)
        else:
            interactive_mode()
    else:
        # 명령어 없이 goal만 주어진 경우 또는 대화형 모드
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # 첫 번째 인자가 명령어가 아니면 goal로 처리
            print_header()
            run_orchestrator(sys.argv[1])
        else:
            interactive_mode()


if __name__ == "__main__":
    main()
