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

    args = parser.parse_args()

    # Ctrl+C 핸들링
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))

    if args.command == "index":
        cmd_index(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "overview":
        cmd_overview(args)
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
