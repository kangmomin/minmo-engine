"""
Code Indexer - Minmo-Engine의 로컬 코드 인덱서 (RAG)
AST 기반 코드 분석 및 FTS5/Vector 검색 지원
"""

import os
import ast
import json
import sqlite3
import hashlib
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent


# ============================================================
# 데이터 클래스
# ============================================================
@dataclass
class CodeSymbol:
    """코드 심볼 (함수, 클래스, 메서드 등)"""
    name: str
    symbol_type: str  # function, class, method, variable
    file_path: str
    line_start: int
    line_end: int
    docstring: str | None = None
    signature: str | None = None
    parent: str | None = None  # 소속 클래스 등
    decorators: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    return_type: str | None = None


@dataclass
class ImportInfo:
    """임포트 정보"""
    module: str
    names: list[str]  # from X import a, b, c
    alias: str | None = None
    is_relative: bool = False
    level: int = 0  # relative import level


@dataclass
class FileIndex:
    """파일 인덱스"""
    file_path: str
    file_hash: str
    language: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    imports: list[ImportInfo] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)  # 의존하는 모듈들
    indexed_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class SearchResult:
    """검색 결과"""
    symbol: CodeSymbol
    file_path: str
    score: float = 1.0
    context: str | None = None  # 주변 코드


# ============================================================
# AST 파서
# ============================================================
class PythonASTParser:
    """Python AST 파서 - 코드 구조 추출"""

    def __init__(self):
        self.current_file = ""

    def parse_file(self, file_path: str) -> FileIndex | None:
        """파일을 파싱하여 인덱스를 생성합니다."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return self.parse_content(content, file_path)
        except Exception as e:
            return None

    def parse_content(self, content: str, file_path: str) -> FileIndex | None:
        """코드 내용을 파싱합니다."""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None

        self.current_file = file_path
        file_hash = hashlib.md5(content.encode()).hexdigest()

        symbols = []
        imports = []

        for node in ast.walk(tree):
            # 함수 정의
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                symbol = self._extract_function(node)
                if symbol:
                    symbols.append(symbol)

            # 클래스 정의
            elif isinstance(node, ast.ClassDef):
                symbol = self._extract_class(node)
                if symbol:
                    symbols.append(symbol)
                # 클래스 메서드 추출
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method = self._extract_function(item, parent=node.name)
                        if method:
                            symbols.append(method)

            # 임포트
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(ImportInfo(
                        module=alias.name,
                        names=[alias.name],
                        alias=alias.asname,
                        is_relative=False
                    ))

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(ImportInfo(
                        module=node.module,
                        names=[a.name for a in node.names],
                        is_relative=node.level > 0,
                        level=node.level
                    ))

        # 의존성 목록 생성
        dependencies = list(set(imp.module for imp in imports if imp.module))

        return FileIndex(
            file_path=file_path,
            file_hash=file_hash,
            language="python",
            symbols=symbols,
            imports=imports,
            dependencies=dependencies
        )

    def _extract_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, parent: str | None = None) -> CodeSymbol:
        """함수/메서드 정보를 추출합니다."""
        # 독스트링
        docstring = ast.get_docstring(node)

        # 파라미터
        params = []
        for arg in node.args.args:
            param = arg.arg
            if arg.annotation:
                param += f": {ast.unparse(arg.annotation)}"
            params.append(param)

        # 시그니처
        signature = f"def {node.name}({', '.join(params)})"
        if node.returns:
            signature += f" -> {ast.unparse(node.returns)}"

        # 데코레이터
        decorators = [ast.unparse(d) for d in node.decorator_list]

        # 반환 타입
        return_type = ast.unparse(node.returns) if node.returns else None

        return CodeSymbol(
            name=node.name,
            symbol_type="method" if parent else "function",
            file_path=self.current_file,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
            parent=parent,
            decorators=decorators,
            parameters=params,
            return_type=return_type
        )

    def _extract_class(self, node: ast.ClassDef) -> CodeSymbol:
        """클래스 정보를 추출합니다."""
        docstring = ast.get_docstring(node)

        # 베이스 클래스
        bases = [ast.unparse(base) for base in node.bases]
        signature = f"class {node.name}"
        if bases:
            signature += f"({', '.join(bases)})"

        # 데코레이터
        decorators = [ast.unparse(d) for d in node.decorator_list]

        return CodeSymbol(
            name=node.name,
            symbol_type="class",
            file_path=self.current_file,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            docstring=docstring,
            signature=signature,
            decorators=decorators
        )


# ============================================================
# 저장소 인터페이스 (추상 클래스)
# ============================================================
class IndexStorage(ABC):
    """인덱스 저장소 추상 인터페이스"""

    @abstractmethod
    def store(self, file_index: FileIndex) -> bool:
        """파일 인덱스를 저장합니다."""
        pass

    @abstractmethod
    def remove(self, file_path: str) -> bool:
        """파일 인덱스를 삭제합니다."""
        pass

    @abstractmethod
    def get(self, file_path: str) -> FileIndex | None:
        """파일 인덱스를 조회합니다."""
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """심볼을 검색합니다."""
        pass

    @abstractmethod
    def get_file_hash(self, file_path: str) -> str | None:
        """파일 해시를 조회합니다."""
        pass

    @abstractmethod
    def get_dependencies(self, file_path: str) -> list[str]:
        """파일의 의존성을 조회합니다."""
        pass

    @abstractmethod
    def get_dependents(self, module_name: str) -> list[str]:
        """모듈을 사용하는 파일들을 조회합니다."""
        pass

    @abstractmethod
    def get_all_symbols(self, symbol_type: str | None = None) -> list[CodeSymbol]:
        """모든 심볼을 조회합니다."""
        pass

    @abstractmethod
    def close(self) -> None:
        """저장소 연결을 종료합니다."""
        pass


# ============================================================
# SQLite FTS5 저장소
# ============================================================
class SQLiteFTS5Storage(IndexStorage):
    """SQLite FTS5 기반 저장소"""

    def __init__(self, db_path: str = "minmo_index.db"):
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """데이터베이스를 초기화합니다."""
        with self._get_connection() as conn:
            # 파일 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    file_path TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    language TEXT,
                    indexed_at TEXT,
                    imports_json TEXT,
                    dependencies_json TEXT
                )
            """)

            # 심볼 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    name TEXT NOT NULL,
                    symbol_type TEXT NOT NULL,
                    line_start INTEGER,
                    line_end INTEGER,
                    docstring TEXT,
                    signature TEXT,
                    parent TEXT,
                    decorators_json TEXT,
                    parameters_json TEXT,
                    return_type TEXT,
                    FOREIGN KEY (file_path) REFERENCES files(file_path) ON DELETE CASCADE
                )
            """)

            # FTS5 가상 테이블 (전문 검색용)
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                    name,
                    docstring,
                    signature,
                    file_path,
                    content='symbols',
                    content_rowid='id'
                )
            """)

            # 트리거: 심볼 삽입 시 FTS 업데이트
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS symbols_ai AFTER INSERT ON symbols BEGIN
                    INSERT INTO symbols_fts(rowid, name, docstring, signature, file_path)
                    VALUES (new.id, new.name, new.docstring, new.signature, new.file_path);
                END
            """)

            # 트리거: 심볼 삭제 시 FTS 업데이트
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS symbols_ad AFTER DELETE ON symbols BEGIN
                    INSERT INTO symbols_fts(symbols_fts, rowid, name, docstring, signature, file_path)
                    VALUES ('delete', old.id, old.name, old.docstring, old.signature, old.file_path);
                END
            """)

            # 인덱스
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(symbol_type)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """SQLite 연결을 반환합니다."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    def store(self, file_index: FileIndex) -> bool:
        """파일 인덱스를 저장합니다."""
        try:
            with self._get_connection() as conn:
                # 기존 데이터 삭제
                conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_index.file_path,))
                conn.execute("DELETE FROM files WHERE file_path = ?", (file_index.file_path,))

                # 파일 정보 저장
                conn.execute("""
                    INSERT INTO files (file_path, file_hash, language, indexed_at, imports_json, dependencies_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    file_index.file_path,
                    file_index.file_hash,
                    file_index.language,
                    file_index.indexed_at,
                    json.dumps([asdict(i) for i in file_index.imports], ensure_ascii=False),
                    json.dumps(file_index.dependencies, ensure_ascii=False)
                ))

                # 심볼 저장
                for symbol in file_index.symbols:
                    conn.execute("""
                        INSERT INTO symbols (file_path, name, symbol_type, line_start, line_end,
                                            docstring, signature, parent, decorators_json, parameters_json, return_type)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        file_index.file_path,
                        symbol.name,
                        symbol.symbol_type,
                        symbol.line_start,
                        symbol.line_end,
                        symbol.docstring,
                        symbol.signature,
                        symbol.parent,
                        json.dumps(symbol.decorators, ensure_ascii=False),
                        json.dumps(symbol.parameters, ensure_ascii=False),
                        symbol.return_type
                    ))

                conn.commit()
                return True
        except Exception:
            return False

    def remove(self, file_path: str) -> bool:
        """파일 인덱스를 삭제합니다."""
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM symbols WHERE file_path = ?", (file_path,))
                conn.execute("DELETE FROM files WHERE file_path = ?", (file_path,))
                conn.commit()
                return True
        except Exception:
            return False

    def get(self, file_path: str) -> FileIndex | None:
        """파일 인덱스를 조회합니다."""
        try:
            with self._get_connection() as conn:
                file_row = conn.execute(
                    "SELECT * FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()

                if not file_row:
                    return None

                symbol_rows = conn.execute(
                    "SELECT * FROM symbols WHERE file_path = ?", (file_path,)
                ).fetchall()

                symbols = []
                for row in symbol_rows:
                    symbols.append(CodeSymbol(
                        name=row["name"],
                        symbol_type=row["symbol_type"],
                        file_path=row["file_path"],
                        line_start=row["line_start"],
                        line_end=row["line_end"],
                        docstring=row["docstring"],
                        signature=row["signature"],
                        parent=row["parent"],
                        decorators=json.loads(row["decorators_json"] or "[]"),
                        parameters=json.loads(row["parameters_json"] or "[]"),
                        return_type=row["return_type"]
                    ))

                imports_data = json.loads(file_row["imports_json"] or "[]")
                imports = [ImportInfo(**i) for i in imports_data]

                return FileIndex(
                    file_path=file_row["file_path"],
                    file_hash=file_row["file_hash"],
                    language=file_row["language"],
                    symbols=symbols,
                    imports=imports,
                    dependencies=json.loads(file_row["dependencies_json"] or "[]"),
                    indexed_at=file_row["indexed_at"]
                )
        except Exception:
            return None

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """FTS5를 사용하여 심볼을 검색합니다."""
        results = []
        try:
            with self._get_connection() as conn:
                # FTS5 매치 검색
                rows = conn.execute("""
                    SELECT s.*, bm25(symbols_fts) as score
                    FROM symbols s
                    JOIN symbols_fts ON s.id = symbols_fts.rowid
                    WHERE symbols_fts MATCH ?
                    ORDER BY score
                    LIMIT ?
                """, (query, limit)).fetchall()

                for row in rows:
                    symbol = CodeSymbol(
                        name=row["name"],
                        symbol_type=row["symbol_type"],
                        file_path=row["file_path"],
                        line_start=row["line_start"],
                        line_end=row["line_end"],
                        docstring=row["docstring"],
                        signature=row["signature"],
                        parent=row["parent"],
                        decorators=json.loads(row["decorators_json"] or "[]"),
                        parameters=json.loads(row["parameters_json"] or "[]"),
                        return_type=row["return_type"]
                    )
                    results.append(SearchResult(
                        symbol=symbol,
                        file_path=row["file_path"],
                        score=abs(row["score"])  # bm25은 음수 반환
                    ))
        except Exception:
            # FTS 쿼리 실패 시 일반 LIKE 검색
            try:
                with self._get_connection() as conn:
                    rows = conn.execute("""
                        SELECT * FROM symbols
                        WHERE name LIKE ? OR docstring LIKE ? OR signature LIKE ?
                        LIMIT ?
                    """, (f"%{query}%", f"%{query}%", f"%{query}%", limit)).fetchall()

                    for row in rows:
                        symbol = CodeSymbol(
                            name=row["name"],
                            symbol_type=row["symbol_type"],
                            file_path=row["file_path"],
                            line_start=row["line_start"],
                            line_end=row["line_end"],
                            docstring=row["docstring"],
                            signature=row["signature"],
                            parent=row["parent"],
                            decorators=json.loads(row["decorators_json"] or "[]"),
                            parameters=json.loads(row["parameters_json"] or "[]"),
                            return_type=row["return_type"]
                        )
                        results.append(SearchResult(
                            symbol=symbol,
                            file_path=row["file_path"],
                            score=1.0
                        ))
            except Exception:
                pass

        return results

    def get_file_hash(self, file_path: str) -> str | None:
        """파일 해시를 조회합니다."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT file_hash FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()
                return row["file_hash"] if row else None
        except Exception:
            return None

    def get_dependencies(self, file_path: str) -> list[str]:
        """파일의 의존성을 조회합니다."""
        try:
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT dependencies_json FROM files WHERE file_path = ?", (file_path,)
                ).fetchone()
                return json.loads(row["dependencies_json"]) if row else []
        except Exception:
            return []

    def get_dependents(self, module_name: str) -> list[str]:
        """모듈을 사용하는 파일들을 조회합니다."""
        try:
            with self._get_connection() as conn:
                rows = conn.execute("""
                    SELECT file_path, dependencies_json FROM files
                """).fetchall()

                dependents = []
                for row in rows:
                    deps = json.loads(row["dependencies_json"] or "[]")
                    if module_name in deps:
                        dependents.append(row["file_path"])
                return dependents
        except Exception:
            return []

    def get_all_symbols(self, symbol_type: str | None = None) -> list[CodeSymbol]:
        """모든 심볼을 조회합니다."""
        try:
            with self._get_connection() as conn:
                if symbol_type:
                    rows = conn.execute(
                        "SELECT * FROM symbols WHERE symbol_type = ?", (symbol_type,)
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM symbols").fetchall()

                symbols = []
                for row in rows:
                    symbols.append(CodeSymbol(
                        name=row["name"],
                        symbol_type=row["symbol_type"],
                        file_path=row["file_path"],
                        line_start=row["line_start"],
                        line_end=row["line_end"],
                        docstring=row["docstring"],
                        signature=row["signature"],
                        parent=row["parent"],
                        decorators=json.loads(row["decorators_json"] or "[]"),
                        parameters=json.loads(row["parameters_json"] or "[]"),
                        return_type=row["return_type"]
                    ))
                return symbols
        except Exception:
            return []

    def get_stats(self) -> dict[str, Any]:
        """인덱스 통계를 반환합니다."""
        try:
            with self._get_connection() as conn:
                file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
                symbol_count = conn.execute("SELECT COUNT(*) FROM symbols").fetchone()[0]
                class_count = conn.execute(
                    "SELECT COUNT(*) FROM symbols WHERE symbol_type = 'class'"
                ).fetchone()[0]
                function_count = conn.execute(
                    "SELECT COUNT(*) FROM symbols WHERE symbol_type = 'function'"
                ).fetchone()[0]
                method_count = conn.execute(
                    "SELECT COUNT(*) FROM symbols WHERE symbol_type = 'method'"
                ).fetchone()[0]

                return {
                    "files": file_count,
                    "symbols": symbol_count,
                    "classes": class_count,
                    "functions": function_count,
                    "methods": method_count
                }
        except Exception:
            return {}

    def close(self) -> None:
        """저장소 연결을 종료합니다."""
        pass  # SQLite는 컨텍스트 매니저로 관리


# ============================================================
# Vector Storage 인터페이스 (확장용)
# ============================================================
class VectorStorage(IndexStorage):
    """
    벡터 저장소 인터페이스 (ChromaDB, Pinecone 등 확장용)

    이 클래스를 상속하여 벡터 DB를 구현할 수 있습니다.
    """

    def __init__(self, collection_name: str = "minmo_code"):
        self.collection_name = collection_name

    def embed_text(self, text: str) -> list[float]:
        """텍스트를 벡터로 변환합니다. (하위 클래스에서 구현)"""
        raise NotImplementedError("벡터 임베딩은 하위 클래스에서 구현해야 합니다.")

    def similarity_search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """유사도 기반 검색을 수행합니다. (하위 클래스에서 구현)"""
        raise NotImplementedError("유사도 검색은 하위 클래스에서 구현해야 합니다.")


# ============================================================
# 파일 감시자 (Watchdog)
# ============================================================
class CodeFileHandler(FileSystemEventHandler):
    """코드 파일 변경 감지 핸들러"""

    def __init__(
        self,
        indexer: "CodeIndexer",
        extensions: set[str] | None = None
    ):
        self.indexer = indexer
        self.extensions = extensions or {".py", ".js", ".ts", ".tsx", ".jsx"}
        self._debounce_timers: dict[str, threading.Timer] = {}
        self._lock = threading.Lock()

    def _should_index(self, path: str) -> bool:
        """인덱싱 대상 파일인지 확인합니다."""
        return Path(path).suffix in self.extensions

    def _debounced_index(self, file_path: str, action: str):
        """디바운스된 인덱싱을 수행합니다."""
        with self._lock:
            # 기존 타이머 취소
            if file_path in self._debounce_timers:
                self._debounce_timers[file_path].cancel()

            # 새 타이머 설정 (0.5초 후 실행)
            def do_index():
                if action == "delete":
                    self.indexer.remove_file(file_path)
                else:
                    self.indexer.index_file(file_path)

            timer = threading.Timer(0.5, do_index)
            self._debounce_timers[file_path] = timer
            timer.start()

    def on_modified(self, event):
        if not event.is_directory and self._should_index(event.src_path):
            self._debounced_index(event.src_path, "update")

    def on_created(self, event):
        if not event.is_directory and self._should_index(event.src_path):
            self._debounced_index(event.src_path, "create")

    def on_deleted(self, event):
        if not event.is_directory and self._should_index(event.src_path):
            self._debounced_index(event.src_path, "delete")

    def on_moved(self, event):
        if not event.is_directory:
            if self._should_index(event.src_path):
                self._debounced_index(event.src_path, "delete")
            if self._should_index(event.dest_path):
                self._debounced_index(event.dest_path, "create")


# ============================================================
# 메인 인덱서 클래스
# ============================================================
class CodeIndexer:
    """
    코드 인덱서 - 프로젝트 코드를 분석하고 검색 가능하게 만듭니다.

    사용 예:
        indexer = CodeIndexer("./my-project")
        indexer.index_all()
        results = indexer.search("login function")
    """

    # 지원 언어 및 확장자
    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".jsx": "javascript",
    }

    # 무시할 디렉토리
    IGNORE_DIRS = {
        "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
        "env", ".env", "dist", "build", ".tox", ".pytest_cache",
        ".mypy_cache", ".ruff_cache", "egg-info", ".eggs"
    }

    def __init__(
        self,
        project_path: str,
        storage: IndexStorage | None = None,
        watch: bool = False
    ):
        """
        CodeIndexer 초기화

        Args:
            project_path: 프로젝트 루트 경로
            storage: 저장소 (기본: SQLite FTS5)
            watch: 파일 변경 감시 활성화 여부
        """
        self.project_path = Path(project_path).resolve()
        self.storage = storage or SQLiteFTS5Storage(
            str(self.project_path / "minmo_index.db")
        )
        self.parser = PythonASTParser()

        self._observer: Observer | None = None
        self._watching = False

        if watch:
            self.start_watching()

    def _iter_files(self) -> Generator[Path, None, None]:
        """인덱싱할 파일들을 순회합니다."""
        for root, dirs, files in os.walk(self.project_path):
            # 무시할 디렉토리 제외
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    yield file_path

    def _compute_file_hash(self, file_path: str) -> str:
        """파일 해시를 계산합니다."""
        try:
            with open(file_path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def index_file(self, file_path: str | Path) -> bool:
        """단일 파일을 인덱싱합니다."""
        file_path = str(file_path)

        # 파일 존재 확인
        if not os.path.exists(file_path):
            return False

        # 지원 확장자 확인
        suffix = Path(file_path).suffix
        if suffix not in self.SUPPORTED_EXTENSIONS:
            return False

        # 파싱
        if suffix == ".py":
            file_index = self.parser.parse_file(file_path)
        else:
            # TODO: 다른 언어 파서 추가
            return False

        if file_index is None:
            return False

        # 저장
        return self.storage.store(file_index)

    def remove_file(self, file_path: str | Path) -> bool:
        """파일 인덱스를 삭제합니다."""
        return self.storage.remove(str(file_path))

    def index_all(self, force: bool = False) -> dict[str, int]:
        """
        전체 프로젝트를 인덱싱합니다.

        Args:
            force: True면 변경 여부와 관계없이 모든 파일 재인덱싱

        Returns:
            인덱싱 결과 통계
        """
        stats = {"indexed": 0, "skipped": 0, "failed": 0}

        for file_path in self._iter_files():
            file_path_str = str(file_path)

            # 증분 업데이트: 해시 비교
            if not force:
                current_hash = self._compute_file_hash(file_path_str)
                stored_hash = self.storage.get_file_hash(file_path_str)

                if current_hash == stored_hash:
                    stats["skipped"] += 1
                    continue

            # 인덱싱
            if self.index_file(file_path):
                stats["indexed"] += 1
            else:
                stats["failed"] += 1

        return stats

    def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """심볼을 검색합니다."""
        return self.storage.search(query, limit)

    def get_symbol(self, name: str, symbol_type: str | None = None) -> list[CodeSymbol]:
        """이름으로 심볼을 찾습니다."""
        results = self.search(name, limit=50)
        symbols = [r.symbol for r in results if r.symbol.name == name]

        if symbol_type:
            symbols = [s for s in symbols if s.symbol_type == symbol_type]

        return symbols

    def get_class_methods(self, class_name: str) -> list[CodeSymbol]:
        """클래스의 메서드들을 찾습니다."""
        all_symbols = self.storage.get_all_symbols("method")
        return [s for s in all_symbols if s.parent == class_name]

    def get_file_summary(self, file_path: str) -> dict[str, Any] | None:
        """파일의 요약 정보를 반환합니다."""
        file_index = self.storage.get(file_path)
        if not file_index:
            return None

        return {
            "file_path": file_index.file_path,
            "language": file_index.language,
            "classes": [s.name for s in file_index.symbols if s.symbol_type == "class"],
            "functions": [s.name for s in file_index.symbols if s.symbol_type == "function"],
            "methods": [f"{s.parent}.{s.name}" for s in file_index.symbols if s.symbol_type == "method"],
            "imports": [i.module for i in file_index.imports],
            "dependencies": file_index.dependencies
        }

    def get_code_context(self, query: str, max_symbols: int = 5) -> dict[str, Any]:
        """
        쿼리와 관련된 코드 컨텍스트를 반환합니다.
        (에이전트에게 제공할 정보)
        """
        results = self.search(query, limit=max_symbols)

        context = {
            "query": query,
            "symbols": [],
            "related_files": set(),
            "dependencies": set()
        }

        for result in results:
            symbol = result.symbol
            context["symbols"].append({
                "name": symbol.name,
                "type": symbol.symbol_type,
                "signature": symbol.signature,
                "docstring": symbol.docstring[:200] if symbol.docstring else None,
                "file": symbol.file_path,
                "line": symbol.line_start,
                "parent": symbol.parent
            })
            context["related_files"].add(symbol.file_path)

            # 의존성 추가
            deps = self.storage.get_dependencies(symbol.file_path)
            context["dependencies"].update(deps)

        context["related_files"] = list(context["related_files"])
        context["dependencies"] = list(context["dependencies"])

        return context

    def get_project_overview(self) -> dict[str, Any]:
        """프로젝트 전체 개요를 반환합니다."""
        stats = self.storage.get_stats() if hasattr(self.storage, "get_stats") else {}

        all_classes = self.storage.get_all_symbols("class")
        all_functions = self.storage.get_all_symbols("function")

        return {
            "project_path": str(self.project_path),
            "stats": stats,
            "main_classes": [
                {"name": c.name, "file": c.file_path, "docstring": c.docstring[:100] if c.docstring else None}
                for c in all_classes[:20]
            ],
            "main_functions": [
                {"name": f.name, "file": f.file_path, "signature": f.signature}
                for f in all_functions[:20]
            ]
        }

    def start_watching(self) -> None:
        """파일 변경 감시를 시작합니다."""
        if self._watching:
            return

        handler = CodeFileHandler(self, set(self.SUPPORTED_EXTENSIONS.keys()))
        self._observer = Observer()
        self._observer.schedule(handler, str(self.project_path), recursive=True)
        self._observer.start()
        self._watching = True

    def stop_watching(self) -> None:
        """파일 변경 감시를 중지합니다."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        self._watching = False

    def close(self) -> None:
        """인덱서를 종료합니다."""
        self.stop_watching()
        self.storage.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
