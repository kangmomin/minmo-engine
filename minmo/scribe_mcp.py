"""
Scribe MCP Server - Minmo-Engine의 핵심 보안 MCP 서버
로컬 Redis와 SQLite를 에이전트들에게 연결하는 가교 역할
"""

import re
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Any
from contextlib import contextmanager

import redis
from fastmcp import FastMCP

# ============================================================
# MCP 서버 초기화
# ============================================================
mcp = FastMCP(name="Scribe")

# ============================================================
# 설정
# ============================================================
REDIS_CONFIG = {
    "host": "localhost",
    "port": 6379,
    "db": 0,
    "decode_responses": True
}

# SQLite DB 경로 - 작업 디렉토리 기준
def get_sqlite_path() -> str:
    """현재 작업 디렉토리에 DB 파일 경로 반환"""
    return str(Path.cwd() / "minmo_history.db")

# Redis 키 프리픽스
REDIS_KEYS = {
    "flow": "minmo:flow",
    "todo": "minmo:todo",
    "events": "minmo:events"
}

# ============================================================
# 보안 필터 - SQL 인젝션 방지
# ============================================================
FORBIDDEN_SQL_PATTERNS = [
    r'\bDELETE\b',
    r'\bDROP\b',
    r'\bUPDATE\b',
    r'\bALTER\b',
    r'\bINSERT\b',
    r'\bCREATE\b',
    r'\bTRUNCATE\b',
    r'\bREPLACE\b',
]

def validate_sql_query(sql: str) -> tuple[bool, str]:
    """
    SQL 쿼리의 안전성을 검증합니다.

    Returns:
        tuple[bool, str]: (유효 여부, 오류 메시지)
    """
    sql_upper = sql.upper()

    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, sql_upper, re.IGNORECASE):
            return False, f"보안 위반: '{pattern.strip(chr(92)).strip('b')}' 구문은 허용되지 않습니다."

    # SELECT만 허용
    if not sql_upper.strip().startswith('SELECT'):
        return False, "보안 정책: SELECT 쿼리만 허용됩니다."

    return True, ""

# ============================================================
# 데이터베이스 연결 관리
# ============================================================
def get_redis_client() -> redis.Redis:
    """Redis 클라이언트 연결을 반환합니다."""
    return redis.Redis(**REDIS_CONFIG)

@contextmanager
def get_sqlite_readonly():
    """읽기 전용 SQLite 연결을 반환합니다."""
    db_path = get_sqlite_path()
    # URI 모드로 read-only 연결
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_sqlite_write():
    """쓰기용 SQLite 연결을 반환합니다."""
    db_path = get_sqlite_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    """SQLite 데이터베이스 초기화 - 테이블 생성"""
    with get_sqlite_write() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS work_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                agent TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_work_history_timestamp
            ON work_history(timestamp)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_work_history_agent
            ON work_history(agent)
        """)

# ============================================================
# MCP Tools
# ============================================================

@mcp.tool()
def search_history(sql: str) -> dict[str, Any]:
    """
    SQLite 로그를 조회합니다. (보안 필터 적용)

    SELECT 쿼리만 허용되며, DELETE/DROP/UPDATE/ALTER 등의
    데이터 변경 구문은 차단됩니다.

    Args:
        sql: 실행할 SELECT SQL 쿼리

    Returns:
        조회 결과 또는 오류 메시지

    Example:
        search_history("SELECT * FROM work_history WHERE agent = 'planner' LIMIT 10")
    """
    # 보안 필터 적용
    is_valid, error_msg = validate_sql_query(sql)
    if not is_valid:
        return {
            "success": False,
            "error": error_msg,
            "blocked": True
        }

    try:
        with get_sqlite_readonly() as conn:
            cursor = conn.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            results = [dict(zip(columns, row)) for row in rows]

            return {
                "success": True,
                "count": len(results),
                "columns": columns,
                "data": results
            }
    except sqlite3.OperationalError as e:
        # 데이터베이스 파일이 없는 경우 등
        if "unable to open database file" in str(e):
            return {
                "success": False,
                "error": "데이터베이스가 아직 초기화되지 않았습니다. log_event를 먼저 호출하세요."
            }
        return {
            "success": False,
            "error": f"SQL 실행 오류: {str(e)}"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"예기치 않은 오류: {str(e)}"
        }


@mcp.tool()
def get_state(category: str) -> dict[str, Any]:
    """
    Redis에서 현재 flow나 todo 상태를 조회합니다.

    Args:
        category: 조회할 카테고리 ("flow" 또는 "todo")

    Returns:
        현재 상태 데이터

    Example:
        get_state("flow")  # 현재 워크플로우 상태 조회
        get_state("todo")  # 현재 할일 목록 조회
    """
    if category not in REDIS_KEYS:
        return {
            "success": False,
            "error": f"유효하지 않은 카테고리입니다. 사용 가능: {list(REDIS_KEYS.keys())}"
        }

    try:
        client = get_redis_client()
        key = REDIS_KEYS[category]

        # Hash 타입으로 저장된 경우
        data = client.hgetall(key)

        if not data:
            # String 타입으로 저장된 경우
            data = client.get(key)
            if data:
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass

        return {
            "success": True,
            "category": category,
            "key": key,
            "data": data if data else {}
        }
    except redis.ConnectionError:
        return {
            "success": False,
            "error": "Redis 서버에 연결할 수 없습니다. Redis가 실행 중인지 확인하세요."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Redis 조회 오류: {str(e)}"
        }


@mcp.tool()
def update_todo(task_id: str, status: str) -> dict[str, Any]:
    """
    특정 작업의 상태를 변경합니다.

    Args:
        task_id: 작업 ID
        status: 새로운 상태 ("pending", "in_progress", "completed", "cancelled")

    Returns:
        업데이트 결과

    Example:
        update_todo("task_001", "completed")
    """
    valid_statuses = ["pending", "in_progress", "completed", "cancelled"]

    if status not in valid_statuses:
        return {
            "success": False,
            "error": f"유효하지 않은 상태입니다. 사용 가능: {valid_statuses}"
        }

    try:
        client = get_redis_client()
        key = REDIS_KEYS["todo"]

        # 현재 todo 데이터 가져오기
        todo_data = client.hget(key, task_id)

        if not todo_data:
            # 새 태스크 생성
            task_data = {
                "id": task_id,
                "status": status,
                "updated_at": datetime.now().isoformat()
            }
        else:
            # 기존 태스크 업데이트
            try:
                task_data = json.loads(todo_data)
            except json.JSONDecodeError:
                task_data = {"id": task_id}

            task_data["status"] = status
            task_data["updated_at"] = datetime.now().isoformat()

        # Redis에 저장
        client.hset(key, task_id, json.dumps(task_data, ensure_ascii=False))

        return {
            "success": True,
            "task_id": task_id,
            "new_status": status,
            "data": task_data
        }
    except redis.ConnectionError:
        return {
            "success": False,
            "error": "Redis 서버에 연결할 수 없습니다."
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Todo 업데이트 오류: {str(e)}"
        }


# ============================================================
# 코드 인덱서 (RAG) 관련
# ============================================================
_code_indexer = None

def get_code_indexer():
    """코드 인덱서 싱글톤을 반환합니다."""
    global _code_indexer
    if _code_indexer is None:
        try:
            from minmo.indexer import CodeIndexer
            _code_indexer = CodeIndexer(str(Path.cwd()), watch=False)
        except Exception:
            pass
    return _code_indexer


@mcp.tool()
def get_code_context(query: str, max_results: int = 5) -> dict[str, Any]:
    """
    코드베이스에서 쿼리와 관련된 코드 컨텍스트를 검색합니다.

    함수명, 클래스명, 독스트링 등을 기반으로 관련 코드를 찾아
    에이전트가 전체 코드를 읽지 않고도 필요한 정보를 파악할 수 있게 합니다.

    Args:
        query: 검색 쿼리 (함수명, 클래스명, 키워드 등)
        max_results: 최대 결과 수 (기본: 5)

    Returns:
        관련 코드 컨텍스트

    Example:
        get_code_context("login")  # 로그인 관련 함수/클래스 검색
        get_code_context("database connection")  # DB 연결 관련 코드 검색
    """
    indexer = get_code_indexer()

    if indexer is None:
        return {
            "success": False,
            "error": "코드 인덱서를 초기화할 수 없습니다. 프로젝트가 인덱싱되지 않았을 수 있습니다."
        }

    try:
        context = indexer.get_code_context(query, max_symbols=max_results)

        return {
            "success": True,
            "query": query,
            "symbols": context.get("symbols", []),
            "related_files": context.get("related_files", []),
            "dependencies": context.get("dependencies", [])
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"코드 컨텍스트 검색 오류: {str(e)}"
        }


@mcp.tool()
def search_symbols(
    name: str,
    symbol_type: str | None = None,
    limit: int = 10
) -> dict[str, Any]:
    """
    코드베이스에서 심볼(함수, 클래스, 메서드)을 검색합니다.

    Args:
        name: 검색할 심볼 이름 또는 패턴
        symbol_type: 심볼 타입 필터 ("function", "class", "method", None=전체)
        limit: 최대 결과 수

    Returns:
        검색된 심볼 목록

    Example:
        search_symbols("User")  # User가 포함된 모든 심볼
        search_symbols("validate", symbol_type="function")  # validate 함수만
    """
    indexer = get_code_indexer()

    if indexer is None:
        return {
            "success": False,
            "error": "코드 인덱서를 초기화할 수 없습니다."
        }

    try:
        results = indexer.search(name, limit=limit)

        symbols = []
        for r in results:
            if symbol_type and r.symbol.symbol_type != symbol_type:
                continue
            symbols.append({
                "name": r.symbol.name,
                "type": r.symbol.symbol_type,
                "file": r.symbol.file_path,
                "line": r.symbol.line_start,
                "signature": r.symbol.signature,
                "docstring": r.symbol.docstring[:200] if r.symbol.docstring else None,
                "parent": r.symbol.parent,
                "score": r.score
            })

        return {
            "success": True,
            "query": name,
            "symbol_type": symbol_type,
            "count": len(symbols),
            "symbols": symbols
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"심볼 검색 오류: {str(e)}"
        }


@mcp.tool()
def get_file_structure(file_path: str) -> dict[str, Any]:
    """
    특정 파일의 구조(클래스, 함수, 임포트 등)를 반환합니다.

    Args:
        file_path: 파일 경로

    Returns:
        파일 구조 정보

    Example:
        get_file_structure("src/auth/login.py")
    """
    indexer = get_code_indexer()

    if indexer is None:
        return {
            "success": False,
            "error": "코드 인덱서를 초기화할 수 없습니다."
        }

    try:
        summary = indexer.get_file_summary(file_path)

        if summary is None:
            # 파일이 인덱싱되지 않은 경우 즉시 인덱싱 시도
            if indexer.index_file(file_path):
                summary = indexer.get_file_summary(file_path)

        if summary is None:
            return {
                "success": False,
                "error": f"파일을 찾을 수 없거나 파싱할 수 없습니다: {file_path}"
            }

        return {
            "success": True,
            "file_path": summary["file_path"],
            "language": summary["language"],
            "classes": summary["classes"],
            "functions": summary["functions"],
            "methods": summary["methods"],
            "imports": summary["imports"],
            "dependencies": summary["dependencies"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"파일 구조 조회 오류: {str(e)}"
        }


@mcp.tool()
def get_project_overview() -> dict[str, Any]:
    """
    프로젝트 전체의 코드 구조 개요를 반환합니다.

    Returns:
        프로젝트 개요 (파일 수, 클래스/함수 목록 등)

    Example:
        get_project_overview()
    """
    indexer = get_code_indexer()

    if indexer is None:
        return {
            "success": False,
            "error": "코드 인덱서를 초기화할 수 없습니다."
        }

    try:
        overview = indexer.get_project_overview()

        return {
            "success": True,
            "project_path": overview["project_path"],
            "stats": overview["stats"],
            "main_classes": overview["main_classes"],
            "main_functions": overview["main_functions"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"프로젝트 개요 조회 오류: {str(e)}"
        }


@mcp.tool()
def index_project(force: bool = False) -> dict[str, Any]:
    """
    프로젝트를 인덱싱하거나 재인덱싱합니다.

    Args:
        force: True면 모든 파일을 강제로 재인덱싱

    Returns:
        인덱싱 결과 통계

    Example:
        index_project()  # 증분 인덱싱 (변경된 파일만)
        index_project(force=True)  # 전체 재인덱싱
    """
    indexer = get_code_indexer()

    if indexer is None:
        return {
            "success": False,
            "error": "코드 인덱서를 초기화할 수 없습니다."
        }

    try:
        stats = indexer.index_all(force=force)

        return {
            "success": True,
            "indexed": stats["indexed"],
            "skipped": stats["skipped"],
            "failed": stats["failed"],
            "total": stats["indexed"] + stats["skipped"] + stats["failed"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"인덱싱 오류: {str(e)}"
        }


@mcp.tool()
def log_event(agent: str, content: str, metadata: str | None = None) -> dict[str, Any]:
    """
    새로운 작업 내역을 SQLite와 Redis에 동시 기록합니다.

    Args:
        agent: 에이전트 이름 (예: "planner", "coder", "reviewer")
        content: 로그 내용
        metadata: 추가 메타데이터 (JSON 문자열, 선택사항)

    Returns:
        기록 결과

    Example:
        log_event("planner", "Task analysis completed", '{"task_id": "task_001"}')
    """
    timestamp = datetime.now().isoformat()

    # 메타데이터 유효성 검사
    if metadata:
        try:
            json.loads(metadata)
        except json.JSONDecodeError:
            return {
                "success": False,
                "error": "metadata는 유효한 JSON 문자열이어야 합니다."
            }

    results = {
        "sqlite": False,
        "redis": False
    }
    errors = []

    # SQLite에 기록
    try:
        init_database()  # 테이블이 없으면 생성

        with get_sqlite_write() as conn:
            cursor = conn.execute(
                """
                INSERT INTO work_history (timestamp, agent, content, metadata)
                VALUES (?, ?, ?, ?)
                """,
                (timestamp, agent, content, metadata)
            )
            record_id = cursor.lastrowid
            results["sqlite"] = True
            results["record_id"] = record_id
    except Exception as e:
        errors.append(f"SQLite 오류: {str(e)}")

    # Redis에 기록 (최근 이벤트 리스트)
    try:
        client = get_redis_client()

        event_data = json.dumps({
            "timestamp": timestamp,
            "agent": agent,
            "content": content,
            "metadata": metadata
        }, ensure_ascii=False)

        # 이벤트 리스트에 추가 (최대 100개 유지)
        client.lpush(REDIS_KEYS["events"], event_data)
        client.ltrim(REDIS_KEYS["events"], 0, 99)

        results["redis"] = True
    except redis.ConnectionError:
        errors.append("Redis 연결 실패")
    except Exception as e:
        errors.append(f"Redis 오류: {str(e)}")

    return {
        "success": results["sqlite"] or results["redis"],
        "timestamp": timestamp,
        "agent": agent,
        "storage_results": results,
        "errors": errors if errors else None
    }


# ============================================================
# 서버 실행
# ============================================================
def run_server():
    """MCP 서버를 실행합니다."""
    print("=" * 50)
    print("Scribe MCP Server 시작")
    print("=" * 50)
    print(f"Redis: {REDIS_CONFIG['host']}:{REDIS_CONFIG['port']}")
    print(f"SQLite: {get_sqlite_path()}")
    print("=" * 50)

    # 데이터베이스 초기화
    try:
        init_database()
        print("SQLite 데이터베이스 초기화 완료")
    except Exception as e:
        print(f"SQLite 초기화 경고: {e}")

    # MCP 서버 실행
    mcp.run()


if __name__ == "__main__":
    run_server()
