<p align="center">
  <img src="https://raw.githubusercontent.com/minmo/minmo-engine/main/assets/logo.png" alt="Minmo Engine" width="400">
</p>

<h1 align="center">Minmo Engine</h1>

<p align="center">
  <strong>MCP 통합 자동화 프레임워크</strong><br>
  Claude Code와 Gemini를 연동한 멀티 에이전트 오케스트레이션 시스템
</p>

<p align="center">
  <a href="#installation">설치</a> •
  <a href="#quick-start">빠른 시작</a> •
  <a href="#features">기능</a> •
  <a href="#architecture">아키텍처</a> •
  <a href="#configuration">설정</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/version-1.0.0-orange.svg" alt="Version 1.0.0">
</p>

---

## Overview

**Minmo Engine**은 AI 에이전트들이 협력하여 소프트웨어 개발 작업을 자동화하는 프레임워크입니다.

```
┌─────────────────────────────────────────────────────────────┐
│                      Minmo Engine                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   👑 Commander (Gemini)     ⚡ Worker (Claude Code)         │
│   ├─ 요구사항 분석           ├─ 코드 작성                    │
│   ├─ 작업 계획 수립          ├─ 파일 수정                    │
│   └─ 에러 분석/수정          └─ 테스트 실행                  │
│                                                             │
│   📚 Indexer (RAG)          📝 Scribe (MCP)                 │
│   ├─ 코드 인덱싱             ├─ 상태 관리 (Redis)            │
│   ├─ FTS5 검색               └─ 이력 저장 (SQLite)           │
│   └─ 의존성 분석                                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Features

### 🎯 멀티 에이전트 오케스트레이션
- **Gemini (지휘관)**: 요구사항 분석, 작업 분해, 계획 수립
- **Claude Code (작업자)**: 실제 코드 작성 및 실행
- **자동 에러 복구**: 실패 시 지휘관에게 피드백 → 계획 수정 → 재시도

### 🔍 로컬 코드 인덱서 (RAG)
- **AST 기반 파싱**: 함수, 클래스, 메서드, 독스트링 추출
- **FTS5 검색**: 빠른 전문 검색 (BM25 랭킹)
- **증분 업데이트**: 변경된 파일만 자동 재인덱싱
- **의존성 추적**: 모듈 간 관계 분석

### 📡 MCP (Model Context Protocol) 통합
- **Scribe MCP Server**: Redis + SQLite 이중 저장소
- **9가지 도구**: 검색, 상태 관리, 로깅, 코드 컨텍스트 등
- **보안 필터**: SQL 인젝션 방지, SELECT 전용 정책

### 💻 세련된 CLI
- **Rich UI**: 컬러풀한 터미널 인터페이스
- **대화형 모드**: REPL 스타일 상호작용
- **서브커맨드**: `index`, `search`, `overview`, `run`

---

## Installation

### 요구 사항

- **Python 3.9+**
- **Redis Server** (상태 관리용)
- **Claude Code CLI** (작업 실행용)
- **Gemini API Key** (계획 수립용)

### 1. 패키지 설치

```bash
# PyPI에서 설치 (권장)
pip install minmo-engine

# 또는 소스에서 설치
git clone https://github.com/minmo/minmo-engine.git
cd minmo-engine
pip install -e .
```

### 2. 의존성 설치

```bash
# Redis 설치 (Windows)
# https://github.com/microsoftarchive/redis/releases 에서 다운로드
# 또는 WSL2에서: sudo apt install redis-server

# Redis 설치 (macOS)
brew install redis
brew services start redis

# Redis 설치 (Linux)
sudo apt install redis-server
sudo systemctl start redis
```

### 3. Claude Code CLI 설치

```bash
# npm을 통한 설치
npm install -g @anthropic-ai/claude-code

# 또는 직접 다운로드
# https://claude.ai/code 참조
```

---

## Configuration

### 환경 변수 설정

```bash
# Linux/macOS
export GEMINI_API_KEY="your-gemini-api-key"
export CLAUDE_CLI_PATH="/path/to/claude"  # 선택사항

# Windows (PowerShell)
$env:GEMINI_API_KEY="your-gemini-api-key"

# Windows (CMD)
set GEMINI_API_KEY=your-gemini-api-key
```

### API 키 발급

| 서비스 | 발급 URL | 용도 |
|--------|----------|------|
| **Gemini API** | [Google AI Studio](https://aistudio.google.com/apikey) | 지휘관 (계획 수립) |
| **Claude Code** | [Anthropic Console](https://console.anthropic.com/) | 작업자 (코드 실행) |

### Redis 설정 (선택사항)

```bash
# 기본값: localhost:6379
# 커스텀 설정은 환경변수로:
export REDIS_HOST="localhost"
export REDIS_PORT="6379"
export REDIS_DB="0"
```

---

## Quick Start

### 1. 프로젝트 인덱싱

```bash
# 현재 디렉토리 인덱싱
minmo index

# 특정 경로 인덱싱
minmo index -p /path/to/project

# 전체 재인덱싱 (강제)
minmo index -f

# 파일 변경 감시 모드
minmo index -w
```

### 2. 코드 검색

```bash
# 심볼 검색
minmo search "login"
minmo search "database" -l 20

# 프로젝트 개요
minmo overview
```

### 3. 작업 실행

```bash
# 직접 실행
minmo run "로그인 기능에 2FA 추가해줘"

# 대화형 모드
minmo
> 사용자 프로필 페이지 만들어줘
> 테스트 코드도 작성해줘
> exit
```

---

## Architecture

### 컴포넌트 구조

```
minmo/
├── __init__.py          # 패키지 초기화, 버전 정보
├── cli.py               # CLI 인터페이스 (Rich UI)
├── orchestrator.py      # 오케스트레이션 엔진
├── gemini_wrapper.py    # Gemini API 래퍼 (지휘관)
├── claude_wrapper.py    # Claude Code 래퍼 (작업자)
├── indexer.py           # 코드 인덱서 (RAG)
└── scribe_mcp.py        # MCP 서버 (Scribe)
```

### 데이터 흐름

```
사용자 입력
     │
     ▼
┌─────────────────┐
│ MinmoOrchestrator│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────────┐
│Gemini │ │Claude Code│
│(계획) │ │(실행)     │
└───┬───┘ └─────┬─────┘
    │           │
    └─────┬─────┘
          ▼
    ┌──────────┐
    │ Scribe   │
    │ MCP      │
    └────┬─────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│ Redis │ │SQLite │
│(상태) │ │(로그) │
└───────┘ └───────┘
```

### MCP 도구 목록

| 도구 | 설명 |
|------|------|
| `search_history` | 작업 이력 검색 (SQL) |
| `get_state` | Redis 상태 조회 |
| `update_todo` | 태스크 상태 업데이트 |
| `log_event` | 이벤트 로깅 |
| `get_code_context` | 코드 컨텍스트 검색 |
| `search_symbols` | 심볼 검색 |
| `get_file_structure` | 파일 구조 분석 |
| `get_project_overview` | 프로젝트 개요 |
| `index_project` | 프로젝트 인덱싱 |

---

## Usage Examples

### Python API 사용

```python
from minmo import MinmoOrchestrator, CodeIndexer

# 오케스트레이터 사용
orchestrator = MinmoOrchestrator()
result = orchestrator.start_loop("사용자 인증 시스템 구현")
print(result["status"])

# 인덱서 사용
indexer = CodeIndexer("./my-project")
indexer.index_all()

# 검색
results = indexer.search("authenticate")
for r in results:
    print(f"{r.symbol.name} in {r.symbol.file_path}")

# 코드 컨텍스트
context = indexer.get_code_context("database connection")
print(context["symbols"])
```

### Gemini 직접 사용

```python
from minmo import GeminiWrapper

gemini = GeminiWrapper()

# 프로젝트 분석
analysis = gemini.init_project("./my-project", file_list)

# 요구사항 명확화
clarified = gemini.clarify_goal("로그인 기능 추가")

# 작업 계획
plan = gemini.plan("JWT 기반 인증 구현")
```

### Claude Code 직접 사용

```python
from minmo import ClaudeCodeWrapper

worker = ClaudeCodeWrapper(
    working_directory="./my-project",
    verbose=True
)

result = worker.execute({
    "id": "task_001",
    "title": "로그인 폼 구현",
    "description": "이메일/비밀번호 입력 폼 생성",
    "type": "implementation"
})
```

---

## Configuration Files

### `.claude_config.json`

프로젝트 루트에 MCP 설정 파일:

```json
{
  "mcpServers": {
    "minmo-scribe": {
      "command": "python",
      "args": ["-m", "minmo.scribe_mcp"],
      "env": {}
    }
  }
}
```

### `.claude/settings.local.json`

Claude Code 권한 설정:

```json
{
  "permissions": {
    "allow": [
      "mcp__minmo-scribe__*"
    ]
  }
}
```

---

## Troubleshooting

### Redis 연결 실패

```bash
# Redis 서버 상태 확인
redis-cli ping

# 서비스 시작
# Windows: Redis 설치 폴더에서 redis-server.exe 실행
# macOS: brew services start redis
# Linux: sudo systemctl start redis
```

### Gemini API 오류

```bash
# API 키 확인
echo $GEMINI_API_KEY

# 키가 설정되지 않은 경우
export GEMINI_API_KEY="your-api-key"
```

### Claude Code를 찾을 수 없음

```bash
# Claude CLI 경로 확인
which claude  # Linux/macOS
where claude  # Windows

# 직접 경로 설정
export CLAUDE_CLI_PATH="/path/to/claude"
```

### 인덱싱 실패

```bash
# 권한 확인
ls -la minmo_index.db

# 강제 재인덱싱
minmo index -f
```

---

## Development

### 개발 환경 설정

```bash
# 저장소 클론
git clone https://github.com/minmo/minmo-engine.git
cd minmo-engine

# 개발 의존성 설치
pip install -e ".[dev]"

# 테스트 실행
pytest

# 린트 검사
ruff check minmo/
mypy minmo/
```

### 벡터 검색 확장 (선택사항)

```bash
# ChromaDB 지원 설치
pip install -e ".[vector]"
```

---

## Roadmap

- [x] v1.0 - 기본 오케스트레이션
- [ ] v1.1 - 벡터 검색 (ChromaDB)
- [ ] v1.2 - 웹 대시보드
- [ ] v1.3 - 멀티 프로젝트 지원
- [ ] v2.0 - 플러그인 시스템

---

## Contributing

기여를 환영합니다! [CONTRIBUTING.md](CONTRIBUTING.md)를 참조해주세요.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

<p align="center">
  Made with ❤️ by Minmo Team
</p>
