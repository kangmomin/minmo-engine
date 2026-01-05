#!/usr/bin/env python
"""
Minmo-Engine Setup Script

이 파일은 하위 호환성을 위해 유지됩니다.
권장 설치 방법: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# README 파일 읽기
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# 버전 정보
version_info = {}
version_file = Path(__file__).parent / "minmo" / "__init__.py"
if version_file.exists():
    with open(version_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                exec(line, version_info)
                break

setup(
    name="minmo-engine",
    version=version_info.get("__version__", "1.0.0"),
    author="Minmo Team",
    author_email="minmo@example.com",
    description="MCP 통합 자동화 프레임워크 - Claude Code와 Gemini를 연동한 멀티 에이전트 오케스트레이션",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/minmo/minmo-engine",
    project_urls={
        "Documentation": "https://github.com/minmo/minmo-engine#readme",
        "Bug Tracker": "https://github.com/minmo/minmo-engine/issues",
        "Source Code": "https://github.com/minmo/minmo-engine",
    },
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "fastmcp>=0.1.0",
        "redis>=5.0.0",
        "google-generativeai>=0.8.0",
        "rich>=13.0.0",
        "pexpect>=4.8.0",
        "watchdog>=3.0.0",
        "pyyaml>=6.0",
        "typing-extensions>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.0.0",
        ],
        "vector": [
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "minmo=minmo.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Typing :: Typed",
    ],
    keywords="mcp automation claude gemini orchestration agent ai llm",
    zip_safe=False,
)
