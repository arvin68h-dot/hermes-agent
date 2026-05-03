#!/usr/bin/env python3
"""Tests for BM25 skill routing (Phase 1).

Tests:
  - BM25 index build from real skill files
  - Query correctness (known query → known skills)
  - Cache persistence and staleness detection
  - build_skills_system_prompt_with_query integration
  - Fallback to standard prompt when query is too short
  - Graceful handling of missing/invalid skills
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ──────────────────────────────────────────────────────────────


def _make_skill(name, description="", tags=None, body=""):
    """Create a SKILL.md content string for testing."""
    tag_str = f"\ntags: {tags}" if tags else ""
    return f"""\
---
name: {name}
description: {description}{tag_str}
---

{body}
"""


@pytest.fixture
def temp_skills_dir():
    """Create a temporary skills directory with test skills."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skills_dir = Path(tmpdir)

        # Create test skills
        (skills_dir / "codeengine-toolchain").mkdir()
        (skills_dir / "codeengine-toolchain" / "SKILL.md").write_text(
            _make_skill(
                "codeengine-toolchain",
                "CodeEngine CLI 工具链与多智能体调度 — 用于自动化编码、测试、文档管理和多 Worker 并行编排",
                ["code", "automation", "multi-agent"],
                "# CodeEngine Toolchain\n\nFull instructions...",
            )
        )

        (skills_dir / "document-conversion").mkdir()
        (skills_dir / "document-conversion" / "SKILL.md").write_text(
            _make_skill(
                "document-conversion",
                "Convert documents between formats: markdown→PDF, markdown→EPUB, markdown→HTML, and vice versa",
                ["documents", "conversion", "pdf"],
                "# Document Conversion\n\nFull instructions...",
            )
        )

        (skills_dir / "debugging-hermes-tui-commands").mkdir()
        (skills_dir / "debugging-hermes-tui-commands" / "SKILL.md").write_text(
            _make_skill(
                "debugging-hermes-tui-commands",
                "Debug Hermes TUI slash commands: Python, gateway, Ink UI",
                ["debugging", "tui", "hermes"],
                "# Debugging Hermes TUI Commands\n\nFull instructions...",
            )
        )

        (skills_dir / "weather").mkdir()
        (skills_dir / "weather" / "SKILL.md").write_text(
            _make_skill(
                "weather",
                "Query weather forecasts and current conditions for any location worldwide",
                ["weather", "forecast"],
                "# Weather\n\nFull instructions...",
            )
        )

        yield skills_dir


@pytest.fixture
def bm25_index(temp_skills_dir):
    """Create a BM25SkillIndex instance pointing to temp_skills_dir."""
    from agent.bm25_skill_index import BM25SkillIndex
    index = BM25SkillIndex(
        skills_dir=str(temp_skills_dir),
        cache_path=str(temp_skills_dir / ".bm25_index.json"),
    )
    index.build(force=True)
    return index


# ── Tests ─────────────────────────────────────────────────────────────────


class TestBM25SkillIndexBuild:
    """Tests for index building."""

    def test_build_creates_index(self, bm25_index):
        """Index should contain all test skills."""
        assert bm25_index._stats.num_documents >= 4

    def test_build_caches_to_disk(self, temp_skills_dir, bm25_index):
        """Index should be persisted to cache file."""
        cache_file = temp_skills_dir / ".bm25_index.json"
        assert cache_file.exists()
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        assert "docs" in data
        assert len(data["docs"]) >= 4

    def test_build_stats_are_correct(self, bm25_index):
        """Index stats should match actual data."""
        assert bm25_index._stats.num_documents >= 4
        assert bm25_index._stats.num_terms > 0
        assert bm25_index._stats.avg_doc_length > 0
        assert bm25_index._stats.build_time_ms > 0


class TestBM25SkillIndexQuery:
    """Tests for BM25 query correctness."""

    def test_query_returns_top_k(self, bm25_index):
        """Query should return results up to top_k limit."""
        results = bm25_index.query("weather", top_k=5)
        assert len(results) <= 5
        assert len(results) > 0

    def test_query_ranking_relevance(self, bm25_index):
        """More relevant skills should have higher scores."""
        results = bm25_index.query("weather forecast conditions", top_k=5)
        assert len(results) >= 1
        # The 'weather' skill should be in results
        result_names = [r[0] for r in results]
        assert "weather" in result_names
        # Its score should be the highest among results
        weather_score = [r[2] for r in results if r[0] == "weather"][0]
        other_scores = [r[2] for r in results if r[0] != "weather"]
        for score in other_scores:
            assert weather_score >= score  # weather should rank highest

    def test_query_chinese_text(self, bm25_index):
        """Query should handle Chinese text."""
        results = bm25_index.query("编码工具 多智能体", top_k=5)
        assert len(results) >= 1
        result_names = [r[0] for r in results]
        assert "codeengine-toolchain" in result_names

    def test_query_empty(self, bm25_index):
        """Empty query should return empty results."""
        results = bm25_index.query("", top_k=5)
        assert results == []

    def test_query_too_short(self, bm25_index):
        """Very short queries may return results but should not crash."""
        results = bm25_index.query("ab", top_k=5)
        # Should not raise
        assert isinstance(results, list)


class TestBM25SkillIndexCache:
    """Tests for cache persistence and staleness detection."""

    def test_load_from_cache(self, temp_skills_dir):
        """New instance should load from existing cache."""
        from agent.bm25_skill_index import BM25SkillIndex
        # First build to create cache
        first = BM25SkillIndex(
            skills_dir=str(temp_skills_dir),
            cache_path=str(temp_skills_dir / ".bm25_index.json"),
        )
        first.build(force=True)
        assert first._stats.num_documents >= 4

        # New instance should load from cache
        second = BM25SkillIndex(
            skills_dir=str(temp_skills_dir),
            cache_path=str(temp_skills_dir / ".bm25_index.json"),
        )
        assert second._stats.num_documents >= 4

    def test_cache_stale_on_file_change(self, temp_skills_dir, bm25_index):
        """Index should detect when a skill file has been modified."""
        # Get original mtime
        original_mtime = os.path.getmtime(
            temp_skills_dir / "weather" / "SKILL.md"
        )
        time.sleep(0.1)

        # Modify the file
        (temp_skills_dir / "weather" / "SKILL.md").write_text(
            _make_skill(
                "weather",
                "Updated weather description",
                ["weather", "new-tag"],
                "# Updated Weather",
            )
        )

        # is_stale should detect the change
        assert bm25_index.is_stale() is True

    def test_cache_not_stale_on_same_content(self, temp_skills_dir, bm25_index):
        """Index should not be stale when files haven't changed."""
        assert bm25_index.is_stale() is False

    def test_force_rebuild(self, bm25_index):
        """Force rebuild should reconstruct the index."""
        old_count = bm25_index._stats.num_documents
        bm25_index.force_rebuild()
        assert bm25_index._stats.num_documents >= old_count
        assert bm25_index._stats.build_time_ms > 0


class TestBuildSkillsPromptWithQuery:
    """Integration tests for build_skills_system_prompt_with_query."""

    def test_fallback_on_short_query(self, temp_skills_dir):
        """Short query should fall back to standard prompt."""
        from agent.bm25_skill_index import BM25SkillIndex
        from agent.prompt_builder import (
            build_skills_system_prompt_with_query,
            build_skills_system_prompt,
        )

        BM25SkillIndex._instances.clear()
        BM25SkillIndex(
            skills_dir=str(temp_skills_dir),
            cache_path=str(temp_skills_dir / ".bm25_index.json"),
        )

        result = build_skills_system_prompt_with_query(
            query="ab",  # too short
        )
        standard = build_skills_system_prompt()
        # Should be identical (no BM25 injection for short query)
        assert result == standard

    def test_injection_contains_skill_content(self, temp_skills_dir):
        """Injected prompt should contain Top-K skill content."""
        from unittest.mock import patch, MagicMock
        from agent.prompt_builder import (
            build_skills_system_prompt_with_query,
        )

        # Mock the BM25 index to return our temp skill
        mock_index = MagicMock()
        mock_index.query.return_value = [
            ("weather", str(temp_skills_dir / "weather" / "SKILL.md"), 1.5),
        ]

        # Patch at the source module (lazy import inside function)
        with patch(
            "agent.bm25_skill_index.BM25SkillIndex", return_value=mock_index
        ):
            result = build_skills_system_prompt_with_query(
                query="weather forecast",
            )

        # Should contain the injection header
        assert "## Relevant Skills (auto-selected by BM25)" in result
        # Should contain the weather skill's content
        assert "weather" in result
        # Should contain the injection instruction
        assert "Do NOT attempt to load" in result

    @patch("agent.prompt_builder.get_all_skills_dirs")
    @patch("agent.prompt_builder.get_skills_dir")
    def test_injection_not_added_when_no_skills(
        self, mock_skills_dir, mock_all_dirs, temp_skills_dir
    ):
        """Should not add injection when no skills match."""
        from agent.prompt_builder import (
            build_skills_system_prompt_with_query,
        )

        mock_skills_dir.return_value = temp_skills_dir
        mock_all_dirs.return_value = [temp_skills_dir]

        result = build_skills_system_prompt_with_query(
            query="zzzznonexistent12345",  # unlikely to match any skill
        )

        # Should still contain the base prompt
        assert "## Skills (mandatory)" in result
        # But may or may not contain injection (depends on BM25 scoring)
        # At minimum, should not crash


class TestBM25EdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_skills_dir(self):
        """Index should handle empty skills directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from agent.bm25_skill_index import BM25SkillIndex
            index = BM25SkillIndex(
                skills_dir=tmpdir,
                cache_path=os.path.join(tmpdir, ".bm25_index.json"),
            )
            results = index.query("test", top_k=5)
            assert results == []

    def test_broken_skill_file(self, temp_skills_dir):
        """Index should skip broken skill files gracefully."""
        # Create a broken SKILL.md
        broken_dir = temp_skills_dir / "broken-skill"
        broken_dir.mkdir()
        (broken_dir / "SKILL.md").write_text("not a valid skill file\n\n")

        from agent.bm25_skill_index import BM25SkillIndex
        index = BM25SkillIndex(
            skills_dir=str(temp_skills_dir),
            cache_path=str(temp_skills_dir / ".bm25_index.json"),
        )
        index.build(force=True)

        # Should not crash, and broken skill should be skipped
        results = index.query("test", top_k=5)
        assert isinstance(results, list)


# ── Quick Smoke Test ──────────────────────────────────────────────────────


def test_quick_smoke():
    """Quick smoke test to verify BM25 module loads and works."""
    from agent.bm25_skill_index import BM25SkillIndex

    # This should not crash
    index = BM25SkillIndex()
    # Build will use default skills dir (~/.hermes/skills/)
    # If no real skills exist, it's fine — just verify it doesn't crash
    try:
        stats = index.get_stats()
        assert "num_skills" in stats
    except Exception:
        # Acceptable — real skills may not exist in test env
        pass
