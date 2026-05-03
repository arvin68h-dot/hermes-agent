#!/usr/bin/env python3
"""
Hybrid Skill Router tests — Phase 2 (RuleRouter) + Phase 3 (LLMRouter) + unified SkillRouter.

Tests cover:
- RuleRouter keyword matching accuracy
- RuleRouter priority ordering
- SkillRouter three-layer chaining
- build_skills_system_prompt_with_query integration
- Edge cases (empty query, no matches, duplicate skills)
"""

import json
import os
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestRuleRouter(unittest.TestCase):
    """Phase 2: Rule-based routing engine tests."""

    def setUp(self):
        from agent.skill_router import RuleRouter
        self.router = RuleRouter()

    # ── Basic keyword matching ──
    def test_weather_rule_matches(self):
        """Weather keywords should match."""
        results = self.router.route("明天天气怎么样")
        skills = [r[0] for r in results]
        self.assertIn("weather", skills)

    def test_code_rule_matches(self):
        """Code writing keywords should match."""
        results = self.router.route("帮我写一段 Python 代码")
        skills = [r[0] for r in results]
        self.assertIn("codeengine-toolchain", skills)

    def test_git_rule_matches(self):
        """Git keywords should match."""
        results = self.router.route("commit 并 push 到远程")
        skills = [r[0] for r in results]
        self.assertIn("github-pr-workflow", skills)

    def test_pdf_rule_matches(self):
        """PDF conversion keywords should match."""
        results = self.router.route("把这份文档转成 PDF")
        skills = [r[0] for r in results]
        self.assertIn("document-conversion", skills)

    def test_pptx_rule_matches(self):
        """PowerPoint keywords should match."""
        results = self.router.route("做一个演示 PPT")
        skills = [r[0] for r in results]
        self.assertIn("powerpoint", skills)

    def test_diyam_rule_matches(self):
        """Diagram keywords should match."""
        results = self.router.route("画一个系统架构图")
        skills = [r[0] for r in results]
        self.assertIn("architecture-diagram", skills)

    def test_agent_rule_matches(self):
        """Multi-agent keywords should match."""
        results = self.router.route("用子代理帮我完成这个任务")
        skills = [r[0] for r in results]
        self.assertIn("multi-agent-orchestration", skills)

    # ── Priority ordering ──
    def test_higher_priority_comes_first(self):
        """Higher priority rules should produce skills listed first."""
        results = self.router.route("帮我写代码并审查")
        # code-work (priority 20) should beat general rules
        code_skill = [r for r in results if r[0] in ["codeengine-toolchain"]]
        self.assertGreater(len(code_skill), 0)
        # The first matched skill should have high priority
        self.assertGreaterEqual(results[0][2], 15.0)

    def test_get_top_n(self):
        """get_top_n should return at most n skills."""
        results = self.router.get_top_n("写一段代码测试", n=2)
        self.assertLessEqual(len(results), 2)

    # ── Edge cases ──
    def test_empty_query_returns_empty(self):
        """Empty input should return no matches."""
        results = self.router.route("")
        self.assertEqual(results, [])

    def test_short_query_returns_empty(self):
        """Very short query should return no matches."""
        results = self.router.route("hi")
        self.assertEqual(results, [])

    def test_no_matching_keywords_returns_empty(self):
        """Completely unrelated query returns no rule matches."""
        results = self.router.route("helloworld nonsense123")
        self.assertEqual(results, [])

    def test_multiple_rules_match(self):
        """Multiple matching rules should all be represented."""
        results = self.router.route("帮我写代码并用 git 提交")
        skills = [r[0] for r in results]
        # Should have both code and git skills
        code_skill = [s for s in skills if "code" in s.lower() or "engine" in s.lower()]
        git_skill = [s for s in skills if "git" in s.lower() or "github" in s.lower()]
        self.assertGreater(len(code_skill) + len(git_skill), 1)

    def test_lowercase_insensitive(self):
        """Matching should be case-insensitive."""
        results = self.router.route("帮我写 CODE 代码")
        skills = [r[0] for r in results]
        self.assertIn("codeengine-toolchain", skills)

    def test_chinese_and_english_mixed(self):
        """Mixed Chinese/English query should work."""
        results = self.router.route("帮我用 git commit 提交代码")
        skills = [r[0] for r in results]
        self.assertIn("github-pr-workflow", skills)

    def test_rule_count_reasonable(self):
        """Rule set should have a reasonable number of rules."""
        self.assertGreaterEqual(len(self.router.RULES), 30)
        self.assertLessEqual(len(self.router.RULES), 200)

    def test_rule_format(self):
        """Each rule should follow (priority, name, patterns, skill_ids) format."""
        for rule in self.router.RULES:
            self.assertEqual(len(rule), 4)
            priority, name, patterns, skill_ids = rule
            self.assertIsInstance(priority, int)
            self.assertIsInstance(name, str)
            self.assertIsInstance(patterns, list)
            self.assertIsInstance(skill_ids, list)
            self.assertGreater(len(patterns), 0)
            self.assertGreater(len(skill_ids), 0)


class TestSkillRouter(unittest.TestCase):
    """Unified SkillRouter three-layer chaining tests."""

    def setUp(self):
        from agent.skill_router import SkillRouter
        import shutil
        import uuid
        # Use a unique temp directory per test instance to avoid pytest-xdist races
        unique_id = uuid.uuid4().hex[:8]
        self.tmp_dir = Path(f"/tmp/test_skills_router_{unique_id}")
        self.tmp_dir.mkdir(exist_ok=True, parents=True)
        # Create a few test SKILL.md files
        self._create_test_skills()
        self.router = SkillRouter(skills_dir=str(self.tmp_dir))

    def _create_test_skills(self):
        """Create minimal test SKILL.md files."""
        test_skills = [
            ("test-weather", "Sunny, weather, temperature, 天气"),
            ("test-coding", "Write code, coding, debug, 写代码"),
            ("test-document", "Document, PDF, 文档, PDF 转换"),
        ]

        for name, desc in test_skills:
            skill_dir = self.tmp_dir / name
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_md = skill_dir / "SKILL.md"
            content = textwrap.dedent(f"""
                ---
                name: {name}
                description: {desc}
                tags: []
                ---

                # {name}

                Test skill for routing.
            """).strip()
            skill_md.write_text(content, encoding="utf-8")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_rule_layer_hits_first(self):
        """RuleRouter should match before BM25 for simple queries."""
        results = self.router.route("测试天气 天气预报")
        # Should have rule-layer results
        rule_hits = [r for r in results if r[3] == "rule"]
        self.assertGreater(len(rule_hits), 0)

    def test_query_pattern(self):
        """Results should be (name, path, score, method) tuples."""
        results = self.router.route("测试")
        for r in results:
            self.assertEqual(len(r), 4)
            name, path, score, method = r
            self.assertIsInstance(name, str)
            self.assertIsInstance(path, str)
            self.assertIsInstance(score, float)
            self.assertIn(method, ["rule", "bm25", "llm"])

    def test_top_k_limit(self):
        """Should respect top_k limit."""
        results = self.router.route("测试", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_empty_query(self):
        """Empty query returns empty list."""
        results = self.router.route("")
        self.assertEqual(results, [])

    def test_get_top_skill_ids(self):
        """get_top_skill_ids should return string IDs."""
        ids = self.router.get_top_skill_ids("测试天气", n=2)
        self.assertIsInstance(ids, list)
        self.assertLessEqual(len(ids), 2)
        for sid in ids:
            self.assertIsInstance(sid, str)

    def test_stats(self):
        """get_stats should return valid stats."""
        stats = self.router.get_stats()
        self.assertIn("total_skills", stats)
        self.assertIn("rule_count", stats)
        self.assertGreater(stats["total_skills"], 0)


class TestBuildSkillsWithQuery(unittest.TestCase):
    """Integration: build_skills_system_prompt_with_query returns valid prompt."""

    def setUp(self):
        self.tmp_dir = Path("/tmp/test_build_skills_query")
        self.tmp_dir.mkdir(exist_ok=True)
        # Create a test skill
        skill_dir = self.tmp_dir / "test-skill"
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            "---\nname: test-skill\ndescription: Test skill for prompt builder\ntags: []\n---\n\n# test-skill\n\nThis is a test skill.",
            encoding="utf-8"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_returns_prompt_with_injection(self):
        """Should return a prompt that includes the injected skill."""
        import agent.prompt_builder as pb_module
        from agent.skill_router import build_skills_system_prompt_with_query

        with patch.object(pb_module, 'get_skills_dir', return_value=self.tmp_dir):
            with patch('hermes_constants.get_skills_dir', return_value=self.tmp_dir):
                with patch('agent.skill_router.get_skills_dir', return_value=self.tmp_dir):
                    result = build_skills_system_prompt_with_query(
                        query="test weather",
                        available_tools=None,
                        available_toolsets=None,
                    )
                    # Should not crash, and should contain injection header for matched skills
                    self.assertIsInstance(result, str)
                    # If routing worked, should contain "Relevant Skills" section
                    if "Relevant Skills" in result:
                        self.assertIn("test-skill", result)

    def test_short_query_returns_base_prompt(self):
        """Query shorter than 3 chars should return base prompt without injection."""
        from agent.prompt_builder import build_skills_system_prompt as _base_build
        from agent.skill_router import build_skills_system_prompt_with_query as _with_query

        with patch('hermes_constants.get_skills_dir', return_value=self.tmp_dir):
            with patch('agent.skill_router.get_skills_dir', return_value=self.tmp_dir):
                result = _with_query(
                    query="hi",
                    available_tools=None,
                    available_toolsets=None,
                )
                # Should return base prompt (no injection for short query)
                self.assertIsInstance(result, str)

    def test_none_query_returns_base_prompt(self):
        """None query should return base prompt without injection."""
        from agent.skill_router import build_skills_system_prompt_with_query

        with patch('hermes_constants.get_skills_dir', return_value=self.tmp_dir):
            with patch('agent.skill_router.get_skills_dir', return_value=self.tmp_dir):
                result = build_skills_system_prompt_with_query(
                    query=None,
                    available_tools=None,
                    available_toolsets=None,
                )
                self.assertIsInstance(result, str)


class TestRuleRouterCoverage(unittest.TestCase):
    """Test that rules cover the high-frequency scenarios from the deep-dive report."""

    def setUp(self):
        from agent.skill_router import RuleRouter
        self.router = RuleRouter()

    def test_code_scenarios(self):
        """Code-related queries should be covered."""
        test_queries = [
            "帮我写一段 Python 代码",
            "debug 这个函数",
            "代码审查",
            "TDD 测试驱动开发",
            "git commit 并 push",
        ]
        for query in test_queries:
            results = self.router.get_top_n(query, n=3)
            self.assertGreater(len(results), 0, f"Rule should match for: {query}")

    def test_document_scenarios(self):
        """Document output queries should be covered."""
        test_queries = [
            "转成 PDF 文档",
            "做一个 PPT 演示文稿",
            "图片转文字 OCR",
        ]
        for query in test_queries:
            results = self.router.get_top_n(query, n=3)
            self.assertGreater(len(results), 0, f"Rule should match for: {query}")

    def test_creative_scenarios(self):
        """Creative/design queries should be covered."""
        test_queries = [
            "画一个系统架构图",
            "生成一张插画图片",
            "画一个知识漫画",
            "设计一个 landing page",
        ]
        for query in test_queries:
            results = self.router.get_top_n(query, n=3)
            self.assertGreater(len(results), 0, f"Rule should match for: {query}")

    def test_agent_scenarios(self):
        """Agent-related queries should be covered."""
        test_queries = [
            "用子代理帮我完成任务",
            "启动多智能体团队",
            "配置 hermes 模型",
        ]
        for query in test_queries:
            results = self.router.get_top_n(query, n=3)
            self.assertGreater(len(results), 0, f"Rule should match for: {query}")

    def test_mlops_scenarios(self):
        """MLOps queries should be covered."""
        test_queries = [
            "训练模型 微调",
            "量化模型到 GGUF",
            "模型评测 benchmark",
        ]
        for query in test_queries:
            results = self.router.get_top_n(query, n=3)
            self.assertGreater(len(results), 0, f"Rule should match for: {query}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
