#!/usr/bin/env python3
"""
Hybrid Skill Router — 三层路由架构

    RuleRouter (规则引擎, ~10ms)
        ↓ 命中
    直接加载 Top-N skills

        ↓ 未命中
    BM25SkillIndex (BM25 检索, ~50ms)
        ↓ 有结果
    注入 Top-K skills

        ↓ 无结果
    LLMRouter (LLM 兜底, ~500ms)
        ↓ 返回
    skill_view() 加载匹配 skills

覆盖 ~95% 常见场景，规则路由 <10ms 且零 token 消耗。

Usage:
    router = SkillRouter()
    skills = router.route("帮我写一段 Python 代码并提交到 git", top_k=5)
    # Returns: [(skill_name, skill_path, score, method), ...]
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hermes_constants import get_skills_dir

logger = logging.getLogger(__name__)


# =========================================================================
# Phase 2: Rule Router — 关键词/正则规则引擎
# =========================================================================

class RuleRouter:
    """
    基于规则的技能路由。覆盖 ~80% 的常见场景，<10ms，零 token 消耗。
    
    规则格式：(priority, name, patterns, matched_skill_ids)
    - priority: 优先级，越高越先匹配
    - patterns: 关键词列表（支持 in-operator 和正则）
    - matched_skill_ids: 匹配后返回的 skill ID 列表
    """
    
    # Core rules — 根据实际 130 个 SKILL.md 的前 20 大 domain 定制
    RULES = [
        # ── 代码/编程 (最高优先级) ──
        (20, "code-work", [
            "写代码", "编写代码", "coding", "write code", "debug",
            "调试", "重构", "refactor", "bug", "报错", "error",
            "function", "implement", "开发", "写个", "帮我写", "编程",
        ], ["codeengine-toolchain", "subagent-driven-development", "systematic-debugging"]),
        
        (20, "code-review", [
            "代码审查", "code review", "review code", "审查代码",
            "代码检查", "check code", "review",
        ], ["requesting-code-review", "github-code-review"]),
        
        (20, "test-driven", [
            "TDD", "测试驱动", "test driven", "单元测试", "unit test",
            "integration test", "自动化测试",
        ], ["test-driven-development"]),
        
        (20, "plan", [
            "规划", "方案", "implementation plan", "怎么实现", "怎么做",
            "设计方案", "架构设计",
        ], ["writing-plans", "plan"]),
        
        (20, "git-ops", [
            "git", "commit", "push", "pull", "branch", "rebase",
            "合并", "cherry-pick", "stash", "git操作",
        ], ["github-pr-workflow", "codebase-inspection"]),
        
        # ── GitHub 专属 ──
        (20, "github-pr", ["PR", "pull request", "pr", "pr Merge", "合并PR"], ["github-pr-workflow"]),
        (20, "github-issue", ["issue", "bug 报告", "问题报告"], ["github-issues"]),
        (20, "github-auth", ["github 认证", "github auth", "ssh key"], ["github-auth"]),
        (20, "github-repo", ["fork", "仓库", "repo", "clone", "初始化"], ["github-repo-management"]),
        
        # ── 文档/输出 ──
        (15, "pdf", ["PDF", "pdf", "转 PDF", "转成 PDF", "pdf 生成", "操作手册"], ["document-conversion"]),
        (15, "pptx", ["PPT", "ppt", "演示", "幻灯片", "powerpoint", "report"], ["powerpoint"]),
        (15, "docx", ["Word", "docx", "文档", "文档格式", "word 文档"], ["document-conversion"]),
        (15, "ocr", ["OCR", "识别图片文字", "扫描文字", "图片转文字"], ["ocr-and-documents"]),
        
        # ── 创意/设计 ──
        (15, "diagram", ["架构图", "diagram", "流程图", "架构图", "visio", "系统图"], ["architecture-diagram", "excalidraw"]),
        (15, "image-gen", ["画图", "生成图片", "生成图", "图片生成", "draw image", "插画"], ["image_gen"]),
        (15, "comic", ["漫画", "知识漫画", "comic"], ["baoyu-comic"]),
        (15, "web-design", ["网页设计", "landing page", "网站", "web design", "前端"], ["popular-web-designs", "sketch"]),
        (15, "animation", ["动画", "manim", "video", "演示视频", "数学动画"], ["manim-video"]),
        (15, "pixel-art", ["像素画", "pixel art", "像素", "像素图"], ["pixel-art"]),
        (15, "p5js", ["p5", "p5js", "webgl", "canvas", "shader", "webgl", "着色器"], ["p5js"]),
        (15, "design-md", ["design.md", "token 文件", "设计文件"], ["design-md"]),
        
        # ── Agent/多智能体 ──
        (20, "agent-work", [
            "代理", "worker", "subagent", "delegate", "委派", "多代理",
            "多智能体", "multi-agent", "团队", "worker池", "tmux",
        ], ["multi-agent-orchestration", "agent-handoff"]),
        
        (20, "hermes-config", [
            "hermes 配置", "hermes 设置", "gateway", "模型", "provider",
            "hermes设置", "hermes配置",
        ], ["hermes-agent"]),
        
        (20, "claude-code", ["claude code", "claude_code"], ["claude-code"]),
        (20, "codex", ["codex", "openai codex"], ["codex"]),
        
        # ── 搜索/研究 ──
        (10, "web-search", ["搜索", "search", "网上", "找一下", "查找"], ["web"]),
        (10, "research", ["调研", "research", "研究", "论文", "paper", "文献"], [
            "open-source-codebase-analysis", "blogwatcher", "arxiv", "llm-wiki",
            "web-research",
        ]),
        (10, "market", ["市场", "市场数据", "polymarket"], ["polymarket"]),
        
        # ── MLOps/模型 ──
        (15, "mlops-serve", ["模型服务", "serve model", "vllm", "推理", "部署模型"], ["serving-llms-vllm"]),
        (15, "mlops-finetune", ["微调", "fine-tune", "训练", "训练模型", "sft", "dpo"], [
            "fine-tuning-with-trl", "unsloth", "axolotl",
        ]),
        (15, "mlops-eval", ["模型评测", "benchmark", "evaluation", "mmlu", "gsm8k"], ["evaluating-llms-harness"]),
        (15, "mlops-inference", ["量化", "gguf", "ggml", "llama-cpp", "本地推理"], ["llama-cpp"]),
        (15, "mlops-agent", ["dspy", "Declarative LM"], ["dspy"]),
        
        # ── 媒体/娱乐 ──
        (10, "spotify", ["spotify", "播放音乐", "歌", "歌曲", "音乐"], ["spotify"]),
        (10, "youtube", ["youtube", "yt", "视频", "视频", "transcript", "youtube转录"], ["youtube-content"]),
        (10, "gif", ["gif", "动图", "搜索gif"], ["gif-search"]),
        (10, "music", ["音乐生成", "suno", "heartmula", "作曲"], ["heartmula", "audiocraft"]),
        
        # ── 天气/位置 ──
        (10, "weather", ["天气", "temperature", "天气预报", "今天冷吗", "climate", "下雨"], ["weather"]),
        (10, "nearby", ["附近", "附近餐厅", "餐厅", "药店", "find nearby", "附近搜索"], ["find-nearby"]),
        
        # ── GitHub 代码知识图谱 ──
        (15, "gitnexus", ["代码知识图谱", "gitnexus", "代码图谱", "影响分析"], ["gitnexus-integration"]),
        
        # ── 记忆/笔记 ──
        (10, "memory", ["记忆", "nocturne", "note", "笔记系统"], ["nocturne-memory"]),
        (10, "obsidian", ["obsidian", "笔记", "obsidian vault"], ["obsidian"]),
        
        # ── 邮件 ──
        (10, "email", ["邮件", "email", "imsg", "发送邮件"], ["himalaya"]),
        
        # ── 备份 ──
        (10, "backup", ["备份", "restic", "nas"], ["deploy-backup-system"]),
        
        # ── Apple ──
        (10, "apple-reminder", ["提醒", "提醒事项", "remind", "reminder"], ["apple-reminders"]),
        (10, "apple-notes", ["笔记", "notes", "备忘录", "apple笔记"], ["apple-notes"]),
        (10, "apple-imessage", ["消息", "imessage", "短信", "iMessage"], ["imessage"]),
        (10, "findmy", ["找手机", "find my", "airtag", "追踪"], ["findmy"]),
        
        # ── 智能家居 ──
        (10, "lights", ["灯", "light", "hue", "场景", "智能灯", "关灯"], ["openhue"]),
        
        # ── 项目管理 ──
        (15, "kanban", ["看板", "kanban", "任务看板", "任务管理"], ["kanban-orchestrator", "kanban-worker"]),
        (15, "linear", ["linear", "项目管理"], ["linear"]),
        (15, "notion", ["notion", "notion数据库"], ["notion"]),
        (15, "airtable", ["airtable"], ["airtable"]),
        
        # ── 自适应深度 ──
        (10, "adaptive-depth", ["深度分析", "deep dive", "详细分析", "深入"], ["adaptive-depth"]),
        
        # ── 自适应迭代 ──
        (10, "act-iteration", ["自适应迭代", "act iteration"], ["act-iteration"]),
        
        # ── 技能开发 ──
        (15, "skill-dev", ["skill", "技能开发", "创建skill", "skill设计"], ["effective-skill-design", "hermes-agent-skill-authoring"]),
        
        # ── 安全 ──
        (10, "security", ["安全", "security", "漏洞", "OWASP", "渗透测试"], ["security-and-hardening"]),
        
        # ── 交付/部署 ──
        (10, "deploy", ["部署", "deploy", "webhook"], ["webhook-subscriptions", "deploy-backup-system"]),
        
        # ── 增量开发 ──
        (10, "incremental", ["增量开发", "incremental", "逐步实现"], ["incremental-implementation"]),
        
        # ── 调试工具 ──
        (10, "debugging", ["debugger", "调试", "断点调试", "pydev"], ["python-debugpy"]),
        
        # ── 上下文优化 ──
        (10, "context-optimize", ["上下文优化", "context 优化", "压缩", "context压缩"], [
            "context-compression-mitigation", "hermes-context-optimizer",
        ]),
        
        # ── MCP ──
        (10, "mcp", ["mcp", "MCP", "模型上下文协议"], ["native-mcp"]),
        
        # ── 数据结构/算法 ──
        (10, "data-science", ["数据分析", "数据科学", "pandas", "numpy", "jupyter"], ["jupyter-live-kernel"]),
        
        # ── 图片生成辅助 ──
        (10, "comfyui", ["comfyui", "stable diffusion"], ["comfyui"]),
        
        # ── 歌曲创作 ──
        (10, "songwriting", ["写歌", "歌词", "歌曲", "suno提示词"], ["songwriting-and-ai-music"]),
        
        # ── 人机对话 ──
        (10, "humanizer", ["人性化", "humanizer", "写作润色", "去AI味"], ["humanizer"]),
        
        # ── 代码审查/改进 ──
        (10, "code-simplify", ["简化代码", "代码清理", "代码可读性"], ["code-simplification"]),
        
        # ── 专项工作 ──
        (10, "tracing", ["tracing", "追踪", "run追踪"], ["agent-tracing"]),
        
        # ── 电子书/OCR ──
        (10, "ebook", ["电子书", "epub"], ["ocr-and-documents"]),
        
        # ── nano-pdf 编辑 ──
        (10, "nano-pdf", ["nano-pdf", "PDF编辑", "修改PDF"], ["nano-pdf"]),
    ]

    def route(self, user_input: str) -> List[Tuple[str, str, float, str]]:
        """
        根据用户输入返回匹配的 skill 列表。
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            [(skill_name, skill_path, score, "rule"), ...]
        """
        matched = {}  # skill_id -> max_priority
        
        for priority, rule_name, patterns, skill_ids in self.RULES:
            input_lower = user_input.lower()
            for pattern in patterns:
                pat_lower = pattern.lower()
                if pat_lower in input_lower:
                    for sid in skill_ids:
                        if sid not in matched or matched[sid] < priority:
                            matched[sid] = priority
                    break  # 一个 rule 命中即跳过同 rule 其他 pattern
        
        if not matched:
            return []
        
        result = sorted(matched.keys(), key=lambda x: -matched[x])
        return [(sid, sid, float(priority), "rule") for sid, priority in 
                [(sid, matched[sid]) for sid in result]]

    def get_top_n(self, user_input: str, n: int = 3) -> List[str]:
        """返回 Top-N 匹配的 skill IDs。"""
        return [r[0] for r in self.route(user_input)[:n]]


# =========================================================================
# Phase 3: LLM Fallback Router
# =========================================================================

class LLMRouter:
    """
    LLM 兜底路由。
    
    仅在规则路由和 BM25 均未返回结果时触发。
    使用压缩的 skill frontmatter（仅 name + 前80 chars description）构建 ~50-100 token 的 system prompt。
    
    通过 hermes_tools.text_to_speech 或终端调用轻量 LLM 做意图分类。
    返回 JSON 格式的 skill IDs。
    """
    
    SYSTEM_PROMPT = """你是一名技能路由助手。根据用户的输入，从以下技能列表中选择最相关的 1-3 个技能。

可用技能列表（仅名称和简短描述）：
{skill_list}

请严格按以下 JSON 格式返回（不要输出其他内容）：
{{"selected_skills": ["skill1", "skill2"]}}

如果无法确定，返回：{{"selected_skills": []}}"""

    def route(self, user_input: str, all_skills_fm: List[Dict[str, Any]]) -> List[Tuple[str, str, float, str]]:
        """
        用 LLM 做意图分类。
        
        Args:
            user_input: 用户输入
            all_skills_fm: 全部 skill 的 frontmatter 列表（已压缩）
            
        Returns:
            [(skill_name, skill_path, score, "llm"), ...]
        """
        if not all_skills_fm:
            return []
        
        # 压缩 skill 列表：仅 name + 前80 chars description
        skill_entries = []
        for s in all_skills_fm[:50]:  # 最多前50个（防止 prompt 过大）
            name = s.get("name", "")
            desc = s.get("description", "")[:80]
            skill_entries.append(f"- {name}: {desc}")
        
        skill_list_text = "\n".join(skill_entries)
        prompt = self.SYSTEM_PROMPT.format(skill_list_text=skill_list_text)
        
        # 构造轻量 LLM 请求
        llm_request = {
            "model": "qwen3.6-35b-a3b-8bit",  # 用小模型
            "messages": [
                {"role": "system", "content": "你是一个技能路由助手，只返回 JSON。"},
                {"role": "user", "content": f"用户输入: {user_input}\n\n请从上述技能中选择最相关的1-3个。"}
            ],
            "max_tokens": 100,
            "temperature": 0.0,
        }
        
        # 通过 hermes CLI 调用 LLM（兼容各种 provider）
        try:
            import subprocess
            import os as _os
            result = subprocess.run(
                ["hermes", "chat", "-m", "qwen3.6-35b-a3b-8bit", 
                 "-s", "只返回JSON格式的技能列表",
                 "-p", f"从以下技能中为以下用户输入选择最相关的1-3个：\n\n用户输入: {user_input}\n\n技能列表:\n{skill_list_text}"],
                capture_output=True, text=True, timeout=10,
                cwd=Path.home() / ".hermes"
            )
            
            if result.returncode != 0:
                logger.debug("LLM router CLI failed: %s", result.stderr[:200])
                return []
            
            # 从输出中提取 JSON
            text = result.stdout.strip()
            json_match = re.search(r'\{[^{}]*"selected_skills"[^{}]*\}', text)
            if json_match:
                data = json.loads(json_match.group())
                skill_ids = data.get("selected_skills", [])
                return [(sid, sid, 0.5, "llm") for sid in skill_ids if sid]
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.debug("LLM router failed: %s", e)
        
        return []


# =========================================================================
# Unified Skill Router — 三层串联
# =========================================================================

class SkillRouter:
    """
    统一技能路由器。三层架构：
    
    1. RuleRouter — 规则引擎（~10ms，覆盖 ~80% 高频场景）
    2. BM25SkillIndex — BM25 检索（~50ms，覆盖 ~15% 中频场景）
    3. LLMRouter — LLM 兜底（~500ms，覆盖 ~2% 长尾场景）
    
    Usage:
        router = SkillRouter()
        results = router.route("帮我写一段 Python 代码并测试", top_k=5)
        # Returns: [(skill_name, skill_path, score, method), ...]
    """
    
    def __init__(self, skills_dir: Optional[str] = None):
        self.skills_dir = Path(skills_dir) if skills_dir else get_skills_dir()
        self.rule_router = RuleRouter()
        self.bm25_index = None  # lazy init
        self._cached_fm_index: Optional[List[Dict[str, Any]]] = None
    
    def _get_bm25_index(self):
        """Lazy init BM25 index."""
        if self.bm25_index is None:
            from agent.bm25_skill_index import BM25SkillIndex
            self.bm25_index = BM25SkillIndex()
        return self.bm25_index
    
    def _get_fm_index(self) -> List[Dict[str, Any]]:
        """
        获取压缩的 frontmatter 索引（用于 LLM 路由）。
        只包含 name, description, tags，不含完整内容。
        """
        if self._cached_fm_index is not None:
            return self._cached_fm_index
        
        try:
            from agent.skill_utils import parse_frontmatter, skill_matches_platform, get_disabled_skill_names
        except ImportError:
            self._cached_fm_index = []
            return []
        
        entries = []
        disabled = get_disabled_skill_names()
        
        for skill_md in self.skills_dir.rglob("SKILL.md"):
            if any(part.startswith(".") and part not in (".", "..")
                   for part in skill_md.relative_to(self.skills_dir).parts):
                continue
            
            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter, body = parse_frontmatter(content)
            except Exception:
                continue
            
            name = frontmatter.get("name", skill_md.parent.name)
            description = frontmatter.get("description", "")
            tags = frontmatter.get("tags", []) or []
            if isinstance(tags, str):
                tags = [tags]
            
            if name in disabled:
                continue
            if not skill_matches_platform(frontmatter):
                continue
            
            entries.append({
                "name": str(name),
                "description": str(description),
                "tags": [str(t) for t in tags],
                "path": str(skill_md.resolve()),
            })
        
        self._cached_fm_index = entries
        return entries
    
    def route(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float, str]]:
        """
        三层路由，按优先级串联。
        
        Args:
            query: 用户查询
            top_k: 最大返回数量
            
        Returns:
            [(skill_name, skill_path, score, method), ...]
            method: "rule" | "bm25" | "llm"
        """
        if not query or len(query.strip()) < 3:
            return []
        
        # ── Layer 1: Rule Router ──
        rule_results = self.rule_router.route(query)
        if rule_results:
            logger.debug("RuleRouter matched %d skills: %s", len(rule_results), [r[0] for r in rule_results])
            return [(name, path, score, method) for name, path, score, method in rule_results[:top_k]]
        
        # ── Layer 2: BM25 Index ──
        try:
            bm25_results = self._get_bm25_index().query(query, top_k=top_k)
            if bm25_results:
                # BM25 returns (skill_name, skill_path, score)
                logger.debug("BM25 matched %d skills: %s", len(bm25_results), [r[0] for r in bm25_results])
                return [(name, path, score, "bm25") for name, path, score in bm25_results[:top_k]]
        except Exception as e:
            logger.debug("BM25 route failed: %s", e)
        
        # ── Layer 3: LLM Fallback ──
        llm_results = LLMRouter().route(query, self._get_fm_index())
        if llm_results:
            logger.debug("LLMRouter matched %d skills: %s", len(llm_results), [r[0] for r in llm_results])
            return [(name, path, score, method) for name, path, score, method in llm_results[:top_k]]
        
        logger.debug("All routing layers returned empty for query: %s", query[:50])
        return []
    
    def get_top_skill_ids(self, query: str, n: int = 3) -> List[str]:
        """快速查询，只返回 skill IDs（不加载路径和内容）。"""
        results = self.route(query, top_k=n)
        return [r[0] for r in results]
    
    def get_stats(self) -> dict:
        """返回路由系统统计信息。"""
        fm_count = len(self._get_fm_index())
        bm25_stats = {}
        try:
            bm25_stats = self._get_bm25_index().get_stats()
        except Exception:
            pass
        
        return {
            "total_skills": fm_count,
            "rule_count": len(RuleRouter.RULES),
            "bm25_skills": bm25_stats.get("num_skills", 0),
            "bm25_terms": bm25_stats.get("num_terms", 0),
        }


# =========================================================================
# Convenience function — 供 prompt_builder 直接调用
# =========================================================================

def build_skills_system_prompt_with_query(
    query: str,
    available_tools: "set[str] | None" = None,
    available_toolsets: "set[str] | None" = None,
) -> str:
    """Build skill system prompt with hybrid routing (Rule + BM25 + LLM fallback)."""
    from agent.prompt_builder import build_skills_system_prompt as _base_build
    
    # Step 1: Build standard skill index (frontmatter only)
    base_prompt = _base_build(
        available_tools=available_tools,
        available_toolsets=available_toolsets,
    )
    
    if not query or len(query.strip()) < 3:
        return base_prompt
    
    # Step 2: Hybrid routing
    try:
        router = SkillRouter()
        results = router.route(query, top_k=5)
    except Exception as e:
        logger.debug("Hybrid skill routing failed, falling back to standard prompt: %s", e)
        return base_prompt
    
    if not results:
        return base_prompt
    
    # Step 3: Load full content for matched skills
    _seen: set[str] = set()
    _sections: list[str] = []
    
    for skill_name, skill_path, score, method in results:
        if skill_name in _seen:
            continue
        _seen.add(skill_name)
        
        try:
            full_content = Path(skill_path).read_text(encoding="utf-8")
        except Exception:
            continue
        
        # Parse frontmatter + body
        prefix = ""
        body = full_content
        if "---\n" in body:
            parts = body.split("---\n", 2)
            if len(parts) >= 3:
                prefix = parts[0] + "---\n" + parts[1] + "---\n"
                body = parts[2]
        
        # Limit injection per skill
        _MAX_BODY = 1500
        if len(body) > _MAX_BODY:
            body = body[:_MAX_BODY] + " [truncated]"
        
        _sections.append(
            f"### Skill: {skill_name} (method: {method}, score: {score:.3f})\n\n"
            f"{prefix}{body}"
        )
    
    if not _sections:
        return base_prompt
    
    injection = "\n\n## Relevant Skills (auto-selected by hybrid router)\n"
    injection += "The following skills are matched by our hybrid routing system. "
    injection += "Read and apply ONLY these skills. Do NOT load any other skills not listed below.\n"
    injection += "\n".join(_sections)
    
    return base_prompt + injection
