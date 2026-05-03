#!/usr/bin/env python3
"""
BM25 Skill Index — Lightweight BM25 retrieval over skill metadata.

Builds an inverted index from skill frontmatter (name, description, tags)
and queries it using the BM25 algorithm for relevance ranking.

Index is cached to disk and automatically rebuilt when any SKILL.md changes.

Usage:
    index = BM25SkillIndex()
    results = index.query("帮我调大语音音量", top_k=5)
    # Returns: [(skill_name, skill_path, score), ...]
"""

import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# BM25 hyperparameters (classic Okapi BM25)
BM25_K1 = 1.5  # Term frequency saturation
BM25_B = 0.75  # Length normalization


@dataclass
class SkillEntry:
    """A single skill's indexed metadata."""
    skill_name: str
    skill_path: str  # absolute path to SKILL.md
    description: str
    tags: List[str] = field(default_factory=list)
    name_text: str = ""
    indexed_text: str = ""

    def __post_init__(self):
        # Combine all searchable fields
        self.name_text = self.skill_name.lower()
        self.indexed_text = f"{self.name_text} {self.description.lower()} " + \
                            " ".join(t.lower() for t in self.tags)


@dataclass
class BM25IndexStats:
    """Statistics about the BM25 index."""
    num_documents: int = 0
    num_terms: int = 0
    total_tokens: int = 0
    avg_doc_length: float = 0.0
    build_time_ms: float = 0.0
    last_build: float = 0.0


class BM25SkillIndex:
    """BM25 index over Hermes skill metadata.

    Thread-safe singleton: multiple instances with the same skills_dir
    share the same in-memory index.
    """

    _instances: Dict[str, "BM25SkillIndex"] = {}
    _lock: Any = None  # lazy import threading.Lock

    def __init__(self, skills_dir: Optional[str] = None,
                 cache_path: Optional[str] = None):
        """Initialize BM25 skill index.

        Args:
            skills_dir: Path to skills directory. Defaults to ~/.hermes/skills/.
            cache_path: Path to cache file. Defaults to .bm25_index.json.
        """
        # Lazy import to avoid blocking cold-start
        import threading
        if BM25SkillIndex._lock is None:
            BM25SkillIndex._lock = threading.Lock()

        from hermes_constants import get_skills_dir
        self.skills_dir = Path(skills_dir) if skills_dir else get_skills_dir()
        self.cache_path = Path(cache_path) if cache_path else \
            self.skills_dir / ".bm25_index.json"

        # In-memory index state
        self._docs: List[SkillEntry] = []
        self._term_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._doc_freq: Dict[str, int] = defaultdict(int)
        self._idfs: Dict[str, float] = {}
        self._stats = BM25IndexStats()

        # Load cache if available and fresh
        self._load()

    # ── Index Building ──────────────────────────────────────────────────

    def build(self, force: bool = False) -> "BM25SkillIndex":
        """Build or rebuild the BM25 index from skill files.

        Args:
            force: If True, rebuild even if cache exists and is fresh.

        Returns:
            self (for chaining)
        """
        build_start = time.time()

        # Parse all SKILL.md files
        skill_entries = self._scan_skills()

        # Check if rebuild is needed
        if not force and self._docs and not self._is_cache_stale(skill_entries):
            logger.debug("BM25 index is up-to-date, skipping rebuild (%d skills)", len(self._docs))
            return self

        logger.info("Building BM25 skill index (%d skills in %s)...", len(skill_entries), self.skills_dir)

        # Build index
        self._docs = skill_entries
        self._build_inverted_index()

        # Update stats
        self._stats.num_documents = len(self._docs)
        self._stats.total_tokens = sum(len(doc.indexed_text.split()) for doc in self._docs)
        self._stats.num_terms = len(self._idfs)
        self._stats.avg_doc_length = self._stats.total_tokens / max(1, self._stats.num_documents)
        self._stats.build_time_ms = (time.time() - build_start) * 1000
        self._stats.last_build = time.time()

        # Save cache
        self._save()

        logger.info("BM25 index built: %d skills, %d terms, %.1fms",
                     self._stats.num_documents, self._stats.num_terms,
                     self._stats.build_time_ms)
        return self

    def _scan_skills(self) -> List[SkillEntry]:
        """Scan all SKILL.md files and extract metadata."""
        from agent.skill_utils import parse_frontmatter, skill_matches_platform, get_disabled_skill_names

        # Use provided skills_dir if set (for testing), otherwise use default
        if self.skills_dir:
            skills_roots = [self.skills_dir]
        else:
            from agent.skill_utils import get_all_skills_dirs
            skills_roots = get_all_skills_dirs()

        entries = []
        disabled = get_disabled_skill_names()

        for skills_root in skills_roots:
            if not skills_root.exists():
                continue

            for skill_md in skills_root.rglob("SKILL.md"):
                # Skip hidden directories
                if any(part.startswith(".") and part not in (".", "..")
                       for part in skill_md.relative_to(skills_root).parts):
                    continue

                try:
                    content = skill_md.read_text(encoding="utf-8")
                    frontmatter, body = parse_frontmatter(content)
                except Exception as e:
                    logger.debug("Failed to parse SKILL.md %s: %s", skill_md, e)
                    continue

                # Extract metadata
                name = frontmatter.get("name", skill_md.parent.name)
                description = frontmatter.get("description", "")
                tags = frontmatter.get("tags", []) or []
                if isinstance(tags, str):
                    tags = [tags]

                # Skip disabled skills
                if name in disabled:
                    continue

                # Check platform compatibility
                if not skill_matches_platform(frontmatter):
                    continue

                entry = SkillEntry(
                    skill_name=str(name),
                    skill_path=str(skill_md.resolve()),
                    description=str(description),
                    tags=[str(t) for t in tags],
                )
                entries.append(entry)

        return entries

    def _build_inverted_index(self) -> None:
        """Build inverted index from all skill entries."""
        # Tokenize all documents
        for doc in self._docs:
            tokens = self._tokenize(doc.indexed_text)
            # Track term frequencies for this document
            doc_terms = set(tokens)
            for term in doc_terms:
                self._doc_freq[term] += 1

        # Compute IDF values
        N = len(self._docs)
        for term, df in self._doc_freq.items():
            # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            self._idfs[term] = (N - df + 0.5) / (df + 0.5) + 1.0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into BM25-compatible tokens.

        Handles English words, numbers, and Chinese text.
        Chinese text is tokenized at the character level for better matching.
        """
        text = text.lower().strip()
        # Extract English words and numbers
        words = re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)*', text)
        # Extract Chinese text blocks
        chinese_blocks = re.findall(r'[\u4e00-\u9fff]+', text)
        # Tokenize Chinese at character level for better recall
        chinese_tokens = []
        for block in chinese_blocks:
            chinese_tokens.extend(list(block))
        tokens = words + chinese_tokens
        return tokens

    # ── Query ───────────────────────────────────────────────────────────

    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Query the BM25 index and return top-k matching skills.

        Args:
            text: User query string.
            top_k: Number of results to return.

        Returns:
            List of (skill_name, skill_path, score) tuples, sorted by score descending.
        """
        if not self._docs:
            # Auto-build if index is empty
            self.build()

        query_tokens = self._tokenize(text)
        if not query_tokens:
            return []

        # Score each document
        scores = []
        for doc in self._docs:
            score = self._bm25_score(doc.indexed_text, query_tokens)
            if score > 0:
                scores.append((doc.skill_name, doc.skill_path, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[2], reverse=True)
        return scores[:top_k]

    def _bm25_score(self, doc_text: str, query_tokens: List[str]) -> float:
        """Compute BM25 score for a document against query tokens."""
        doc_tokens = self._tokenize(doc_text)
        if not doc_tokens:
            return 0.0

        doc_len = len(doc_tokens)
        N = len(self._docs)
        avg_len = self._stats.avg_doc_length or doc_len

        score = 0.0
        for q_term in query_tokens:
            # Count term frequency in document (unique occurrences only)
            tf = sum(1 for t in doc_tokens if t == q_term)
            if tf == 0:
                continue

            idf = self._idfs.get(q_term, 0.0)
            if idf <= 0:
                continue

            # BM25 term frequency component
            k1 = BM25_K1
            b = BM25_B
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * doc_len / avg_len)
            tf_score = numerator / (denominator + 1e-10)

            score += idf * tf_score

        return score

    # ── Caching ─────────────────────────────────────────────────────────

    def _save(self) -> None:
        """Save the index to disk cache."""
        try:
            cache_data = {
                "stats": {
                    "num_documents": self._stats.num_documents,
                    "num_terms": self._stats.num_terms,
                    "avg_doc_length": self._stats.avg_doc_length,
                    "build_time_ms": self._stats.build_time_ms,
                    "last_build": self._stats.last_build,
                },
                "docs": [
                    {
                        "skill_name": doc.skill_name,
                        "skill_path": doc.skill_path,
                        "description": doc.description,
                        "tags": doc.tags,
                    }
                    for doc in self._docs
                ],
                "mtimes": {
                    doc.skill_path: os.path.getmtime(doc.skill_path)
                    for doc in self._docs
                },
            }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(cache_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.debug("Failed to save BM25 cache: %s", e)

    def _load(self) -> bool:
        """Load index from disk cache. Returns True if loaded successfully."""
        if not self.cache_path.exists():
            return False

        try:
            cache_data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.debug("Failed to load BM25 cache: %s", e)
            return False

        stats = cache_data.get("stats", {})
        self._stats = BM25IndexStats(
            num_documents=stats.get("num_documents", 0),
            num_terms=stats.get("num_terms", 0),
            avg_doc_length=stats.get("avg_doc_length", 0),
            build_time_ms=stats.get("build_time_ms", 0),
            last_build=stats.get("last_build", 0),
        )

        # Rebuild skill entries from cache
        cached_docs = cache_data.get("docs", [])
        cached_mtimes = cache_data.get("mtimes", {})
        if not cached_docs:
            return False

        # Check if any file has changed since last build
        stale = False
        for doc_info in cached_docs:
            path = doc_info.get("skill_path", "")
            expected_mtime = cached_mtimes.get(path)
            if expected_mtime is None:
                stale = True
                break
            try:
                if os.path.getmtime(path) != expected_mtime:
                    stale = True
                    break
            except OSError:
                stale = True
                break

        if not stale:
            self._docs = [
                SkillEntry(
                    skill_name=doc.get("skill_name", ""),
                    skill_path=doc.get("skill_path", ""),
                    description=doc.get("description", ""),
                    tags=doc.get("tags", []),
                )
                for doc in cached_docs
            ]
            self._build_inverted_index()
            logger.debug("BM25 index loaded from cache (%d skills)", len(self._docs))
            return True

        return False

    def _is_cache_stale(self, current_entries: List[SkillEntry]) -> bool:
        """Check if any current SKILL.md files differ from cached state."""
        if not self.cache_path.exists():
            return True

        try:
            cached_mtimes = json.loads(self.cache_path.read_text(encoding="utf-8")).get("mtimes", {})
        except Exception:
            return True

        current_paths = {doc.skill_path for doc in current_entries}
        cached_paths = set(cached_mtimes.keys())

        # If the set of files changed, rebuild
        if current_paths != cached_paths:
            return True

        # Check each file's mtime
        for path in current_paths:
            try:
                if os.path.getmtime(path) != cached_mtimes.get(path):
                    return True
            except OSError:
                return True

        return False

    # ── Utilities ───────────────────────────────────────────────────────

    def is_stale(self) -> bool:
        """Check if any skill files have changed since the index was built."""
        current_entries = self._scan_skills()
        return self._is_cache_stale(current_entries)

    def get_stats(self) -> dict:
        """Return index statistics."""
        return {
            "num_skills": self._stats.num_documents,
            "num_terms": self._stats.num_terms,
            "avg_doc_length": round(self._stats.avg_doc_length, 1),
            "build_time_ms": round(self._stats.build_time_ms, 1),
            "last_build": self._stats.last_build,
            "cache_path": str(self.cache_path),
        }

    def force_rebuild(self) -> "BM25SkillIndex":
        """Force a full rebuild of the index."""
        self._docs = []
        self._term_freq.clear()
        self._doc_freq.clear()
        self._idfs.clear()
        return self.build(force=True)
