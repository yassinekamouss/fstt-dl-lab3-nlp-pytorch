"""Arabic Wikipedia web scraper and relevance scoring module."""

from __future__ import annotations

import hashlib
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import numpy as np
import requests
from bs4 import BeautifulSoup, Tag
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

_ARABIC_BLOCK_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]")
_CITATION_RE = re.compile(r"\[[^\]]{1,40}\]")
_BIDI_CONTROL_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")
_MULTI_SPACE_RE = re.compile(r"\s+")
_TASHKEEL_RE = re.compile(r"[\u0617-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]")
_NON_ARABIC_RE = re.compile(r"[^\u0600-\u06FF\s]")


def _basic_normalize_arabic(text: str) -> str:
    """Lightweight Arabic normalization used internally for deduplication and scoring."""
    text = _TASHKEEL_RE.sub("", text)
    text = re.sub(r"[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    text = _NON_ARABIC_RE.sub(" ", text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def arabic_char_ratio(text: str) -> float:
    """Compute the proportion of Arabic-script characters among non-whitespace chars."""
    compact = _MULTI_SPACE_RE.sub("", text)
    if not compact:
        return 0.0
    arabic_chars = len(_ARABIC_BLOCK_RE.findall(compact))
    return arabic_chars / len(compact)


def clean_wiki_paragraph(text: str) -> str:
    """Strip Wikipedia citation markers, invisible control characters, and extra spaces."""
    cleaned = _BIDI_CONTROL_RE.sub("", text)
    cleaned = _CITATION_RE.sub(" ", cleaned)
    cleaned = cleaned.replace("\xa0", " ")
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def extract_wiki_links(container: Tag) -> list[str]:
    """Extract clean Arabic Wikipedia article titles from internal wiki links."""
    titles: list[str] = []
    seen: set[str] = set()

    for anchor in container.find_all("a", href=True):
        href = str(anchor["href"])
        if not href.startswith("/wiki/"):
            continue
        raw_slug = href[len("/wiki/") :].split("#", 1)[0].split("?", 1)[0]
        if not raw_slug:
            continue
        title = unquote(raw_slug).replace("_", " ").strip()
        if not title or ":" in title or title == "الصفحة الرئيسية":
            continue
        if title not in seen:
            seen.add(title)
            titles.append(title)

    return titles


class BaseScraper:
    """Base scraper interface for collecting text datasets."""

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> Any:
        """Run the scraping pipeline and return collected records."""
        raise NotImplementedError("Implement scraping logic in BaseScraper.run().")


# Backward-compatible alias matching the initial starter class name.
Scraper = BaseScraper


class ArabicWikipediaScraper(BaseScraper):
    """Scrape Arabic Wikipedia articles across relevance tiers using BeautifulSoup4."""

    _NOISY_SELECTORS: tuple[str, ...] = (
        "table",
        "div.infobox",
        "div.navbox",
        "div.vertical-navbox",
        "div.reflist",
        "div.hatnote",
        "div.thumb",
        "div.toc",
        "div#toc",
        "div.printfooter",
        "div.catlinks",
        "sup.reference",
        "span.mw-editsection",
        "span.mwe-math-element",
        "ol.references",
        "style",
        "script",
        "noscript",
    )

    def __init__(
        self,
        output_dir: str | Path,
        config: dict[str, Any] | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__(output_dir=output_dir)
        scraping_cfg = (config or {}).get("scraping", config or {})

        self.base_url: str = str(scraping_cfg.get("base_url", "https://ar.wikipedia.org/wiki/"))
        self.user_agent: str = str(
            scraping_cfg.get(
                "user_agent",
                "FSTT-LSI-Lab3-ArabicNLP/1.0 (Academic Research)",
            )
        )
        self.request_delay: float = float(scraping_cfg.get("request_delay", 0.35))
        self.timeout: int = int(scraping_cfg.get("timeout", 15))
        self.max_retries: int = int(scraping_cfg.get("max_retries", 3))
        self.min_words: int = int(scraping_cfg.get("min_words", 25))
        self.max_words: int = int(scraping_cfg.get("max_words", 300))
        self.min_arabic_ratio: float = float(scraping_cfg.get("min_arabic_ratio", 0.70))
        self.max_pages_per_tier: int = int(scraping_cfg.get("max_pages_per_tier", 40))
        self.max_paragraphs_per_tier: int = int(scraping_cfg.get("max_paragraphs_per_tier", 850))
        self.enable_link_expansion: bool = bool(scraping_cfg.get("enable_link_expansion", True))
        self.seed_titles: dict[str, list[str]] = dict(scraping_cfg.get("seed_titles", {}))
        self.domain_lexicon: list[str] = list(scraping_cfg.get("domain_lexicon", []))

        self.cache_dir: Path | None = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = self._build_session()
        self._seen_hashes: set[str] = set()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Language": "ar,en;q=0.5",
            }
        )
        retry = Retry(
            total=self.max_retries,
            backoff_factor=0.6,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def build_article_url(self, title: str) -> str:
        """Build an encoded Arabic Wikipedia URL for a page title."""
        slug = quote(title.strip().replace(" ", "_"), safe="()_-'")
        return f"{self.base_url.rstrip('/')}/{slug}"

    def _cache_path(self, title: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.md5(title.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.html"

    def fetch_html(self, title: str) -> tuple[str | None, str]:
        """Fetch raw HTML for an Arabic Wikipedia page, using disk cache when available."""
        url = self.build_article_url(title)
        cache_file = self._cache_path(title)
        if cache_file is not None and cache_file.exists():
            return cache_file.read_text(encoding="utf-8"), url

        try:
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code != 200:
                logger.warning("Skipping '%s' (HTTP %s)", title, response.status_code)
                return None, url
            response.encoding = "utf-8"
            html = response.text
            if cache_file is not None:
                cache_file.write_text(html, encoding="utf-8")
            time.sleep(self.request_delay)
            return html, url
        except requests.RequestException as exc:
            logger.warning("Request failed for '%s': %s", title, exc)
            return None, url

    def parse_paragraphs(self, html: str) -> tuple[list[str], list[str]]:
        """Extract clean Arabic paragraphs and internal wiki links from Wikipedia HTML."""
        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        content = soup.select_one("div#mw-content-text div.mw-parser-output")
        if content is None:
            content = soup.select_one("div#mw-content-text") or soup.body
        if content is None:
            return [], []

        for selector in self._NOISY_SELECTORS:
            for element in content.select(selector):
                element.decompose()

        links = extract_wiki_links(content)
        paragraphs: list[str] = []

        for p_tag in content.find_all("p"):
            text = clean_wiki_paragraph(p_tag.get_text(" ", strip=True))
            if not text:
                continue
            words = text.split()
            if not (self.min_words <= len(words) <= self.max_words):
                continue
            if arabic_char_ratio(text) < self.min_arabic_ratio:
                continue
            paragraphs.append(text)

        return paragraphs, links

    def scrape_tier(
        self,
        tier_name: str,
        seed_titles: list[str],
        visited_titles: set[str],
        progress: bool = True,
    ) -> list[dict[str, Any]]:
        """Scrape a single relevance tier (core, related, or off_topic)."""
        queue: deque[str] = deque(seed_titles)
        queued_or_seen: set[str] = set(visited_titles) | set(seed_titles)
        records: list[dict[str, Any]] = []
        pages_fetched = 0

        iterator_fn = None
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator_fn = tqdm(total=self.max_pages_per_tier, desc=f"Scraping [{tier_name}]")
            except ImportError:
                iterator_fn = None

        try:
            while (
                queue
                and pages_fetched < self.max_pages_per_tier
                and len(records) < self.max_paragraphs_per_tier
            ):
                title = queue.popleft()
                if title in visited_titles:
                    continue
                visited_titles.add(title)

                html, url = self.fetch_html(title)
                if html is None:
                    continue
                pages_fetched += 1
                if iterator_fn is not None:
                    iterator_fn.update(1)

                paragraphs, links = self.parse_paragraphs(html)
                for paragraph in paragraphs:
                    if len(records) >= self.max_paragraphs_per_tier:
                        break
                    norm_hash = hashlib.md5(
                        _basic_normalize_arabic(paragraph).encode("utf-8")
                    ).hexdigest()
                    if norm_hash in self._seen_hashes:
                        continue
                    self._seen_hashes.add(norm_hash)
                    records.append(
                        {
                            "text": paragraph,
                            "title": title,
                            "url": url,
                            "tier": tier_name,
                            "word_count": len(paragraph.split()),
                        }
                    )

                if self.enable_link_expansion and len(records) < self.max_paragraphs_per_tier:
                    for linked_title in links:
                        if linked_title not in queued_or_seen:
                            queued_or_seen.add(linked_title)
                            queue.append(linked_title)
        finally:
            if iterator_fn is not None:
                iterator_fn.close()

        return records

    @staticmethod
    def _percentile_scale(values: np.ndarray, low_q: float = 5.0, high_q: float = 95.0) -> np.ndarray:
        if len(values) == 0:
            return values
        lo = float(np.percentile(values, low_q))
        hi = float(np.percentile(values, high_q))
        if hi <= lo + 1e-12:
            return np.zeros_like(values, dtype=float)
        return np.clip((values - lo) / (hi - lo), 0.0, 1.0)

    def compute_relevance_scores(
        self,
        records: list[dict[str, Any]],
        domain_lexicon: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compute continuous relevance scores in [0.0, 10.0] for each scraped paragraph."""
        if not records:
            return records

        lexicon = {_basic_normalize_arabic(term) for term in (domain_lexicon or self.domain_lexicon)}
        lexicon.discard("")

        norm_texts = [_basic_normalize_arabic(r["text"]) for r in records]
        tier_priors = np.array(
            [1.0 if r["tier"] == "core" else (0.5 if r["tier"] == "related" else 0.0) for r in records],
            dtype=float,
        )

        # Lexicon density normalized by sqrt(length) to avoid penalizing concise paragraphs
        lex_raw = np.zeros(len(records), dtype=float)
        for idx, norm_txt in enumerate(norm_texts):
            tokens = norm_txt.split()
            if not tokens or not lexicon:
                continue
            hits = sum(1 for tok in tokens if tok in lexicon or any(term in tok for term in lexicon if len(term) >= 4))
            lex_raw[idx] = hits / np.sqrt(len(tokens) + 1.0)
        lex_scaled = self._percentile_scale(lex_raw)

        # Leave-one-out TF-IDF cosine similarity against core AI topic centroid
        sim_raw = tier_priors.copy()
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=25000)
            tfidf = vectorizer.fit_transform(norm_texts)
            core_mask = np.array([r["tier"] == "core" for r in records], dtype=bool)
            if core_mask.sum() > 1:
                core_sum = np.asarray(tfidf[core_mask].sum(axis=0))
                core_count = int(core_mask.sum())
                for idx in range(len(records)):
                    if core_mask[idx]:
                        centroid = (core_sum - tfidf[idx].toarray()) / max(core_count - 1, 1)
                    else:
                        centroid = core_sum / core_count
                    sim_raw[idx] = float(cosine_similarity(tfidf[idx], centroid)[0, 0])
        except ImportError:
            logger.warning("scikit-learn not installed; falling back to lexicon + tier scoring.")

        sim_scaled = self._percentile_scale(sim_raw)
        composite = 0.50 * sim_scaled + 0.30 * lex_scaled + 0.20 * tier_priors
        scores = np.round(np.clip(composite * 10.0, 0.0, 10.0), 1)

        enriched: list[dict[str, Any]] = []
        for rec, score in zip(records, scores):
            item = dict(rec)
            item["score"] = float(score)
            enriched.append(item)
        return enriched

    def run(self, progress: bool = True) -> Any:
        """Execute multi-tier scraping, compute relevance scores, and save CSV artifacts."""
        import pandas as pd

        visited_titles: set[str] = set()
        all_records: list[dict[str, Any]] = []

        for tier_name in ("core", "related", "off_topic"):
            seeds = self.seed_titles.get(tier_name, [])
            if not seeds:
                continue
            tier_records = self.scrape_tier(
                tier_name=tier_name,
                seed_titles=seeds,
                visited_titles=visited_titles,
                progress=progress,
            )
            all_records.extend(tier_records)

        scored_records = self.compute_relevance_scores(all_records)
        df = pd.DataFrame(scored_records)
        if df.empty:
            raise RuntimeError("No Arabic Wikipedia paragraphs were collected.")

        cols = ["text", "score", "title", "url", "tier", "word_count"]
        df = df[cols].reset_index(drop=True)

        raw_path = self.output_dir / "arabic_wiki_raw.csv"
        df.to_csv(raw_path, index=False, encoding="utf-8-sig")

        spec_df = df[["text", "score"]].rename(columns={"text": "Text", "score": "Score"})
        spec_path = self.output_dir / "dataset_text_score.csv"
        spec_df.to_csv(spec_path, index=False, encoding="utf-8-sig")

        logger.info("Saved %d scraped rows to %s and %s", len(df), raw_path, spec_path)
        return df


if __name__ == "__main__":
    print("ArabicWikipediaScraper ready. Run via notebooks/01_scraping_and_dataset.ipynb.")
