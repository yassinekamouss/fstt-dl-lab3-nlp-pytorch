"""Local verification unit tests runnable without GPU/PyTorch/Transformers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.metrics import compute_rouge_and_diversity
from src.preprocessing import ArabicTextPreprocessor, discretize_scores
from src.scraper import (
    ArabicWikipediaScraper,
    arabic_char_ratio,
    clean_wiki_paragraph,
)

SAMPLE_WIKI_HTML = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<body>
  <div id="mw-content-text">
    <div class="mw-parser-output">
      <div class="infobox"><table><tr><td>معلومات جانبية يجب تجاهلها</td></tr></table></div>
      <div class="hatnote">انظر أيضاً: صفحة أخرى</div>
      <p>قصير جدا.</p>
      <p>
        يعد الذكاء الاصطناعي من أهم فروع علوم الحاسوب الحديثة التي تهتم ببناء أنظمة ذكية قادرة على التعلم والاستدلال واتخاذ القرارات المعقدة بطريقة تحاكي القدرات الإدراكية البشرية في معالجة البيانات الضخمة وتحليل الأنماط المتنوعة عبر الشبكات العصبية العميقة وخوارزميات التعلم الآلي المتقدمة في تطبيقات متعددة.<sup class="reference"><a href="#cite_note-1">[1]</a></sup>
        <span class="mw-editsection">[عدل]</span>
      </p>
      <p>
        تعتمد <a href="/wiki/%D8%AA%D8%B9%D9%84%D9%85_%D8%A7%D9%84%D8%A2%D9%84%D8%A9">تعلم الآلة</a>
        على نماذج رياضية وإحصائية متطورة لاستخراج المعرفة من البيانات وتصنيف النصوص العربية بدقة عالية وتوليد فقرات متماسكة في مجالات علم البيانات والروبوتية والرؤية الحاسوبية ومعالجة اللغات الطبيعية الحديثة.<sup class="reference">[2]</sup>
        <a href="/wiki/%D9%85%D9%84%D9%81:Example.png">صورة</a>
        <a href="/wiki/%D8%AA%D8%B5%D9%86%D9%8A%D9%81:%D8%B0%D9%83%D8%A7%D8%A1">تصنيف</a>
      </p>
      <p>English only paragraph that should be filtered out completely because it has no Arabic words at all in this sample text block and exceeds twenty five words to verify ratio filtering properly works.</p>
    </div>
  </div>
</body>
</html>
"""


class TestConfigAndModules(unittest.TestCase):
    def test_config_yaml_loads_and_has_required_sections(self) -> None:
        cfg_path = REPO_ROOT / "configs" / "config.yaml"
        self.assertTrue(cfg_path.exists())
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        for key in ("project", "scraping", "preprocessing", "part1_models", "part2_gpt2"):
            self.assertIn(key, cfg)
        seeds = cfg["scraping"]["seed_titles"]
        for tier in ("core", "related", "off_topic"):
            self.assertGreaterEqual(len(seeds[tier]), 15)

    def test_arabic_wikipedia_scraper_parsing(self) -> None:
        scraper = ArabicWikipediaScraper(
            output_dir=REPO_ROOT / "data" / "raw",
            config={"scraping": {"min_words": 20, "max_words": 250, "min_arabic_ratio": 0.70}},
        )
        paragraphs, links = scraper.parse_paragraphs(SAMPLE_WIKI_HTML)

        self.assertEqual(len(paragraphs), 2)
        for p in paragraphs:
            self.assertNotIn("[1]", p)
            self.assertNotIn("[عدل]", p)
            self.assertNotIn("معلومات جانبية", p)
            self.assertGreaterEqual(arabic_char_ratio(p), 0.70)

        self.assertIn("تعلم الآلة", links)
        self.assertFalse(any(":" in title for title in links))

        url = scraper.build_article_url("الذكاء الاصطناعي")
        self.assertTrue(url.startswith("https://ar.wikipedia.org/wiki/"))

        cleaned = clean_wiki_paragraph("نص عربي [12] مع\u200f مسافات   زائدة")
        self.assertEqual(cleaned, "نص عربي مع مسافات زائدة")

    def test_arabic_preprocessing_and_discretization(self) -> None:
        prep = ArabicTextPreprocessor()
        raw = "الذَّكَاءُ الإصْطِنَاعِيُّ في المُسْتَقْبَلِ https://example.com 123 — وَالتَّعَلُّمُ الآلِيُّ!"
        norm = prep.normalize_arabic(raw)

        self.assertNotIn("https", norm)
        self.assertNotIn("123", norm)
        self.assertNotIn("َ", norm)
        self.assertIn("الذكاء", norm)
        self.assertIn("الاصطناعي", norm)
        self.assertIn("الالي", norm)

        tokens = prep.tokenize(norm)
        self.assertIn("الذكاء", tokens)
        filtered = prep.remove_stopwords(tokens)
        self.assertNotIn("في", filtered)
        self.assertIn("الذكاء", filtered)

        labels, names = discretize_scores(
            [0.0, 3.9, 4.0, 6.9, 7.0, 10.0],
            bins=(0.0, 4.0, 7.0, 10.01),
            class_names=["Faible", "Moyenne", "Elevee"],
        )
        self.assertEqual(labels.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(names, ["Faible", "Faible", "Moyenne", "Moyenne", "Elevee", "Elevee"])

    def test_arabic_aware_rouge_and_diversity(self) -> None:
        refs = ["يعتمد الذكاء الاصطناعي على الشبكات العصبية العميقة وتعلم الآلة."]
        hyps = ["يعتمد الذكاء الاصطناعي على تعلم الآلة والشبكات العصبية."]
        scores = compute_rouge_and_diversity(refs, hyps)

        self.assertGreater(scores["rouge1"], 0.6)
        self.assertGreater(scores["rouge2"], 0.2)
        self.assertGreater(scores["rougeL"], 0.5)
        self.assertGreater(scores["distinct_1"], 0.5)
        self.assertGreater(scores["distinct_2"], 0.5)


if __name__ == "__main__":
    unittest.main()
