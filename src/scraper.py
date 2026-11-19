"""Reusable scraping script for Arabic text dataset collection."""

from __future__ import annotations

from pathlib import Path


class Scraper:
    """Base scraper placeholder.

    Extend this class with website-specific scraping logic.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the scraping pipeline.

        Replace this method with actual scraping implementation.
        """
        raise NotImplementedError("Implement scraping logic in Scraper.run().")


if __name__ == "__main__":
    print("Scraper module ready. Implement Scraper.run() to start scraping.")
