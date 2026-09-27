import argparse
from pathlib import Path

from src.news_scraper import (
    NEWS_FEEDS,
    fetch_live_news,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect public RSS news data for the fake news detection project."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(NEWS_FEEDS.keys()),
        default=sorted(NEWS_FEEDS.keys()),
        help="News sources to collect.",
    )
    parser.add_argument(
        "--limit-per-feed",
        type=int,
        default=20,
        help="Maximum items to collect from each RSS feed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=8,
        help="Seconds to wait for each RSS feed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataframe, errors = fetch_live_news(
        source_ids=args.sources,
        limit_per_feed=args.limit_per_feed,
        timeout=args.timeout,
    )

    print(f"Collected rows: {len(dataframe)}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(args.output, index=False, encoding="utf-8")
        print(f"Output: {args.output}")

    if not dataframe.empty:
        print(dataframe[["source", "title", "published"]].head(10).to_string(index=False))

    if errors:
        print("Feed errors:")
        for error in errors:
            print(f"- {error.source} / {error.feed}: {error.message}")


if __name__ == "__main__":
    main()
