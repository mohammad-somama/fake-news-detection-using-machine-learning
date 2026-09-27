from __future__ import annotations

from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd


# NEWS RSS SOURCES

NEWS_FEEDS = {
    "times_of_india": {
        "name": "Times of India",
        "feeds": [
            (
                "Top Stories",
                "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
            ),
            (
                "Most Recent",
                "https://timesofindia.indiatimes.com/rssfeedmostrecent.cms",
            ),
        ],
    },

    "indian_express": {
        "name": "Indian Express",
        "feeds": [
            (
                "Latest",
                "https://indianexpress.com/feed/",
            ),
            (
                "India",
                "https://indianexpress.com/section/india/feed/",
            ),
        ],
    },

    "ndtv": {
        "name": "NDTV",
        "feeds": [
            (
                "Top Stories",
                "https://feeds.feedburner.com/ndtvnews-top-stories",
            ),
            (
                "India",
                "https://feeds.feedburner.com/ndtvnews-india-news",
            ),
            (
                "World",
                "https://feeds.feedburner.com/ndtvnews-world-news",
            ),
        ],
    },

    
    # BBC

    "bbc": {
        "name": "BBC",
        "feeds": [
            (
                "Top Stories",
                "https://feeds.bbci.co.uk/news/rss.xml",
            ),
            (
                "World",
                "https://feeds.bbci.co.uk/news/world/rss.xml",
            ),
            (
                "Technology",
                "https://feeds.bbci.co.uk/news/technology/rss.xml",
            ),
            (
                "Business",
                "https://feeds.bbci.co.uk/news/business/rss.xml",
            ),
            (
                "Asia",
                "https://feeds.bbci.co.uk/news/world/asia/rss.xml",
            ),
        ],
    },

    
    # BBC HINDI
    
    "bbc_hindi": {
    "name": "BBC Hindi",
    "feeds": [
        (
            "Latest",
            "https://feeds.bbci.co.uk/hindi/rss.xml",
        ),
    ],
},
            #"india_today": {
            #   "name": "India Today",
            #  "feeds": [
            #     (
                #        "Latest",
                #       "https://www.indiatoday.in/rss/home",
                #  ),
            # ],
            #},
}


# ERROR DATA CLASS

@dataclass(frozen=True)
class ScrapeError:
    source: str
    feed: str
    url: str
    message: str



# HTML TEXT CLEANER

class _HTMLTextExtractor(HTMLParser):

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        cleaned = data.strip()

        if cleaned:
            self.parts.append(cleaned)

    def text(self) -> str:
        return " ".join(self.parts)



# AVAILABLE SOURCES


def available_source_labels() -> dict[str, str]:
    """
    Returns:
        {
            "times_of_india": "Times of India",
            "indian_express": "Indian Express",
            "ndtv": "NDTV"
        }
    """

    return {
        source_id: config["name"]
        for source_id, config in NEWS_FEEDS.items()
    }



# MAIN NEWS FETCH FUNCTION


def fetch_live_news(
    source_ids: Iterable[str] | None = None,
    limit_per_feed: int = 8,
    timeout: int = 8,
) -> tuple[pd.DataFrame, list[ScrapeError]]:

    selected_sources = list(
        source_ids or NEWS_FEEDS.keys()
    )

    rows: list[dict[str, str]] = []
    errors: list[ScrapeError] = []

    # Safety check
    if limit_per_feed < 1:
        limit_per_feed = 1

    if timeout < 1:
        timeout = 1

    
    # Loop through selected sources
    

    for source_id in selected_sources:

        if source_id not in NEWS_FEEDS:

            errors.append(
                ScrapeError(
                    source=source_id,
                    feed="",
                    url="",
                    message="Unknown source",
                )
            )

            continue

        source_config = NEWS_FEEDS[source_id]

        
        # Loop through feeds of each source
        

        for feed_name, feed_url in source_config["feeds"]:

            try:

                feed_rows = _fetch_feed(
                    source=source_config["name"],
                    feed_name=feed_name,
                    feed_url=feed_url,
                    limit=limit_per_feed,
                    timeout=timeout,
                )

                rows.extend(feed_rows)

            except (
                HTTPError,
                URLError,
                ET.ParseError,
                TimeoutError,
                OSError,
            ) as exc:

                errors.append(
                    ScrapeError(
                        source=source_config["name"],
                        feed=feed_name,
                        url=feed_url,
                        message=str(exc),
                    )
                )

            except Exception as exc:

                # Prevent one unexpected feed error
                # from crashing the complete application.
                errors.append(
                    ScrapeError(
                        source=source_config["name"],
                        feed=feed_name,
                        url=feed_url,
                        message=f"Unexpected error: {exc}",
                    )
                )

    
    # CREATE DATAFRAME
    

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:
        return _empty_dataframe(), errors

    
    # Remove duplicate links
    

    if "link" in dataframe.columns:

        dataframe = dataframe.drop_duplicates(
            subset=["link"],
            keep="first",
        )

    
    # Remove duplicate source + title
    

    if "source" in dataframe.columns and "title" in dataframe.columns:

        dataframe = dataframe.drop_duplicates(
            subset=["source", "title"],
            keep="first",
        )

    return (
        dataframe.reset_index(drop=True),
        errors,
    )



# FETCH ONE RSS FEED


def _fetch_feed(
    source: str,
    feed_name: str,
    feed_url: str,
    limit: int,
    timeout: int,
) -> list[dict[str, str]]:

    
    # HTTP REQUEST
    

    request = Request(
        feed_url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/140.0 Safari/537.36"
            ),
            "Accept": (
                "application/rss+xml, "
                "application/xml, "
                "text/xml, "
                "*/*;q=0.8"
            ),
        },
    )

    
    # DOWNLOAD RSS
    

    with urlopen(
        request,
        timeout=timeout,
    ) as response:

        charset = (
            response.headers.get_content_charset()
            or "utf-8"
        )

        content = response.read().decode(
            charset,
            errors="replace",
        )

    
    # PARSE XML


    root = ET.fromstring(content)

    # Standard RSS
    items = root.findall(".//item")

    # Atom fallback
    if not items:

        items = root.findall(
            ".//{http://www.w3.org/2005/Atom}entry"
        )

    rows: list[dict[str, str]] = []

    
    # PROCESS ITEMS


    for item in items[:limit]:

        title = _clean_text(
            _child_text(
                item,
                ("title",),
            )
        )

        summary = _clean_text(
            _child_text(
                item,
                (
                    "description",
                    "summary",
                    "encoded",
                    "content",
                ),
            )
        )

        link = _item_link(item)

        published = _clean_text(
            _child_text(
                item,
                (
                    "pubdate",
                    "published",
                    "updated",
                ),
            )
        )

        
        # Ignore completely empty item
        

        if not title and not summary:
            continue

        
        # Combine title + summary
        

        text_parts = []

        if title:
            text_parts.append(title)

        if summary:
            text_parts.append(summary)

        text = ". ".join(text_parts)


        # Add row
        

        rows.append(
            {
                "source": source,
                "feed": feed_name,
                "title": title,
                "summary": summary,
                "text": text,
                "link": link,
                "published": published,
            }
        )

    return rows



# GET CHILD TEXT


def _child_text(
    element: ET.Element,
    names: Iterable[str],
) -> str:

    wanted = {
        name.lower()
        for name in names
    }

    for child in element:

        local_name = (
            child.tag
            .rsplit("}", 1)[-1]
            .lower()
        )

        if local_name in wanted:

            return child.text or ""

    return ""



# GET ARTICLE LINK


def _item_link(
    element: ET.Element,
) -> str:

    for child in element:

        local_name = (
            child.tag
            .rsplit("}", 1)[-1]
            .lower()
        )

        if local_name != "link":
            continue

        # Atom style
        href = child.attrib.get(
            "href",
            "",
        ).strip()

        # RSS style
        text = (
            child.text or ""
        ).strip()

        return href or text

    # GUID fallback
    return _child_text(
        element,
        ("guid",),
    ).strip()



# CLEAN HTML


def _clean_text(
    value: str,
) -> str:

    parser = _HTMLTextExtractor()

    parser.feed(
        unescape(
            value or ""
        )
    )

    parser.close()

    cleaned = (
        parser.text()
        or unescape(value or "")
    )

    return " ".join(
        cleaned.split()
    )



# EMPTY DATAFRAME


def _empty_dataframe() -> pd.DataFrame:

    return pd.DataFrame(
        columns=[
            "source",
            "feed",
            "title",
            "summary",
            "text",
            "link",
            "published",
        ]
    )
