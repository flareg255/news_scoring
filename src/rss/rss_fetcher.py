import feedparser
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.rss.rss_constants import RssConstants
from src.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RssArticle:
    """RSS記事の1件分のデータ"""
    title: str
    url: str
    source: str
    published_at: Optional[datetime]


class RssFetcher:
    """RSSフィードから記事一覧を取得するクラス"""

    def fetch(self, feed_url: str) -> list[RssArticle]:
        """
        指定したRSSフィードURLから記事一覧を取得する。

        Args:
            feed_url: RSSフィードのURL

        Returns:
            RssArticleのリスト
        """
        feed = feedparser.parse(feed_url)
        articles = []

        for entry in feed.entries:
            published_at = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                published_at = datetime(*entry.published_parsed[:6])

            articles.append(RssArticle(
                title=entry.get("title", ""),
                url=entry.get("link", ""),
                source=feed.feed.get("title", feed_url),
                published_at=published_at,
            ))

        return articles

    def fetch_all(self) -> list[RssArticle]:
        """
        定義済みの全フィードから記事一覧をまとめて取得する。

        Returns:
            全ソースのRssArticleを結合したリスト
        """
        all_articles = []
        for feed_url in RssConstants.ALL_FEEDS:
            try:
                articles = self.fetch(feed_url)
                all_articles.extend(articles)
                logger.info(f"[OK] {feed_url} -> {len(articles)}件取得")
            except Exception as e:
                logger.error(f"[ERROR] {feed_url} -> {e}")

        return all_articles
