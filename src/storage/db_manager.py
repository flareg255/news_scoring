import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.rss.rss_fetcher import RssArticle


DB_PATH = Path("data/news_scoring.db")


class DbManager:
    """SQLiteデータベースの管理クラス"""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """テーブルを初期化する（初回実行時のみ作成）"""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    url          TEXT    UNIQUE NOT NULL,
                    title        TEXT    NOT NULL,
                    source       TEXT    NOT NULL,
                    published_at TEXT,
                    fetched_at   TEXT    NOT NULL,
                    is_crawled   INTEGER DEFAULT 0,
                    is_labeled   INTEGER DEFAULT 0,
                    content_path TEXT,
                    anger        REAL,
                    sadness      REAL,
                    joy          REAL
                )
            """)

    def save_articles(self, articles: list[RssArticle]) -> int:
        """
        記事一覧をDBに保存する（upsert）。
        URLが既存の場合はtitle/source/published_atを更新し、処理状態は維持する。

        Returns:
            新規追加された件数（更新はカウントしない）
        """
        now = datetime.now().isoformat()
        inserted = 0

        with self._connect() as conn:
            for article in articles:
                published = (
                    article.published_at.isoformat()
                    if article.published_at else None
                )
                before = conn.total_changes
                conn.execute(
                    """
                    INSERT INTO articles (url, title, source, published_at, fetched_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(url) DO UPDATE SET
                        title        = excluded.title,
                        source       = excluded.source,
                        published_at = excluded.published_at
                    """,
                    (article.url, article.title, article.source, published, now),
                )
                # total_changes が1増えた = 新規INSERT（更新は増えない）
                if conn.total_changes - before == 1:
                    inserted += 1

        return inserted

    def get_uncrawled(self, limit: int = 100) -> list[sqlite3.Row]:
        """本文未取得の記事を取得する"""
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM articles WHERE is_crawled = 0 LIMIT ?", (limit,)
            ).fetchall()

    def get_unlabeled(self, limit: int = 100) -> list[sqlite3.Row]:
        """感情スコア未付与の記事（本文取得完了済み）を取得する"""
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM articles WHERE is_crawled = 1 AND is_labeled = 0 LIMIT ?",
                (limit,),
            ).fetchall()

    def mark_crawled(self, article_id: int, content_path: str) -> None:
        """本文取得完了をマークする"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE articles SET is_crawled = 1, content_path = ? WHERE id = ?",
                (content_path, article_id),
            )

    def mark_labeled(
        self, article_id: int, anger: float, sadness: float, joy: float
    ) -> None:
        """感情スコア付与完了をマークする"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles
                SET is_labeled = 1, anger = ?, sadness = ?, joy = ?
                WHERE id = ?
                """,
                (anger, sadness, joy, article_id),
            )

    def stats(self) -> dict:
        """各フェーズの処理件数サマリーを返す"""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
            crawled = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE is_crawled = 1"
            ).fetchone()[0]
            labeled = conn.execute(
                "SELECT COUNT(*) FROM articles WHERE is_labeled = 1"
            ).fetchone()[0]
        return {"total": total, "crawled": crawled, "labeled": labeled}
