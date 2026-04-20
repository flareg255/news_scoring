import sqlite3
import os
import zipfile
from collections import defaultdict
from datetime import datetime, timedelta
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
                    joy          REAL,
                    fear         REAL,
                    disgust      REAL,
                    surprise     REAL
                )
            """)
            
            # 既存のデータベースを使っている場合のための自動マイグレーション
            try:
                conn.execute("ALTER TABLE articles ADD COLUMN fear REAL")
                conn.execute("ALTER TABLE articles ADD COLUMN disgust REAL")
                conn.execute("ALTER TABLE articles ADD COLUMN surprise REAL")
            except sqlite3.OperationalError:
                pass # カラムが既に存在する場合のエラーは無視する

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

    def get_unlabeled(self, limit: Optional[int] = 100) -> list[sqlite3.Row]:
        """感情スコア未付与の記事（本文取得完了済み）を取得する"""
        with self._connect() as conn:
            if limit is not None:
                return conn.execute(
                    "SELECT * FROM articles WHERE is_crawled = 1 AND is_labeled = 0 LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                return conn.execute(
                    "SELECT * FROM articles WHERE is_crawled = 1 AND is_labeled = 0"
                ).fetchall()

    def mark_crawled(self, article_id: int, content_path: str) -> None:
        """本文取得完了をマークする"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE articles SET is_crawled = 1, content_path = ? WHERE id = ?",
                (content_path, article_id),
            )

    def mark_labeled(
        self, article_id: int, anger: float, sadness: float, joy: float,
        fear: float, disgust: float, surprise: float
    ) -> None:
        """感情スコア付与完了をマークする"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles
                SET is_labeled = 1, anger = ?, sadness = ?, joy = ?, fear = ?, disgust = ?, surprise = ?
                WHERE id = ?
                """,
                (anger, sadness, joy, fear, disgust, surprise, article_id),
            )

    def reset_labels(self) -> int:
        """全記事のラベルをリセットする（再採点用）。スコアもNULLに戻す。"""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE articles
                SET is_labeled = 0, anger = NULL, sadness = NULL,
                    joy = NULL, fear = NULL, disgust = NULL, surprise = NULL
                WHERE is_labeled = 1
                """
            )
            count = conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        return count

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

    def archive_old_articles(self, retention_days: int = 90) -> dict:
        """
        指定日数（retention_days）以上経過した記事のMarkdownをZIP化し、実体ファイルを削除する。
        DBのレコードは残し、content_path のみ NULL にクリアする。
        """
        threshold_date = (datetime.now() - timedelta(days=retention_days)).isoformat()
        archive_dir = self.db_path.parent / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)

        archived_counts = {}

        with self._connect() as conn:
            # 削除・アーカイブ対象を取得
            rows = conn.execute(
                "SELECT id, fetched_at, content_path FROM articles WHERE fetched_at < ?",
                (threshold_date,)
            ).fetchall()

            if not rows:
                return archived_counts

            # 月ごとにグループ化 (fetched_at のYYYY-MM)
            by_month = defaultdict(list)
            for row in rows:
                if row["fetched_at"]:
                    month_str = row["fetched_at"][:7]  # "YYYY-MM"
                    by_month[month_str].append(row)

            for month_str, articles in by_month.items():
                zip_path = archive_dir / f"{month_str}.zip"
                
                # ZIPに追記
                with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
                    for article in articles:
                        cpath = article["content_path"]
                        if cpath and os.path.exists(cpath):
                            # ZIP内でのファイル名
                            arcname = os.path.basename(cpath)
                            # 既に同一ファイル名がZIPにある場合の重複対策は上書きまたは無視（通常ID単位でファイル名が一意なら問題ない）
                            if arcname not in zf.namelist():
                                zf.write(cpath, arcname)
                            try:
                                os.remove(cpath)
                            except OSError:
                                pass
                
                # DBのレコードは残し、不要になった実体ファイルへのパスのみクリアする
                ids = [a["id"] for a in articles]
                chunk_size = 900 # SQLiteのプレースホルダ上限を考慮
                for i in range(0, len(ids), chunk_size):
                    chunk = ids[i:i+chunk_size]
                    placeholders = ",".join("?" * len(chunk))
                    conn.execute(f"UPDATE articles SET content_path = NULL WHERE id IN ({placeholders})", chunk)
                
                archived_counts[month_str] = len(ids)

        return archived_counts
