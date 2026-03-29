import asyncio
import traceback
from datetime import datetime
from pathlib import Path

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from src.storage.db_manager import DbManager
from src.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")


class ArticleCrawler:
    """DBから未クロール記事を取得し、crawl4aiで本文を取得してMarkdown保存するクラス"""

    def __init__(self, db: DbManager = None, interval: float = 0.5):
        """
        Args:
            db: DbManagerインスタンス（Noneの場合はデフォルト設定で生成）
            interval: 1記事ごとのウェイト秒数（サーバー負荷軽減）
        """
        self.db = db or DbManager()
        self.interval = interval
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def crawl_all(self, limit: int = 50) -> int:
        """
        未クロール記事を最大 limit 件取得して本文をクロールする。

        Returns:
            クロールに成功した件数
        """
        return asyncio.run(self._crawl_all_async(limit))

    async def _crawl_all_async(self, limit: int) -> int:
        articles = self.db.get_uncrawled(limit=limit)
        if not articles:
            logger.info("[Crawler] 未クロール記事なし")
            return 0

        logger.info(f"[Crawler] {len(articles)}件をクロール開始")
        success = 0

        async with AsyncWebCrawler() as crawler:
            # ユーザー提案の最適化: クローラーの段階でヘッダー・フッター・ナビゲーション・iframe等を除外する
            config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                exclude_external_images=True,
                excluded_tags=['header', 'footer', 'nav', 'aside', 'form', 'iframe', 'title', 'meta'],
                markdown_generator=DefaultMarkdownGenerator(
                    content_filter=PruningContentFilter(
                        threshold=0.4, 
                        threshold_type="dynamic", 
                        min_word_threshold=50
                    )
                )
            )
            for article in articles:
                ok = await self._crawl_one(crawler, config, article)
                if ok:
                    success += 1
                await asyncio.sleep(self.interval)

        logger.info(f"[Crawler] 完了: {success}/{len(articles)}件成功")
        return success

    async def _crawl_one(self, crawler, config, article) -> bool:
        """
        1件の記事をクロールして data/raw/{id}.md に保存し、DBをマークする。

        Returns:
            成功した場合 True
        """
        article_id = article["id"]
        url = article["url"]
        title = article["title"]

        try:
            result = await crawler.arun(url=url, config=config)

            if not result.success:
                logger.warning(f"[Crawler] SKIP (取得失敗) id={article_id} {url}")
                return False

            # crawl4ai新旧API両対応:
            #   旧: result.markdown -> str
            #   新: result.markdown -> MarkdownGenerationResult (.raw_markdownで取得)
            md = result.markdown
            if hasattr(md, "raw_markdown"):
                md = md.raw_markdown
            md = str(md).strip() if md else ""

            if not md:
                logger.warning(f"[Crawler] SKIP (本文なし) id={article_id} {url}")
                return False

            # Markdownファイルを保存
            content_path = str(RAW_DIR / f"{article_id}.md")
            self._save_markdown(
                path=content_path,
                title=title,
                source=article["source"],
                url=url,
                published_at=article["published_at"],
                body=md,
            )

            # DBにクロール完了をマーク
            self.db.mark_crawled(article_id, content_path)
            logger.info(f"[Crawler] OK id={article_id} {url}")
            return True

        except Exception:
            logger.error(f"[Crawler] ERROR id={article_id} {url}")
            traceback.print_exc()
            return False

    def _save_markdown(
        self,
        path: str,
        title: str,
        source: str,
        url: str,
        published_at: str | None,
        body: str,
    ) -> None:
        header = (
            f"# {title}\n\n"
            f"source: {source}\n"
            f"url: {url}\n"
            f"published_at: {published_at or ''}\n"
            f"fetched_at: {datetime.now().isoformat()}\n\n"
            f"---\n\n"
        )
        Path(path).write_text(header + body, encoding="utf-8")
