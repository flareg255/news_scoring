import asyncio
import time
import traceback
import urllib.error
import urllib.request
import urllib.robotparser
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

from src.storage.db_manager import DbManager
from src.logger import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")

# 問い合わせ先を含む識別可能なUser-Agent。
# 匿名でアクセスするより、相手が問題を認識した際に連絡できる状態にしておく。
USER_AGENT = "news_scoring-bot/1.0 (+personal research; emotion scoring; contact: flare.g255@gmail.com)"

# robots.txt が Crawl-delay を宣言していない場合に用いる間隔（秒）
DEFAULT_INTERVAL = 0.5

# Crawl-delay の上限。極端な値で1回の実行が終わらなくなるのを防ぐ
MAX_CRAWL_DELAY = 120.0


class ArticleCrawler:
    """DBから未クロール記事を取得し、crawl4aiで本文を取得してMarkdown保存するクラス"""

    def __init__(self, db: DbManager = None, interval: float = DEFAULT_INTERVAL):
        """
        Args:
            db: DbManagerインスタンス（Noneの場合はデフォルト設定で生成）
            interval: Crawl-delay の宣言が無いドメインに適用するウェイト秒数
        """
        self.db = db or DbManager()
        self.interval = interval
        self._delays: dict[str, float] = {}      # ドメイン → 待機秒数
        self._last_access: dict[str, float] = {}  # ドメイン → 最終アクセス時刻
        RAW_DIR.mkdir(parents=True, exist_ok=True)

    def _delay_for(self, domain: str) -> float:
        """
        ドメインごとの待機秒数を返す。robots.txt が Crawl-delay を宣言していれば従う。

        robots.txt に法的拘束力は無いが、Crawl-delay は相手が想定する負荷水準の
        表明なので尊重する（例: gigazine.net は100秒を宣言している）。
        """
        if domain in self._delays:
            return self._delays[domain]

        delay = self.interval
        try:
            # RobotFileParser.read() は urllib 既定のUAを使い、これを拒否するサイトがある。
            # その場合 read() は例外を出さずパーサが空のままになり、Crawl-delay を
            # 取りこぼす。自前で取得して parse() に渡す。
            req = urllib.request.Request(
                f"https://{domain}/robots.txt", headers={"User-Agent": USER_AGENT}
            )
            with urllib.request.urlopen(req, timeout=15) as res:
                body = res.read().decode("utf-8", errors="replace")

            rp = urllib.robotparser.RobotFileParser()
            rp.parse(body.splitlines())
            declared = rp.crawl_delay(USER_AGENT) or rp.crawl_delay("*")
            if declared:
                delay = min(float(declared), MAX_CRAWL_DELAY)
                logger.info(
                    f"[Crawler] {domain}: Crawl-delay {declared}秒 の宣言あり → {delay}秒 を適用"
                )
        except (urllib.error.URLError, OSError, ValueError) as e:
            logger.warning(
                f"[Crawler] {domain}: robots.txt を取得できず既定値 {delay}秒 を使用 ({type(e).__name__})"
            )

        self._delays[domain] = delay
        return delay

    async def _wait_for(self, domain: str) -> None:
        """同一ドメインへの前回アクセスから所定の間隔が空くまで待つ"""
        delay = self._delay_for(domain)
        last = self._last_access.get(domain)
        if last is not None:
            remain = delay - (time.monotonic() - last)
            if remain > 0:
                await asyncio.sleep(remain)
        self._last_access[domain] = time.monotonic()

    @staticmethod
    def _interleave(articles: list) -> list:
        """
        ドメインが偏らないよう巡回順に並べ替える。

        待機はドメイン単位なので、同一ドメインを連続処理すると毎回待たされる。
        別ドメインを挟めばその時間を有効に使えるため、全体の所要時間が短くなる。
        """
        buckets: dict[str, deque] = defaultdict(deque)
        for a in articles:
            buckets[urlparse(a["url"]).netloc].append(a)

        ordered = []
        while buckets:
            for domain in list(buckets):
                ordered.append(buckets[domain].popleft())
                if not buckets[domain]:
                    del buckets[domain]
        return ordered

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

        articles = self._interleave(articles)
        logger.info(f"[Crawler] {len(articles)}件をクロール開始")
        success = 0

        browser_config = BrowserConfig(user_agent=USER_AGENT)
        async with AsyncWebCrawler(config=browser_config) as crawler:
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
                # 待機はドメイン単位。同一サイトへの連続アクセスだけを抑える
                await self._wait_for(urlparse(article["url"]).netloc)
                ok = await self._crawl_one(crawler, config, article)
                if ok:
                    success += 1

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
