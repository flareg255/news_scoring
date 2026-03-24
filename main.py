"""
ニューススコアリングシステム エントリーポイント

フェーズ0: データ収集（RSS取得 → クローリング → 保存）
フェーズ1: AIラベリング（感情スコア付与）
フェーズ2: LoRAファインチューニング
フェーズ3: RAG統合
"""

from src.logger import setup, get_logger
from src.rss.rss_fetcher import RssFetcher
from src.storage.db_manager import DbManager
from src.crawler.article_crawler import ArticleCrawler

setup()
logger = get_logger(__name__)


def phase0_collect():
    """フェーズ0: RSSからニュース記事を収集する"""
    logger.info("=== フェーズ0: データ収集開始 ===")

    # RSS取得
    fetcher = RssFetcher()
    articles = fetcher.fetch_all()

    # DBに保存（upsert: 既存URLはtitle等を更新）
    db = DbManager()
    inserted = db.save_articles(articles)

    # サマリー表示
    stats = db.stats()
    logger.info(f"取得: {len(articles)}件 / 新規保存: {inserted}件")
    logger.info(f"DB状況 → 合計: {stats['total']}件 | クローリング済: {stats['crawled']}件 | ラベリング済: {stats['labeled']}件")

    # クローリング（crawl4aiで本文取得 → data/raw/ にMarkdown保存）
    crawler = ArticleCrawler(db=db)
    crawler.crawl_all(limit=50)

    return articles


def phase1_label(articles):
    """フェーズ1: APIで感情スコアを付与する（1回限り）"""
    logger.info("=== フェーズ1: AIラベリング開始 ===")
    # TODO: Claude / Gemini API を呼び出して感情スコアを付与
    # TODO: スコア済みデータを data/labeled/ に保存
    pass


def phase2_train():
    """フェーズ2: ローカルLLMをLoRAでファインチューニングする"""
    logger.info("=== フェーズ2: LoRAファインチューニング開始 ===")
    # TODO: data/labeled/ のデータを使ってLoRAファインチューニング
    # TODO: 学習済みモデルを models/ に保存
    pass


def phase3_rag():
    """フェーズ3: RAGを使った感情文脈分析"""
    logger.info("=== フェーズ3: RAG統合開始 ===")
    # TODO: スコアリング済み記事をFAISS等のベクトルDBに格納
    # TODO: 新着記事に対して類似記事の感情推移を参照して分析
    pass


if __name__ == "__main__":
    articles = phase0_collect()
    # phase1_label(articles)
    # phase2_train()
    # phase3_rag()
