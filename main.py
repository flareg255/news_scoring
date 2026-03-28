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
from pathlib import Path
from src.cleaner.text_cleaner import clean_markdown_text

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
    uncrawled_count = stats['total'] - stats['crawled']
    if uncrawled_count > 0:
        crawler.crawl_all(limit=uncrawled_count)

    # アーカイブ処理（90日以上前のデータをZIP化して削除）
    archived_stats = db.archive_old_articles(retention_days=90)
    for month, count in archived_stats.items():
        if count > 0:
            logger.info(f"アーカイブ実行: {month} のデータを {count} 件 ZIP化しました")

    return articles


def phase1_label():
    """フェーズ1: APIで感情スコアを付与する前のデバッグ・クリーニング処理"""
    logger.info("=== フェーズ1: テキストのクリーニング検証（デバッグモード） ===")
    
    db = DbManager()
    unlabeled_rows = db.get_unlabeled(limit=30)  # テスト用に最新30件を取得
    
    if not unlabeled_rows:
        logger.info("ラベリング対象の記事がありません")
        return

    debug_dir = Path("data/debug_cleaned")
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for row in unlabeled_rows:
        content_path = row["content_path"]
        if not content_path or not Path(content_path).exists():
            continue
            
        with open(content_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
            
        # クリーニング実行
        cleaned_text = clean_markdown_text(raw_text)
        
        # デバッグフォルダに書き込み
        file_name = Path(content_path).name
        debug_path = debug_dir / file_name
        
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
            
        count += 1
        
    logger.info(f"{count} 件のクリーニング結果を {debug_dir} に出力しました。エディタの機能で差分を確認してください。")


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
    phase1_label()
    # phase2_train()
    # phase3_rag()

