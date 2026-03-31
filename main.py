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
from src.cleaner.text_cleaner import TextCleaner
from src.labeling.llm_labeler import LlmLabeler
from src.labeling.labeling_constants import LMSTUDIO_API_URL
from src.training.export_dataset import DatasetExporter
from src.training.training_constants import TRAIN_JSONL_PATH

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
    logger.info(f"エンドポイント: {LMSTUDIO_API_URL} | モデル名は最初のAPIレスポンスで確認")
    
    db = DbManager()
    cleaner = TextCleaner()
    labeler = LlmLabeler()
    unlabeled_rows = db.get_unlabeled(limit=None)  # 制限を完全に解除し、本当にすべてを取得
    
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
        cleaned_text = cleaner.clean(raw_text)
        
        # デバッグ・品質確認用に、最終的にLLMに投げるテキストを保存
        debug_file_path = debug_dir / f"{row['id']}.md"
        with open(debug_file_path, "w", encoding="utf-8") as df:
            df.write(cleaned_text)
        
        # 2. LM Studio 定数設定のモデルを呼び出し、感情スコアを取得
        scores, raw_response = labeler.label(cleaned_text)
        
        # AIの生出力（根拠・無振りの山り小）をデバッグ用に保存
        if raw_response:
            reason_path = debug_dir / f"{row['id']}_reason.txt"
            with open(reason_path, "w", encoding="utf-8") as rf:
                rf.write(raw_response)
        
        if scores is not None:
            # 3. 取得したスコアをDBに保存し、is_labeled を 1 に更新
            db.mark_labeled(
                row["id"], 
                scores["anger"], scores["sadness"], scores["joy"],
                scores["fear"], scores["disgust"], scores["surprise"]
            )
            count += 1
            logger.info(f"ラベリング成功 [ID:{row['id']}] 喜:{scores['joy']} 怒:{scores['anger']} 悲:{scores['sadness']} 恐:{scores['fear']} 嫌:{scores['disgust']} 驚:{scores['surprise']}")
        else:
            logger.warning(f"ラベリング失敗 [ID:{row['id']}]")
            
    logger.info(f"Phase 1 完了: {count} 件の記事に感情スコアを付与しました！")
    stats = db.stats()
    logger.info(f"DB現在状況 → クロール済: {stats['crawled']}件 | ラベリング済: {stats['labeled']}件 | 未ラベリング: {stats['crawled'] - stats['labeled']}件")


def phase2_train():
    """フェーズ2: 教師データをエクスポートし、LM StudioでLoRAファインチューニングを行う"""
    logger.info("=== フェーズ2: 教師データエクスポート開始 ===")
    exporter = DatasetExporter()
    count = exporter.export()
    logger.info(f"エクスポート完了: {count} 件 → {TRAIN_JSONL_PATH}")
    logger.info("【次のステップ】LM Studio (Windows) の Train タブで以下を設定して学習を実行してください:")
    logger.info(f"  - Dataset: {TRAIN_JSONL_PATH.resolve()}")
    logger.info("  - Base Model: mistral-7b-instruct (Apache 2.0)")
    logger.info("  - Epochs: 3, Learning Rate: 2e-4, LoRA Rank: 16")


def phase3_rag():
    """フェーズ3: RAGを使った感情文脈分析"""
    logger.info("=== フェーズ3: RAG統合開始 ===")
    # TODO: スコアリング済み記事をFAISS等のベクトルDBに格納
    # TODO: 新着記事に対して類似記事の感情推移を参照して分析
    pass


if __name__ == "__main__":
    articles = phase0_collect()
    phase1_label()
    phase2_train()
    # phase3_rag()

