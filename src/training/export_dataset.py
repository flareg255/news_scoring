"""
フェーズ2: 教師データエクスポートスクリプト

SQLiteのラベリング済み記事とその本文を組み合わせ、
LM Studio の Fine-tuning 機能が受け付ける JSONL 形式に変換して出力する。

本文は data/raw の実体ファイル、無ければ月次アーカイブZIPから読み出し、
ラベリング時と同じ TextCleaner を通す。これによりラベル付与に使われた
テキストと、学習に渡すテキストが一致する。
"""

import json
from urllib.parse import urlparse

from src.cleaner.text_cleaner import TextCleaner
from src.storage.db_manager import DbManager
from src.training.training_constants import (
    TRAIN_JSONL_PATH,
    PROMPT_TEMPLATE,
)
from src.logger import get_logger

logger = get_logger(__name__)

# 学習利用を利用規約で禁じている媒体。DBには記事が残っていても教師データには含めない。
#
# www.asahi.com: 朝日新聞デジタル利用規約 第10条(7)
#   「データマイニング、ロボット等によるデータの収集、抽出、解析または蓄積等をする行為、
#     並びにAI（機械学習モデルを含む）の開発・学習・改善・トレーニングその他これらに
#     類する目的のために利用する行為」を、事前の書面による許可なく行うことを禁止。
#
# 判断の経緯は docs/DATA_COLLECTION_POLICY.md
EXCLUDED_DOMAINS = frozenset({"www.asahi.com", "asahi.com", "digital.asahi.com"})


class DatasetExporter:
    """
    ラベリング済み記事をLM Studio Fine-tuning用のJSONL形式にエクスポートするクラス。
    """

    def export(self) -> int:
        """
        ラベリング済み記事を JSONL 形式でエクスポートする。

        Returns:
            エクスポートした件数
        """
        db = DbManager()
        cleaner = TextCleaner()

        conn = db._connect()
        rows = conn.execute(
            "SELECT * FROM articles WHERE is_labeled = 1"
        ).fetchall()
        conn.close()

        logger.info(f"ラベリング済み記事数: {len(rows)} 件")

        TRAIN_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        excluded = 0

        with open(TRAIN_JSONL_PATH, "w", encoding="utf-8") as f:
            for row in rows:
                if urlparse(row["url"] or "").netloc in EXCLUDED_DOMAINS:
                    excluded += 1
                    continue

                raw_text = db.read_article_text(row)
                if raw_text is None:
                    skipped += 1
                    continue

                text = cleaner.clean(raw_text).strip()
                if not text:
                    skipped += 1
                    continue

                prompt = PROMPT_TEMPLATE.format(text=text)

                # DBはREALで保持しているが、プロンプトは「0〜10の整数」を要求しており
                # モデルの実出力も整数なので、教師データも整数に揃える（7.0ではなく7）
                completion = json.dumps({
                    key: int(row[key] or 0)
                    for key in ("joy", "anger", "sadness", "fear", "disgust", "surprise")
                }, ensure_ascii=False)

                record = {"prompt": prompt, "completion": completion}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        logger.info(f"データセットエクスポート完了: {exported} 件 → {TRAIN_JSONL_PATH}")
        if excluded > 0:
            logger.info(f"学習利用を規約で禁じている媒体のため除外: {excluded} 件")
        if skipped > 0:
            logger.warning(f"本文を取得できずスキップ: {skipped} 件")

        return exported
