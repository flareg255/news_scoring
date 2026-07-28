"""
フェーズ2: 教師データエクスポートスクリプト

SQLiteのラベリング済み記事とその本文を組み合わせ、
LM Studio の Fine-tuning 機能が受け付ける JSONL 形式に変換して出力する。

本文は data/raw の実体ファイル、無ければ月次アーカイブZIPから読み出し、
ラベリング時と同じ TextCleaner を通す。これによりラベル付与に使われた
テキストと、学習に渡すテキストが一致する。
"""

import json
from src.cleaner.text_cleaner import TextCleaner
from src.storage.db_manager import DbManager
from src.training.training_constants import (
    TRAIN_JSONL_PATH,
    PROMPT_TEMPLATE,
)
from src.logger import get_logger

logger = get_logger(__name__)


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

        with open(TRAIN_JSONL_PATH, "w", encoding="utf-8") as f:
            for row in rows:
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
        if skipped > 0:
            logger.warning(f"本文を取得できずスキップ: {skipped} 件")

        return exported
