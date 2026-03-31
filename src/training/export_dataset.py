"""
フェーズ2: 教師データエクスポートスクリプト

SQLiteのラベリング済み記事と data/debug_cleaned/ のテキストを組み合わせ、
LM Studio の Fine-tuning 機能が受け付ける JSONL 形式に変換して出力する。
"""

import json
from src.storage.db_manager import DbManager
from src.training.training_constants import (
    TRAIN_JSONL_PATH,
    DEBUG_CLEANED_DIR,
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

        conn = db._connect()
        rows = conn.execute(
            "SELECT id, joy, anger, sadness, fear, disgust, surprise "
            "FROM articles WHERE is_labeled = 1"
        ).fetchall()
        conn.close()

        logger.info(f"ラベリング済み記事数: {len(rows)} 件")

        TRAIN_JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0

        with open(TRAIN_JSONL_PATH, "w", encoding="utf-8") as f:
            for row in rows:
                article_id = row["id"]
                cleaned_path = DEBUG_CLEANED_DIR / f"{article_id}.md"

                if not cleaned_path.exists():
                    skipped += 1
                    continue

                text = cleaned_path.read_text(encoding="utf-8").strip()
                if not text:
                    skipped += 1
                    continue

                prompt = PROMPT_TEMPLATE.format(text=text)

                completion = json.dumps({
                    "joy":      row["joy"],
                    "anger":    row["anger"],
                    "sadness":  row["sadness"],
                    "fear":     row["fear"],
                    "disgust":  row["disgust"],
                    "surprise": row["surprise"],
                }, ensure_ascii=False)

                record = {"prompt": prompt, "completion": completion}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1

        logger.info(f"データセットエクスポート完了: {exported} 件 → {TRAIN_JSONL_PATH}")
        if skipped > 0:
            logger.warning(f"debug_cleanedファイルが存在しないためスキップ: {skipped} 件")

        return exported
