"""
モデル比較スクリプト

LM Studioは1モデルしか同時にロードできないため、
モデルごとに別々に実行してJSONに保存し、両方揃ったらレポートを生成する。

--- 使い方 ---
# Step 1: Mistral NeMo をLM Studioでロードした状態で実行
python compare_models.py --model mistral-nemo-instruct-2407

# Step 2: Gemma 4 をLM Studioでロードした状態で実行
python compare_models.py --model gemma-4-e4b-it

# Step 3: 両方揃ったらレポートを生成
python compare_models.py --report
"""

import json
import random
import argparse
from pathlib import Path
from datetime import datetime

from src.logger import setup, get_logger
from src.labeling.llm_labeler import LlmLabeler
from src.labeling.labeling_constants import LMSTUDIO_API_URL

setup()
logger = get_logger(__name__)

# ===== 設定 =====
SAMPLE_SIZE = 10
DEBUG_CLEANED_DIR = Path("data/debug_cleaned")
REPORT_DIR = Path("reports")
SAMPLE_IDS_FILE = REPORT_DIR / "compare_sample_ids.json"
EMOTIONS = ["joy", "anger", "sadness", "fear", "disgust", "surprise"]
EMOTION_JP = {"joy": "喜", "anger": "怒", "sadness": "悲", "fear": "恐", "disgust": "嫌", "surprise": "驚"}


def pick_and_save_sample(n: int) -> list[int]:
    """debug_cleaned からサンプルIDを選んでJSONに保存する"""
    md_files = [
        p for p in DEBUG_CLEANED_DIR.glob("*.md")
        if not p.stem.endswith("_reason")
    ]
    sample = random.sample(md_files, min(n, len(md_files)))
    ids = sorted(int(p.stem) for p in sample)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_IDS_FILE.write_text(json.dumps(ids), encoding="utf-8")
    logger.info(f"サンプルIDs: {ids}")
    return ids


def load_sample_ids() -> list[int]:
    """保存済みのサンプルIDを読み込む"""
    if not SAMPLE_IDS_FILE.exists():
        return pick_and_save_sample(SAMPLE_SIZE)
    return json.loads(SAMPLE_IDS_FILE.read_text(encoding="utf-8"))


def score_model(model_name: str):
    """現在LM Studioでロードされているモデルで採点してJSONに保存する"""
    ids = load_sample_ids()
    labeler = LlmLabeler(api_url=LMSTUDIO_API_URL, model_name=model_name)

    results = {}
    for article_id in ids:
        text_path = DEBUG_CLEANED_DIR / f"{article_id}.md"
        if not text_path.exists():
            logger.warning(f"ID:{article_id} のファイルが見つかりません")
            continue

        text = text_path.read_text(encoding="utf-8").strip()
        scores, _ = labeler.label(text)
        results[str(article_id)] = scores

        if scores:
            summary = " ".join(f"{EMOTION_JP[e]}:{int(scores[e])}" for e in EMOTIONS)
            logger.info(f"  ID:{article_id} [{summary}]")
        else:
            logger.warning(f"  ID:{article_id} → 失敗")

    out_path = REPORT_DIR / f"scores_{model_name}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"採点結果を保存: {out_path}")


def generate_report():
    """保存済みの採点結果を比較してMarkdownレポートを生成する"""
    score_files = list(REPORT_DIR.glob("scores_*.json"))
    if len(score_files) < 2:
        print(f"レポート生成には2つ以上のスコアファイルが必要です。現在: {[f.name for f in score_files]}")
        return

    ids = load_sample_ids()
    all_scores = {}
    model_names = []
    for f in score_files:
        model = f.stem.replace("scores_", "")
        model_names.append(model)
        all_scores[model] = json.loads(f.read_text(encoding="utf-8"))

    model_a, model_b = model_names[0], model_names[1]
    lines = [
        "# モデル比較レポート",
        f"\n生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n比較: `{model_a}` vs `{model_b}`",
        f"\nサンプル数: {len(ids)} 件\n",
        "---\n",
    ]

    total_diffs = []
    for article_id in ids:
        key = str(article_id)
        sa = all_scores[model_a].get(key)
        sb = all_scores[model_b].get(key)

        lines.append(f"## ID:{article_id}")
        lines.append(f"| 感情 | {model_a} | {model_b} | 差 |")
        lines.append("|---|---|---|---|")

        if sa and sb:
            diffs = []
            for e in EMOTIONS:
                va = int(sa.get(e, 0))
                vb = int(sb.get(e, 0))
                diff = vb - va
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                lines.append(f"| {EMOTION_JP[e]}({e}) | {va} | {vb} | {diff_str} |")
                diffs.append(abs(diff))
            avg = sum(diffs) / len(diffs)
            total_diffs.append(avg)
            lines.append(f"\n平均乖離: **{avg:.1f}**\n")
        else:
            lines.append(f"| （失敗） | {'✅' if sa else '❌'} | {'✅' if sb else '❌'} | - |\n")

    if total_diffs:
        lines.append("---")
        lines.append(f"\n## 総合平均乖離: **{sum(total_diffs)/len(total_diffs):.2f}**")
        lines.append("\n> 乖離が大きいほど2モデルの採点傾向が異なることを示す")

    report_path = REPORT_DIR / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ レポートを出力しました: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデル比較ツール")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", type=str, help="採点するモデル名（LM Studioでロード済みであること）")
    group.add_argument("--report", action="store_true", help="保存済みスコアからレポートを生成する")
    args = parser.parse_args()

    if args.model:
        logger.info(f"=== モデル採点: {args.model} ===")
        score_model(args.model)
    elif args.report:
        generate_report()
