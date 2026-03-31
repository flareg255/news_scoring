import re
from src.cleaner.cleaner_constants import CUTOFF_KEYWORDS, HEADER_CUTOFF_KEYWORDS


class TextCleaner:
    """
    ニュース本文（Markdown形式）から不要なメタデータ・メニュー・フッター・広告を除去し、
    LLMに渡すためのクリーンなテキストを生成するクラス。
    """

    MAX_CHARS = 2500  # LLMのContext Length上限エラーを防ぐための強制カット文字数

    def clean(self, raw_text: str) -> str:
        """
        生Markdownテキストをクリーニングして返す。

        Args:
            raw_text: クローラーが取得した生のMarkdownテキスト

        Returns:
            クリーニング済みのテキスト（最大MAX_CHARS文字）
        """
        lines = raw_text.split('\n')
        cleaned_lines = []
        seen_lines = set()

        # メタデータブロックの中かどうか判定フラグ
        in_header = True

        for line in lines:
            stripped = line.strip()

            # 1. クローラーが付与した上部のメタデータ（--- で囲まれた部分）を丸ごとスキップ
            if in_header:
                if stripped == "---":
                    in_header = False
                continue

            # 2. 足切りキーワードの判定
            if any(keyword in stripped for keyword in CUTOFF_KEYWORDS):
                break

            # 2.5 見出し（##）形式になっている「ランキング」「関連記事」は足切りラインとみなす
            if stripped.startswith("#"):
                if any(k in stripped for k in HEADER_CUTOFF_KEYWORDS):
                    break

            # 3. 画像タグの削除 ![alt](url) -> 完全削除
            line = re.sub(r'!\[.*?\]\(.*?\)', '', line)

            # 4. リンクタグのテキスト化 [text](url) -> text
            line = re.sub(r'\[([^\]]*)\]\([^\)]+\)', r'\1', line)

            # 5. HTMLタグの除去
            line = re.sub(r'<[^>]+>', '', line)

            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue

            # 5.5 長めの文の完全重複はノイズとしてスキップ
            if len(stripped) > 15:
                if stripped in seen_lines:
                    continue
                seen_lines.add(stripped)

            # 6. 短すぎるナビゲーション行のフィルタ
            if len(stripped) < 15 and all(c not in stripped for c in ["。", "、", "！", "？", "「", "」"]):
                if not stripped.startswith("#"):
                    continue

            cleaned_lines.append(line)

        # 連続する空行を1つにまとめる
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text[:self.MAX_CHARS]
