import re
from src.cleaner.cleaner_constants import CUTOFF_KEYWORDS, HEADER_CUTOFF_KEYWORDS


class TextCleaner:
    """
    ニュース本文（Markdown形式）から不要なメタデータ・メニュー・フッター・広告を除去し、
    LLMに渡すためのクリーンなテキストを生成するクラス。
    """

    MAX_CHARS = 2500  # LLMのContext Length上限エラーを防ぐための強制カット文字数
    PROSE_LEN = 80    # この長さを超える行は本文の一文とみなし、足切りキーワードを無視する

    def clean(self, raw_text: str) -> str:
        """
        生Markdownテキストをクリーニングして返す。

        Args:
            raw_text: クローラーが取得した生のMarkdownテキスト

        Returns:
            クリーニング済みのテキスト（最大MAX_CHARS文字）
        """
        text = self._clean_lines(raw_text)

        # クリーニング結果が空になった場合は、足切り判定を行わない最小限の処理にフォールバックする
        # （画像クレジットや本文中の単語で誤爆しても、記事を丸ごと失わないための保険）
        if not text:
            text = self._clean_lines(raw_text, apply_cutoff=False)

        return text[:self.MAX_CHARS]

    def _clean_lines(self, raw_text: str, apply_cutoff: bool = True) -> str:
        lines = raw_text.split('\n')
        cleaned_lines = []
        seen_lines = set()

        # メタデータブロックの中かどうか判定フラグ
        in_header = True
        # 本文をまだ1行も拾えていない間は、足切りキーワードを「その行のスキップ」として扱う
        body_started = False

        for line in lines:
            stripped = line.strip()

            # 1. クローラーが付与した上部のメタデータ（--- で囲まれた部分）を丸ごとスキップ
            if in_header:
                if stripped == "---":
                    in_header = False
                continue

            # 2. 画像タグの削除 ![alt](url) -> 完全削除
            #    足切り判定より先に消す。画像クレジット（© ... All Rights Reserved）が
            #    alt textに埋まっているケースで本文が丸ごと切られるのを防ぐため。
            line = re.sub(r'!\[.*?\]\(.*?\)', '', line)

            # 3. リンクタグのテキスト化 [text](url) -> text
            line = re.sub(r'\[([^\]]*)\]\([^\)]+\)', r'\1', line)

            # 4. HTMLタグの除去
            line = re.sub(r'<[^>]+>', '', line)

            stripped = line.strip()

            if apply_cutoff:
                # 5. 足切りキーワードの判定
                #    長い行は本文の一文（例:「…利用規約に違反する…」）とみなして対象外にする。
                #    本文開始前にヒットした場合はフッターではなくクレジット行なので、その行だけ捨てる。
                if len(stripped) <= self.PROSE_LEN and any(k in stripped for k in CUTOFF_KEYWORDS):
                    if body_started:
                        break
                    continue

                # 6. 見出し形式になっている「ランキング」「関連記事」は足切りラインとみなす。
                #    ただし記事タイトル（# 見出し）自体にキーワードが含まれる場合があるため、
                #    ## 以降のセクション見出しのみを対象にする。
                if stripped.startswith("##"):
                    if any(k in stripped for k in HEADER_CUTOFF_KEYWORDS):
                        break
            if not stripped:
                cleaned_lines.append("")
                continue

            # 7. 長めの文の完全重複はノイズとしてスキップ
            if len(stripped) > 15:
                if stripped in seen_lines:
                    continue
                seen_lines.add(stripped)

            # 8. 短すぎるナビゲーション行のフィルタ
            if len(stripped) < 15 and all(c not in stripped for c in ["。", "、", "！", "？", "「", "」"]):
                if not stripped.startswith("#"):
                    continue

            cleaned_lines.append(line)
            body_started = True

        # 連続する空行を1つにまとめる
        text = '\n'.join(cleaned_lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
