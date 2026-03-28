import re
from src.cleaner.cleaner_constants import CUTOFF_KEYWORDS, HEADER_CUTOFF_KEYWORDS

def clean_markdown_text(raw_text: str) -> str:
    """
    ニュース本文（Markdown形式）から不要なメタデータやメニュー、フッター、広告文などを除去し、
    LLMに渡すためのクリーンなテキストを抽出する。
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
            
        # 2.5 見出し（##）形式になっている「ランキング」や「関連記事」は安全な足切りラインとみなす
        # （トップのナビゲーションの「* [ランキング]」には反応せず、下部のエリアだけを綺麗にカットできる）
        if stripped.startswith("#"):
            if any(k in stripped for k in HEADER_CUTOFF_KEYWORDS):
                break
            
        # 3. 画像タグの削除 ![alt](url) -> 完全削除
        line = re.sub(r'!\[.*?\]\(.*?\)', '', line)
        
        # 4. リンクタグのテキスト化 [text](url) -> text (空文字も許容)
        line = re.sub(r'\[([^\]]*)\]\([^\)]+\)', r'\1', line)
        
        # 5. HTMLタグの除去
        line = re.sub(r'<[^>]+>', '', line)
        
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
            
        # 5.5 長めの文の完全重複はノイズ（ニュースメディア特有の見出しとリード文の２重出力など）としてスキップ
        if len(stripped) > 15:
            if stripped in seen_lines:
                continue
            seen_lines.add(stripped)
            
        # 6. 「短すぎる＆リンクしかない」ようなナビゲーション行のフィルタ
        # （15文字未満で読点や句点、カギ括弧などがなく、見出しでもない短いテキストはメニューとみなす）
        if len(stripped) < 15 and all(c not in stripped for c in ["。", "、", "！", "？", "「", "」"]):
            if not stripped.startswith("#"):
                continue

        cleaned_lines.append(line)
        
    # 連続する空行を1つにまとめる
    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()
