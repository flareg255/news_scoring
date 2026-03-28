import re
from src.cleaner.cleaner_constants import CUTOFF_KEYWORDS, HEADER_CUTOFF_KEYWORDS

def clean_markdown_text(raw_text: str) -> str:
    """
    ニュース本文（Markdown形式）から不要なメタデータやメニュー、フッター、広告文などを除去し、
    LLMに渡すためのクリーンなテキストを抽出する。
    """
    lines = raw_text.split('\n')
    cleaned_lines = []
    
    # これ以降の文が出現したら記事の終わりとみなして切り捨てる（フッター等の足切りキーワード）
    # 定数 (cleaner_constants.py) から読み込む
    
    for line in lines:
        stripped = line.strip()
        
        # 1. クローラーが付与した上部のメタデータの削除
        if stripped.startswith(("source:", "url:", "published_at:", "fetched_at:")) or stripped == "---":
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
        
        # 4. リンクタグのテキスト化 [text](url) -> text
        line = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
        
        # 5. HTMLタグの除去
        line = re.sub(r'<[^>]+>', '', line)
        
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append("")
            continue
            
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
