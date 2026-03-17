"""
ニューススコアリングシステム エントリーポイント

フェーズ0: データ収集（RSS取得 → クローリング → 保存）
フェーズ1: AIラベリング（感情スコア付与）
フェーズ2: LoRAファインチューニング
フェーズ3: RAG統合
"""

from src.rss.rss_fetcher import RssFetcher


def phase0_collect():
    """フェーズ0: RSSからニュース記事を収集する"""
    print("=== フェーズ0: データ収集開始 ===")

    # RSS取得
    fetcher = RssFetcher()
    articles = fetcher.fetch_all()
    print(f"\n合計 {len(articles)} 件の記事を取得しました。")

    # TODO: クローリング（crawl4ai で本文取得）
    # TODO: 取得した本文をdata/raw/ にMarkdown形式で保存

    return articles


def phase1_label(articles):
    """フェーズ1: APIで感情スコアを付与する（1回限り）"""
    print("=== フェーズ1: AIラベリング開始 ===")
    # TODO: Claude / Gemini API を呼び出して感情スコアを付与
    # TODO: スコア済みデータを data/labeled/ に保存
    pass


def phase2_train():
    """フェーズ2: ローカルLLMをLoRAでファインチューニングする"""
    print("=== フェーズ2: LoRAファインチューニング開始 ===")
    # TODO: data/labeled/ のデータを使ってLoRAファインチューニング
    # TODO: 学習済みモデルを models/ に保存
    pass


def phase3_rag():
    """フェーズ3: RAGを使った感情文脈分析"""
    print("=== フェーズ3: RAG統合開始 ===")
    # TODO: スコアリング済み記事をFAISS等のベクトルDBに格納
    # TODO: 新着記事に対して類似記事の感情推移を参照して分析
    pass


if __name__ == "__main__":
    articles = phase0_collect()
    # phase1_label(articles)
    # phase2_train()
    # phase3_rag()
