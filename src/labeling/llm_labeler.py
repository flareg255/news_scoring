import json
import urllib.request
from typing import Dict, Optional
from src.labeling.labeling_constants import LMSTUDIO_API_URL, LMSTUDIO_MODEL_NAME

def get_emotion_scores_from_lmstudio(text: str, model_name: str = LMSTUDIO_MODEL_NAME) -> Optional[Dict[str, float]]:
    """
    ローカルで稼働している LM Studio (デフォルトポート 1234) を呼び出し、
    指定したモデルに感情をJSONで出力させる関数。OpenAI互換APIを使用します。
    """
    url = LMSTUDIO_API_URL
    
    # 完全にJSONだけを吐き出させるための強固なプロンプト
    prompt = f"""あなたはニュース記事の感情分析AIです。
以下のニュース記事を読み、そこから読み取れる基本的な6つの感情「喜び(joy)」「怒り(anger)」「悲しみ(sadness)」「恐れ(fear)」「嫌悪(disgust)」「驚き(surprise)」を、それぞれ 0〜10 の整数値で評価してください。
出力は「必ず以下のJSON形式のみ」とし、前置きやMarkdownの装飾(```jsonなど)は一切含めないでください。

{{
    "joy": 5,
    "anger": 3,
    "sadness": 0,
    "fear": 0,
    "disgust": 0,
    "surprise": 0
}}

【ニュース記事】
{text}
"""

    # LM Studioが準拠している「OpenAI互換API形式」でのデータ作成
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.0,
        "response_format": { "type": "json_object" }  # LM StudioにJSONとしての回答を強制
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))
            
            # OpenAI互換APIのレスポンス構造からテキストを抽出
            response_text = result["choices"][0]["message"]["content"]
            
            # JSON文字列をPythonの辞書に変換
            scores = json.loads(response_text)
            return {
                "joy": float(scores.get("joy", 0)),
                "anger": float(scores.get("anger", 0)),
                "sadness": float(scores.get("sadness", 0)),
                "fear": float(scores.get("fear", 0)),
                "disgust": float(scores.get("disgust", 0)),
                "surprise": float(scores.get("surprise", 0))
            }
    except Exception as e:
        print(f"[LLMエラー] LM Studioの通信またはJSONパースに失敗しました: {e}")
        return None
