import json
import re
import urllib.request
import urllib.error
from src.labeling.labeling_constants import LMSTUDIO_API_URL, LMSTUDIO_MODEL_NAME
from src.logger import get_logger

logger = get_logger(__name__)


class LlmLabeler:
    """
    LM Studio（OpenAI互換API）を呼び出し、ニュース記事の感情スコアを取得するクラス。
    """

    PROMPT_TEMPLATE = """\
あなたはニュース記事の感情分析AIです。
以下の記事を読み、6つの感情（喜び・怒り・悲しみ・恐れ・嫌悪・驚き）をそれぞれ0〜10の整数で評価してください。
記事が日本語・英語・数字やテーブルのみで構成されている場合でも、必ず以下の形式で出力してください。

【評価の視点】
特定の立場（当事者・専門家・活動家など）からではなく、**一般的な日本人読者が感じる平均的な感情反応**として評価してください。
記事の文体・トーン・内容から読み取れる感情を客観的に判断してください。

【出力形式】（必ずこの2行のみ）
1行目: JSONオブジェクト（Markdownの装飾なし、1行で）
2行目: 採点の根拠を1文で（日本語）

例:
{{"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "disgust": 0, "surprise": 0}}
この記事は中立的な事実報告で、強い感情は読み取れない。

【ニュース記事】
{text}"""

    def __init__(self, api_url: str = LMSTUDIO_API_URL, model_name: str = LMSTUDIO_MODEL_NAME):
        self.api_url = api_url
        self.model_name = model_name

    def label(self, text: str):
        """
        テキストを感情分析し、スコアとAIの生出力を返す。

        Returns:
            (scores: dict | None, raw_response: str)
        """
        prompt = self.PROMPT_TEMPLATE.format(text=text)

        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))

                actual_model = result.get("model", "unknown")
                logger.info(f"確認済モデル: {actual_model}")

                response_text = result["choices"][0]["message"]["content"]

                match = re.search(r'\{[^{}]+\}', response_text, re.DOTALL)
                if not match:
                    raise json.decoder.JSONDecodeError("No JSON object found in response", response_text, 0)
                cleaned_response = match.group(0)

                scores = json.loads(cleaned_response)
                return {
                    "joy":      float(scores.get("joy", 0)),
                    "anger":    float(scores.get("anger", 0)),
                    "sadness":  float(scores.get("sadness", 0)),
                    "fear":     float(scores.get("fear", 0)),
                    "disgust":  float(scores.get("disgust", 0)),
                    "surprise": float(scores.get("surprise", 0))
                }, response_text
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"[LLMエラー] LM Studioがリクエストを拒否しました (HTTP {e.code}): {error_body}")
            return None, ""
        except urllib.error.URLError as e:
            logger.error(f"[LLMエラー] LM Studioに接続できません ({self.api_url}): {e.reason}")
            return None, ""
        except json.decoder.JSONDecodeError as e:
            raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.error(f"[LLMエラー] AIの出力が不正なJSONでした: {e} | AIの生出力=''{raw[:200]}''")
            return None, raw
        except Exception as e:
            logger.error(f"[LLMエラー] 予期せぬエラーが発生しました: {e}")
            return None, ""
