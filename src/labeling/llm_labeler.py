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

    SCORE_KEYS = ("joy", "anger", "sadness", "fear", "disgust", "surprise")

    # 思考モデルが出力する制御トークン（<|channel>thought ... や <think>...</think>）
    THINK_BLOCK_RE = re.compile(r'<think>.*?</think>', re.DOTALL)
    CHANNEL_TOKEN_RE = re.compile(r'<\|?[a-z_]+\|?>')

    def __init__(self, api_url: str = LMSTUDIO_API_URL, model_name: str = LMSTUDIO_MODEL_NAME):
        self.api_url = api_url
        self.model_name = model_name

    @classmethod
    def _extract_scores(cls, response_text: str) -> dict | None:
        """
        AIの出力から感情スコアのJSONを取り出す。見つからなければ None を返す。
        Markdownの装飾・思考トークン・前後の解説文が混ざっていても拾えるようにする。
        """
        text = cls.THINK_BLOCK_RE.sub('', response_text or '')
        text = cls.CHANNEL_TOKEN_RE.sub('', text)

        # 対応する括弧を数えながら {...} を切り出す（ネストしたJSONにも対応）
        depth = 0
        start = -1
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}' and depth > 0:
                depth -= 1
                if depth == 0:
                    try:
                        candidate = json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        continue
                    # 6感情のいずれかを含むオブジェクトだけを採用する
                    found = cls._find_score_dict(candidate)
                    if found is not None:
                        return found
        return None

    @classmethod
    def _find_score_dict(cls, obj) -> dict | None:
        """dictを再帰的にたどり、感情キーを持つオブジェクトを探す（{"scores": {...}} 形式に対応）"""
        if not isinstance(obj, dict):
            return None
        if any(k in obj for k in cls.SCORE_KEYS):
            return obj
        for value in obj.values():
            found = cls._find_score_dict(value)
            if found is not None:
                return found
        return None

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
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"[LLMエラー] LM Studioがリクエストを拒否しました (HTTP {e.code}): {error_body}")
            return None, ""
        except urllib.error.URLError as e:
            logger.error(f"[LLMエラー] LM Studioに接続できません ({self.api_url}): {e.reason}")
            return None, ""
        except Exception as e:
            logger.error(f"[LLMエラー] 予期せぬエラーが発生しました: {e}")
            return None, ""

        try:
            actual_model = result.get("model", "unknown")
            logger.info(f"確認済モデル: {actual_model}")

            message = result["choices"][0]["message"]
            response_text = message.get("content") or ""

            scores = self._extract_scores(response_text)
            if scores is None:
                # 思考モデルは content が空で reasoning_content 側に本文を返すことがある
                scores = self._extract_scores(message.get("reasoning_content") or "")

            if scores is None:
                logger.error(
                    f"[LLMエラー] AIの出力からスコアJSONを取り出せませんでした | AIの生出力={response_text[:200]!r}"
                )
                return None, response_text

            return {key: float(scores.get(key, 0)) for key in self.SCORE_KEYS}, response_text
        except Exception as e:
            logger.error(f"[LLMエラー] レスポンスの解釈に失敗しました: {e}")
            return None, ""
