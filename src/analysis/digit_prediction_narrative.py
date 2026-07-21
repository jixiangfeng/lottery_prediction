# -*- coding: utf-8 -*-
"""数字彩预测状态的确定性说明与可选DeepSeek润色。"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/chat/completions"
_DEEPSEEK_HOST = "api.deepseek.com"
_DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


@dataclass(frozen=True)
class DeepSeekNarrativeConfig:
    """DeepSeek文案请求配置；密钥来自本机忽略配置。"""

    api_key: str
    model: str
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("DeepSeek apiKey不能为空")
        if not self.model.strip():
            raise ValueError("DeepSeek model不能为空")
        if self.timeout <= 0:
            raise ValueError("AI请求超时必须大于零")


def deterministic_prediction_narrative(result: Mapping[str, object]) -> str:
    """根据结构化状态生成不夸大模型能力的中文说明。"""

    new_draws_value = result.get("newDrawsUsed", [])
    new_draws = new_draws_value if isinstance(new_draws_value, list) else []
    if bool(result.get("abstained", True)):
        update_text = (
            f"本次已纳入{len(new_draws)}期新增开奖，"
            if new_draws
            else "本次使用当前本地历史，"
        )
        return (
            f"{update_text}但影子模型尚未通过正式准入，"
            "因此不展示候选号码；内部排序仅用于继续收集前瞻证据。"
        )
    candidates_value = result.get("userVisibleCandidates", [])
    candidates = candidates_value if isinstance(candidates_value, list) else []
    return (
        f"模型已通过当前准入检查，共输出{len(candidates)}个锁定预算内候选。"
        "排序权重仅用于候选比较，不代表真实开奖概率。"
    )


def _narrative_context(result: Mapping[str, object]) -> dict[str, object]:
    """只向文案模型提供状态信息，不提供内部研究号码。"""

    visible_candidates = result.get("userVisibleCandidates", [])
    visible_count = (
        len(visible_candidates) if isinstance(visible_candidates, list) else 0
    )
    return {
        "lottery": result.get("lottery"),
        "latestHistoryIssue": result.get("latestHistoryIssue"),
        "newDrawsUsed": result.get("newDrawsUsed", []),
        "status": result.get("status"),
        "abstained": result.get("abstained"),
        "abstentionReasons": result.get("abstentionReasons", []),
        "signal": result.get("signal", {}),
        "prospectiveValidation": result.get("prospectiveValidation", {}),
        "userVisibleCandidateCount": visible_count,
    }


def load_deepseek_narrative_config(
    path: str | Path,
    *,
    model_override: str | None = None,
    timeout_override: float | None = None,
) -> DeepSeekNarrativeConfig:
    """从不会提交的本机JSON配置读取DeepSeek凭据。"""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"AI配置不存在：{source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"AI配置不是有效JSON：{source}") from error
    if not isinstance(payload, dict):
        raise ValueError("AI配置根节点必须是JSON对象")
    if payload.get("provider", "deepseek") != "deepseek":
        raise ValueError("AI配置provider只支持deepseek")
    api_key = str(payload.get("apiKey", "")).strip()
    model = str(model_override or payload.get("model", _DEFAULT_DEEPSEEK_MODEL)).strip()
    timeout_value = (
        timeout_override
        if timeout_override is not None
        else payload.get("timeout", 30.0)
    )
    return DeepSeekNarrativeConfig(
        api_key=api_key,
        model=model,
        timeout=float(timeout_value),
    )


def _extract_response_text(payload: Mapping[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("DeepSeek响应缺少choices数组")
    first = choices[0]
    if not isinstance(first, dict) or not isinstance(first.get("message"), dict):
        raise RuntimeError("DeepSeek响应缺少message对象")
    content = first["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("DeepSeek响应没有可用文案")
    return content.strip()


def request_deepseek_prediction_narrative(
    result: Mapping[str, object], config: DeepSeekNarrativeConfig
) -> str:
    """通过固定DeepSeek Chat API润色说明，不允许改变候选。"""

    instructions = (
        "你是彩票研究报告的中文编辑，只能根据输入事实写2到4句简体中文。"
        "不得新增、删除或重排候选号码，不得编造概率、信号或结论，"
        "不得承诺中奖或盈利。若abstained为true，必须明确说明本期不提供候选；"
        "不要使用Markdown标题或列表。"
    )
    body = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": json.dumps(
                    _narrative_context(result), ensure_ascii=False, sort_keys=True
                ),
            },
        ],
        "thinking": {"type": "disabled"},
        "stream": False,
        "max_tokens": 300,
    }
    request = urllib.request.Request(
        DEEPSEEK_CHAT_COMPLETIONS_URL,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            final_host = urllib.parse.urlparse(response.geturl()).hostname
            if final_host != _DEEPSEEK_HOST:
                raise RuntimeError(f"DeepSeek接口重定向到非白名单域名：{final_host}")
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"DeepSeek文案请求失败：HTTP {error.code}") from error
    except (urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"DeepSeek文案请求失败：{error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("DeepSeek响应根节点不是JSON对象")
    return _extract_response_text(payload)


__all__ = [
    "DEEPSEEK_CHAT_COMPLETIONS_URL",
    "DeepSeekNarrativeConfig",
    "deterministic_prediction_narrative",
    "load_deepseek_narrative_config",
    "request_deepseek_prediction_narrative",
]
