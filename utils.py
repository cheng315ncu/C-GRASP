#!/usr/bin/env python3
"""
MedGemma + PIKE-RAG 整合版本 - 工具函數模組
包含各種工具函數、驗證函數等
"""

import torch
import re
import os
import ast
import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
from transformers import LogitsProcessor, LogitsProcessorList

# 從 pikerag_medical_integration 導入元資料分隔符
try:
    from pikerag_medical_integration import METADATA_LIST_DELIMITER
except ImportError:
    METADATA_LIST_DELIMITER = "|"  # 預設分隔符


# ============================================================================
# 安全處理與清理函數
# ============================================================================

# 在生成前對 logits 做安全處理，避免 NaN/Inf 造成 CUDA 斷言
class SafeLogitsProcessor(LogitsProcessor):
    def __init__(self, clamp_min: float = -50.0, clamp_max: float = 50.0):
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 將 NaN/Inf 轉成有限值，並限制 logits 範圍，避免 multinomial 失敗
        scores = torch.nan_to_num(scores, nan=0.0, neginf=self.clamp_min, posinf=self.clamp_max)
        return scores.clamp(min=self.clamp_min, max=self.clamp_max)


def clean_catch():
    """
    Clean up CUDA cache and free unused memory on both GPU (if available) and CPU.
    """
    # Clear CUDA GPU cache if available
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            warnings.warn(f"Failed to clean CUDA cache: {e}")
    # Collect garbage on CPU
    try:
        import gc
        gc.collect()
    except Exception as e:
        warnings.warn(f"Failed to perform CPU RAM garbage collection: {e}")


def _safe_float(value):
    """嘗試轉換為浮點數，失敗則回傳 None。"""
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================================
# 文本處理與清理函數
# ============================================================================

INTERMEDIATE_LINE_FILTERS = [
    "i apologize",
    "user:",
    "assistant:",
    "great job",
    "here's the analysis",
    "here is the breakdown",
]


def sanitize_intermediate_text(text: str) -> str:
    """移除對話式填充與多餘換行，僅保留資訊內容。"""
    if not text:
        return ""

    filtered_lines: List[str] = []
    for line in str(text).splitlines():
        line_lower = line.strip().lower()
        if any(keyword in line_lower for keyword in INTERMEDIATE_LINE_FILTERS):
            continue
        filtered_lines.append(line)

    sanitized = "\n".join(filtered_lines)
    sanitized = re.sub(r"\bI apologize[^\n]*", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\bThanks?[^\n]*", "", sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r"\n{3,}", "\n\n", sanitized)
    return sanitized.strip()


def enforce_structured_output(raw_text: str, step_name: str) -> Tuple[str, Optional[Dict]]:
    """
    清理 Step1~Step7 的 LLM 輸出文本，移除代碼塊和異常內容。
    直接返回清理後的文本，不再進行 JSON 解析。
    """
    # 先進行基本清理
    sanitized = sanitize_intermediate_text(raw_text)

    # 移除所有 markdown 代碼塊標記
    sanitized = re.sub(r'```[a-z]*\s*\n?', '', sanitized, flags=re.IGNORECASE)
    sanitized = re.sub(r'```\s*\n?', '', sanitized)

    # 移除常見的解釋性前綴
    sanitized = re.sub(r'^(Here\'s|Here is|The JSON|JSON|Output|Result|Final Attempt|Key improvements|Explanation):\s*', '', sanitized, flags=re.IGNORECASE | re.MULTILINE)

    # 移除整個 Python 代碼塊（從 ```python 到 ```）
    sanitized = re.sub(r'```python.*?```', '', sanitized, flags=re.DOTALL | re.IGNORECASE)

    # 移除 Python 代碼相關的內容
    sanitized = re.sub(r'import\s+json.*?(?=\n\n|```|\Z)', '', sanitized, flags=re.DOTALL | re.IGNORECASE)
    sanitized = re.sub(r'def\s+\w+\([^)]*\):.*?(?=\n\n|```|\Z)', '', sanitized, flags=re.DOTALL)
    sanitized = re.sub(r'print\([^)]*\)', '', sanitized)

    # 移除包含 Python 關鍵字的整段內容
    lines = sanitized.split('\n')
    cleaned_lines = []
    skip_until_empty = False

    for line in lines:
        line_stripped = line.strip()
        # 檢測 Python 關鍵字
        if any(kw in line for kw in ['import json', 'def generate', 'if artifact_rate', 'elif artifact_rate', 'else:', 'return json.dumps']):
            skip_until_empty = True
            continue
        # 如果在跳過模式中
        if skip_until_empty:
            # 遇到空行或非代碼內容，停止跳過
            if line_stripped == '' or (not line_stripped.startswith('#') and not line_stripped.startswith('    ') and not line_stripped.startswith('\t')):
                skip_until_empty = False
                if line_stripped:  # 如果不是空行，保留這一行
                    cleaned_lines.append(line)
            continue
        cleaned_lines.append(line)

    sanitized = '\n'.join(cleaned_lines)

    # 移除多餘的空白行
    sanitized = re.sub(r'\n{3,}', '\n\n', sanitized)
    sanitized = sanitized.strip()

    return sanitized, None


# ============================================================================
# 數據處理與驗證函數
# ============================================================================

def detect_zscore_conflicts(step2_metrics: Dict, step6_metrics: Dict, tolerance: float = 1e-6) -> List[str]:
    """
    檢查 Step2 與 Step6 的 Z-score 是否出現符號矛盾；若有則回傳衝突鍵列表。
    """
    conflicts: List[str] = []
    if not step2_metrics or not step6_metrics:
        return conflicts

    for key, val in step2_metrics.items():
        z2 = _safe_float(val)
        z6 = _safe_float(step6_metrics.get(key))
        if z2 is None or z6 is None:
            continue
        if abs(z2) <= tolerance or abs(z6) <= tolerance:
            continue
        if (z2 > 0 and z6 < 0) or (z2 < 0 and z6 > 0):
            conflicts.append(key)
    return conflicts


def _parse_metadata_list(metadata: Dict, key: str) -> List[str]:
    """解析向量庫元資料中的列表欄位（支援分隔字串或原生列表）。"""
    value = metadata.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        iterable = value
    elif isinstance(value, str):
        iterable = value.split(METADATA_LIST_DELIMITER)
    else:
        iterable = [value]
    items: List[str] = []
    for item in iterable:
        text = str(item).strip()
        if text:
            items.append(text)
    return items


def build_clinical_summary(retrieved_docs: List[Dict]) -> str:
    """彙總檢索到的臨床文獻主題、指標與關鍵發現。"""
    if not retrieved_docs:
        return ""

    topic_counter: Counter = Counter()
    metric_counter: Counter = Counter()
    design_counter: Counter = Counter()
    unique_key_points: List[str] = []
    ranked_sources = []

    for entry in retrieved_docs:
        metadata = entry.get('metadata', {})
        topics = _parse_metadata_list(metadata, 'hrv_topics')
        metrics = _parse_metadata_list(metadata, 'hrv_metrics')
        key_points = _parse_metadata_list(metadata, 'key_points')
        study_design = metadata.get('study_design')

        for topic in topics:
            topic_counter[topic] += 1
        for metric in metrics:
            metric_counter[metric] += 1
        if study_design:
            design_counter[str(study_design)] += 1

        for point in key_points:
            if point not in unique_key_points:
                unique_key_points.append(point)

        source = metadata.get('source_file', metadata.get('source', '未知來源'))
        if source and source != '未知來源':
            source_name = os.path.basename(str(source))
        else:
            source_name = str(source)

        adjusted_score = entry.get('score', 0.0) or 0.0
        domain_weight = _safe_float(metadata.get('domain_weight'))
        evidence_weight = _safe_float(metadata.get('evidence_weight'))

        ranked_sources.append(
            (adjusted_score, source_name, domain_weight, evidence_weight)
        )

    ranked_sources.sort(key=lambda x: x[0], reverse=True)

    summary_lines = [
        "臨床證據總覽：",
        f"- 檢索到 {len(retrieved_docs)} 篇高相關臨床來源（已套用領域加權）。",
    ]

    if topic_counter:
        top_topics = ", ".join(
            f"{topic}×{count}" for topic, count in topic_counter.most_common(3)
        )
        summary_lines.append(f"- 主題焦點：{top_topics}")

    if metric_counter:
        top_metrics = ", ".join(
            f"{metric}×{count}" for metric, count in metric_counter.most_common(3)
        )
        summary_lines.append(f"- 指標重點：{top_metrics}")

    if design_counter:
        designs = ", ".join(
            f"{design}×{count}" for design, count in design_counter.most_common(3)
        )
        summary_lines.append(f"- 研究設計：{designs}")

    if ranked_sources:
        highlight_sources = []
        for score, source_name, domain_weight, evidence_weight in ranked_sources[:3]:
            detail_parts = [f"相關度 {score:.2f}"]
            if domain_weight is not None:
                detail_parts.append(f"DW {domain_weight:.2f}")
            if evidence_weight is not None:
                detail_parts.append(f"EW {evidence_weight:.2f}")
            highlight_sources.append(f"{source_name} [{' | '.join(detail_parts)}]")
        summary_lines.append("- 最高加權來源：" + "； ".join(highlight_sources))

    if unique_key_points:
        summary_lines.append("- 核心發現摘要：")
        for point in unique_key_points[:3]:
            summary_lines.append(f"    * {point}")

    return "\n".join(summary_lines)


# ============================================================================
# 輸出格式驗證函數
# ============================================================================

def validate_zscore_usage(decoded_text: str, zscore_features: Dict) -> Tuple[bool, Optional[str]]:
    """確認有 Z-score 時推理文本是否明確引用。"""
    features = zscore_features or {}
    has_zscore = any(_safe_float(v) is not None for v in features.values())
    if not has_zscore:
        return True, None
    lowered = decoded_text.lower()
    if any(keyword in lowered for keyword in ["z-score", "zscore", "baseline", "within-subject"]):
        return True, None
    warning = "警告：輸入含 Z-score，但推理文本未引用 Z-score / baseline，請人工覆核。"
    return False, warning


def validate_output_format(decoded_text: str, subject: str, trial: str) -> dict:
    """
    驗證模型輸出是否符合強制格式要求

    參數:
        decoded_text: 模型生成的文本
        subject: 受試者編號
        trial: 試次編號

    返回:
        dict: {
            'is_valid': bool,
            'missing_elements': list,
            'warnings': list,
            'extracted_state': str or None,
            'extracted_learning': str or None
        }
    """
    result = {
        'is_valid': True,
        'missing_elements': [],
        'warnings': [],
        'extracted_state': None,
        'extracted_learning': None
    }

    # 檢查 <think> 和 <answer> 標籤
    if '<think>' not in decoded_text or '</think>' not in decoded_text:
        result['is_valid'] = False
        result['missing_elements'].append('<think></think> section')

    if '<answer>' not in decoded_text or '</answer>' not in decoded_text:
        result['is_valid'] = False
        result['missing_elements'].append('<answer></answer> section')

    # 提取 <answer> 區塊內容
    answer_match = re.search(r'<answer>(.*?)</answer>', decoded_text, re.DOTALL)
    if answer_match:
        answer_content = answer_match.group(1)

        # 檢查必要的 6 個欄位
        required_fields = [
            'Inferred Psychophysiological State',
            'Inferred Affective State',
            'Inferred Learning State Correlate',
            'Confidence Level',
            'Key Rationale and Evidence',
            'Notes on Input Limitations'
        ]

        for field in required_fields:
            if field not in answer_content:
                result['is_valid'] = False
                result['missing_elements'].append(f'"{field}" field in <answer>')

        # 提取生理狀態標籤（HVLA/LVHA/LVLA/HVHA）
        state_patterns = [
            r'\*\*HVLA\*\*',
            r'\*\*LVHA\*\*',
            r'\*\*LVLA\*\*',
            r'\*\*HVHA\*\*',
            r'HVLA',
            r'LVHA',
            r'LVLA',
            r'HVHA'
        ]

        state_found = False
        for pattern in state_patterns:
            match = re.search(pattern, answer_content)
            if match:
                result['extracted_state'] = match.group(0).replace('**', '')
                state_found = True
                break

        if not state_found:
            result['warnings'].append('未找到明確的生理狀態標籤 (HVLA/LVHA/LVLA/HVHA)')

        # 提取學習狀態
        learning_keywords = [
            'Focused', 'Confident', 'Receptive', 'Zone',
            'Anxious', 'Stressed', 'Agitated', 'Overwhelmed',
            'Confused', 'Disengaged', 'Zoned Out', 'Fatigued',
            'Engaged', 'Curious', 'Problem-Solving'
        ]

        for keyword in learning_keywords:
            if keyword.lower() in answer_content.lower():
                result['extracted_learning'] = keyword
                break

        if not result['extracted_learning']:
            result['warnings'].append('未找到明確的學習狀態關鍵詞')

    # 檢查 <think> 區塊是否包含 7 個步驟
    think_match = re.search(r'<think>(.*?)</think>', decoded_text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1)
        step_count = len(re.findall(r'\*\*\d+\.', think_content))
        if step_count < 7:
            result['warnings'].append(f'<think> 區塊只包含 {step_count}/7 個步驟')

    return result


def _build_demographics_text(sample_data: Dict) -> str:
    """根據年齡與性別建立簡單個體背景字串。"""
    age = sample_data.get("age")
    gender = sample_data.get("gender")
    if age is not None and gender is not None:
        return (
            f"Please analyze the following physiological data for a "
            f"**{age}-year-old {gender} subject**."
        )
    return "Please analyze the following physiological data."
