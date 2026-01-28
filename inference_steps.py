#!/usr/bin/env python3
"""
MedGemma + PIKE-RAG 整合版本 - 推理步驟模組
包含 Step1 到 Step7 的所有推理函數
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from typing import Dict, Optional
from utils import _safe_float, SafeLogitsProcessor, LogitsProcessorList


def _run_llm_simple(
    base_model,
    processor,
    system_prompt: str,
    user_prompt: str,
    image: Optional[Image.Image] = None,
    max_new_tokens: int = 1024,
) -> str:
    """
    執行一次簡化版的 MedGemma 推理（不強制 <think>/<answer> 結構），
    用於 Step1–Step7 等子報告生成。
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]
    if image is not None:
        messages[1]["content"].append({"type": "image", "image": image})

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(base_model.device)

    input_len = inputs["input_ids"].shape[-1]

    # 嘗試取得特殊終止符號
    eot_id = None
    try:
        if "<|eot_id|>" in processor.tokenizer.get_vocab():
            eot_id = processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    except Exception:
        eot_id = None

    eos_ids = [processor.tokenizer.eos_token_id]
    if eot_id is not None:
        eos_ids.append(eot_id)

    with torch.inference_mode():
        logits_processor = LogitsProcessorList([SafeLogitsProcessor()])
        generation = base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.3,  # 使用配置中的參數
            top_p=0.85,  # 使用配置中的參數
            # no_repeat_ngram_size=3,
            repetition_penalty=1.05,  # 略微增加重複懲罰以減少重複
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=eos_ids,
            logits_processor=logits_processor,
        )
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

    # 清理異常輸出：移除 HTML 標籤和異常重複文本
    # 0. 首先檢測並截斷異常的單字符重複模式（最優先）
    def detect_single_char_repetition_step(text: str) -> int:
        """檢測連續重複的單個字符或符號組合，返回異常開始位置"""
        single_char_patterns = [
            r'>{10,}',  # 連續的 >
            r'<{10,}',  # 連續的 <
            r'\({10,}',  # 連續的 (
            r'\){10,}',  # 連續的 )
            r'\[{10,}',  # 連續的 [
            r'\]{10,}',  # 連續的 ]
            r':{10,}',   # 連續的 :
            r';{10,}',   # 連續的 ;
            r',{10,}',   # 連續的 ,
            r'\.{10,}',  # 連續的 .
            r'"{10,}',   # 連續的 "
            r"'{10,}",   # 連續的 '
        ]

        earliest_pos = len(text)
        for pattern in single_char_patterns:
            matches = list(__import__('re').finditer(pattern, text))
            for match in matches:
                if match.start() < earliest_pos:
                    earliest_pos = match.start()

        # 檢測異常的符號組合模式
        combo_patterns = [
            r'(><){5,}',      # ><><><><><
            r'(\(\)){5,}',    # ()()()()()
            r'(\[\]){5,}',    # [][][][][]
            r'(:;){5,}',      # :;:;:;:;:;
            r'(::){5,}',      # ::::::
        ]

        for pattern in combo_patterns:
            matches = list(__import__('re').finditer(pattern, text))
            for match in matches:
                if match.start() < earliest_pos:
                    earliest_pos = match.start()

        return earliest_pos if earliest_pos < len(text) else -1

    single_char_pos = detect_single_char_repetition_step(decoded)
    if single_char_pos >= 0:
        decoded = decoded[:single_char_pos].rstrip()

    # 移除所有 HTML 標籤
    import re
    decoded = re.sub(r'<[^>]+>', '', decoded)

    # 1. 首先移除整個 Python 代碼塊（從 ```python 到 ```）
    decoded = re.sub(r'```python.*?```', '', decoded, flags=re.DOTALL | re.IGNORECASE)

    # 2. 移除其他代碼塊標記（```json, ``` 等），但保留內容
    decoded = re.sub(r'```[a-z]*\s*\n?', '', decoded, flags=re.IGNORECASE)
    decoded = re.sub(r'```\s*\n?', '', decoded)

    # 3. 移除 Python 代碼塊（如果包含 import, def, class 等關鍵字）
    # 使用更激進的方法：檢測到 Python 關鍵字後，移除直到下一個空段落或非代碼內容
    lines = decoded.split('\n')
    cleaned_lines = []
    in_code_block = False
    code_keywords = ['import ', 'def ', 'class ', 'print(', 'return ', 'if __name__', 'if artifact_rate', 'elif artifact_rate', 'else:', '    ']
    consecutive_code_lines = 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # 檢測是否進入代碼塊（包含 Python 關鍵字）
        if any(keyword in line for keyword in code_keywords) and not line_stripped.startswith('"'):
            in_code_block = True
            consecutive_code_lines = 0
            continue

        # 如果在代碼塊中
        if in_code_block:
            # 檢查是否是代碼行（縮進、註釋、字符串等）
            if (line_stripped.startswith('#') or
                line_stripped.startswith('"""') or
                line_stripped.startswith("'''") or
                (line_stripped and (line.startswith('    ') or line.startswith('\t')))):
                consecutive_code_lines += 1
                continue
            # 如果遇到空行，檢查後續是否還有代碼
            elif line_stripped == '':
                consecutive_code_lines = 0
                # 檢查下一行是否還是代碼
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if not any(keyword in next_line for keyword in code_keywords) and not next_line.startswith('#'):
                        # 如果下一行不是代碼，結束代碼塊
                        in_code_block = False
                continue
            else:
                # 遇到非代碼內容，結束代碼塊
                in_code_block = False
                cleaned_lines.append(line)
        else:
            # 不在代碼塊中，保留這一行
            cleaned_lines.append(line)

    decoded = '\n'.join(cleaned_lines)

    # 4. 移除殘留的 Python 函數定義和調用
    decoded = re.sub(r'def\s+\w+\([^)]*\):.*?(?=\n\n|\Z)', '', decoded, flags=re.DOTALL)
    decoded = re.sub(r'import\s+json.*?(?=\n\n|\Z)', '', decoded, flags=re.DOTALL | re.IGNORECASE)
    decoded = re.sub(r'print\([^)]*\)', '', decoded)

    # 移除異常重複文本
    abnormal_patterns = [
        r'executedterminating[^\s]*\s*',
        r'verification[^\s]*\s*',
        r'Termination Sequence Executed[^\n]*',
        r'ERROR:.*?JSON',
        r'Let\'s try.*?again',
        r'Final Attempt:',
        r'Key improvements.*?',
        r'Explanation:.*?',
    ]
    for pattern in abnormal_patterns:
        decoded = re.sub(pattern, '', decoded, flags=re.IGNORECASE | re.DOTALL)

    # 移除多餘的空白行
    decoded = re.sub(r'\n{3,}', '\n\n', decoded)

    return decoded.strip()


def _run_step1_quality(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step1: 訊號品質與前處理審查（擴展版：包含實際數據量化特徵 + 綜合品質指標）

    使用 SignalQuality_panel 圖 + 品質指標（artifact_rate, ectopy 等）+
    實際數據量化特徵（避免圖像縮放失真）+ 綜合品質指標（overall_quality 等），
    產出品質等級與對後續分析的建議。
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH

    quality = sample_data.get("quality_features") or {}
    img_feats = sample_data.get("image_features") or {}
    sq_data_feats = sample_data.get("signal_quality_data_features") or {}

    def gf(name: str):
        return _safe_float(quality.get(name))

    # 原有品質指標
    artifact_rate = gf("artifact_rate")
    ectopy_count = gf("ectopy_count")
    ectopy_flag = gf("ectopy_flag")
    valid_rr_ratio = gf("valid_rr_ratio")
    window_start_s = gf("window_start_s")
    window_end_s = gf("window_end_s")
    window_duration_s = gf("window_duration_s")
    
    # === 新增：綜合品質指標 ===
    overall_quality = gf("overall_quality")
    data_length_quality = gf("data_length_quality")
    artifact_quality = gf("artifact_quality")
    nonlinear_quality = gf("nonlinear_quality")
    respiratory_interference = gf("respiratory_interference")
    quality_flags_str = quality.get("quality_flags_str", "")
    
    # === 新增：頻域品質指標 ===
    hf_rsa_overlap = gf("HF_RSA_overlap")
    hf_reliability = gf("HF_reliability")
    ulf_dominance_flag = gf("ULF_dominance_flag")
    spectral_quality = gf("spectral_quality")
    
    # === 新增：SampEn 品質指標 ===
    sampen_confidence = gf("SampEn_confidence")
    sampen_reliability = quality.get("SampEn_reliability", "")

    # 圖像像素特徵
    sq_mean = _safe_float(img_feats.get("signal_quality_img_mean"))
    sq_std = _safe_float(img_feats.get("signal_quality_img_std"))
    sq_slope = _safe_float(img_feats.get("signal_quality_row_slope"))

    # === 新增：實際數據量化特徵（避免圖像縮放失真） ===
    sq_rri_min = _safe_float(sq_data_feats.get("sq_rri_min"))
    sq_rri_max = _safe_float(sq_data_feats.get("sq_rri_max"))
    sq_rri_range = _safe_float(sq_data_feats.get("sq_rri_range"))
    sq_time_duration = _safe_float(sq_data_feats.get("sq_time_duration"))
    sq_ecg_amplitude_min = _safe_float(sq_data_feats.get("sq_ecg_amplitude_min"))
    sq_ecg_amplitude_max = _safe_float(sq_data_feats.get("sq_ecg_amplitude_max"))
    sq_ecg_amplitude_range = _safe_float(sq_data_feats.get("sq_ecg_amplitude_range"))
    sq_ecg_amplitude_mean = _safe_float(sq_data_feats.get("sq_ecg_amplitude_mean"))
    sq_ecg_amplitude_std = _safe_float(sq_data_feats.get("sq_ecg_amplitude_std"))

    desc_lines = [
        "Signal quality metrics for this ECG/HRV window:",
        (
            f"- Window: start={window_start_s:.2f}s, end={window_end_s:.2f}s, duration={window_duration_s:.2f}s"
            if window_start_s is not None and window_end_s is not None and window_duration_s is not None
            else "- Window: not fully specified."
        ),
        f"- Artifact rate (approx.): {artifact_rate:.3f}" if artifact_rate is not None else "- Artifact rate: NA",
        f"- Valid RR ratio: {valid_rr_ratio:.3f}" if valid_rr_ratio is not None else "- Valid RR ratio: NA",
        f"- Ectopy count (RR outside 300–2000 ms): {int(ectopy_count) if ectopy_count is not None else 'NA'}",
        f"- Ectopy flag: {int(ectopy_flag) if ectopy_flag is not None else 'NA'}",
        "",
        "=== Comprehensive Quality Indicators (0-1 scale, higher is better) ===",
        f"- Overall quality: {overall_quality:.3f}" if overall_quality is not None else "- Overall quality: NA",
        f"- Data length quality: {data_length_quality:.3f}" if data_length_quality is not None else "- Data length quality: NA",
        f"- Artifact quality: {artifact_quality:.3f}" if artifact_quality is not None else "- Artifact quality: NA",
        f"- Nonlinear quality: {nonlinear_quality:.3f}" if nonlinear_quality is not None else "- Nonlinear quality: NA",
        f"- Spectral quality: {spectral_quality:.3f}" if spectral_quality is not None else "- Spectral quality: NA",
        f"- Respiratory interference: {respiratory_interference:.3f}" if respiratory_interference is not None else "- Respiratory interference: NA",
        f"- Quality flags: {quality_flags_str}" if quality_flags_str else "- Quality flags: none",
        "",
        "=== Frequency-domain Quality Indicators ===",
        f"- HF-RSA overlap (0=none, 1=full): {hf_rsa_overlap:.3f}" if hf_rsa_overlap is not None else "- HF-RSA overlap: NA",
        f"- HF reliability: {hf_reliability:.3f}" if hf_reliability is not None else "- HF reliability: NA",
        f"- ULF dominance flag (0=normal, 1=dominant): {ulf_dominance_flag:.3f}" if ulf_dominance_flag is not None else "- ULF dominance flag: NA",
        "",
        "=== Nonlinear Feature Quality ===",
        f"- SampEn confidence (0-1): {sampen_confidence:.3f}" if sampen_confidence is not None else "- SampEn confidence: NA",
        f"- SampEn reliability: {sampen_reliability}" if sampen_reliability else "- SampEn reliability: NA",
        "",
        "SignalQuality_panel image quantitative features (grayscale, 896x896):",
        f"- Mean intensity: {sq_mean:.4f}" if sq_mean is not None else "- Mean intensity: NA",
        f"- Intensity std: {sq_std:.4f}" if sq_std is not None else "- Intensity std: NA",
        f"- Row-wise mean slope (top→bottom): {sq_slope:.4e}" if sq_slope is not None else "- Row-wise mean slope: NA",
        "",
        "=== Actual Data Quantitative Features (from raw signals, immune to image scaling) ===",
        "RRI (RR Interval) Statistics:",
        f"- RRI min: {sq_rri_min:.2f} ms" if sq_rri_min is not None else "- RRI min: NA",
        f"- RRI max: {sq_rri_max:.2f} ms" if sq_rri_max is not None else "- RRI max: NA",
        f"- RRI range: {sq_rri_range:.2f} ms" if sq_rri_range is not None else "- RRI range: NA",
        f"- Time duration: {sq_time_duration:.2f} s" if sq_time_duration is not None else "- Time duration: NA",
        "",
        "ECG Amplitude Statistics (from raw ECG signal, units: arbitrary/μV - device-dependent):",
        "NOTE: Amplitude values are in raw sensor units (typically μV or ADC counts). Normal ECG R-wave amplitude",
        "      varies widely by device (10-1000+ units). Assess RELATIVE variability, not absolute magnitude.",
        f"- Amplitude min: {sq_ecg_amplitude_min:.2f}" if sq_ecg_amplitude_min is not None else "- Amplitude min: NA",
        f"- Amplitude max: {sq_ecg_amplitude_max:.2f}" if sq_ecg_amplitude_max is not None else "- Amplitude max: NA",
        f"- Amplitude range: {sq_ecg_amplitude_range:.2f}" if sq_ecg_amplitude_range is not None else "- Amplitude range: NA",
        f"- Amplitude mean: {sq_ecg_amplitude_mean:.2f}" if sq_ecg_amplitude_mean is not None else "- Amplitude mean: NA",
        f"- Amplitude std: {sq_ecg_amplitude_std:.2f}" if sq_ecg_amplitude_std is not None else "- Amplitude std: NA",
    ]

    user_prompt = (
        "Step 1 - Signal Quality Review.\n"
        "Evaluate signal quality and preprocessing adequacy using your clinical knowledge.\n\n"
        + "\n".join(desc_lines)
        + "\n\n"
        "Provide a comprehensive analysis. Think through all aspects carefully before concluding."
    )

    system_prompt = (
        "You are assessing ECG/HRV signal quality and preprocessing adequacy.\n"
        "Essential constraints: Prioritize quantitative signal metrics over image features. "
        "Focus strictly on data quality; avoid emotional/psychological interpretations."
    )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=sample_data.get("signal_quality_image"),
        max_new_tokens=INFERENCE_OUTPUT_LENGTH,
    )


def _run_step2_time_domain(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step2: 時域 HRV 指標歸納（RMSSD, SDNN, NN50, pNN50, MeanHR 等）。
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH

    raw = sample_data.get("raw_features") or {}
    zscores = sample_data.get("zscore_features") or {}

    names = [
        "RMSSD_ms",
        "SDNN_ms",
        "NN50",
        "pNN50",
        "MeanHR_bpm",
        "SDHR_bpm",
    ]

    lines = [
        "Time-domain HRV features (with within-sample z-scores when available):",
        "",
        "=== UNIT REFERENCE ===",
        "- RMSSD_ms: Root Mean Square of Successive Differences (milliseconds)",
        "- SDNN_ms: Standard Deviation of NN intervals (milliseconds)",
        "- NN50: Count of NN intervals differing by >50ms (unitless count)",
        "- pNN50: Percentage of NN50 (%, range 0-100)",
        "- MeanHR_bpm: Mean Heart Rate (beats per minute, typical resting: 60-100 bpm)",
        "- SDHR_bpm: Standard Deviation of Heart Rate (bpm)",
        "",
        "=== MEASURED VALUES ===",
    ]
    for name in names:
        v = raw.get(name)
        zv = zscores.get(f"{name}_zscore")
        try:
            v_f = float(v) if v is not None else None
        except (TypeError, ValueError):
            v_f = None
        try:
            z_f = float(zv) if zv is not None else None
        except (TypeError, ValueError):
            z_f = None

        if v_f is not None and z_f is not None:
            lines.append(f"- {name}: {v_f:.4f} (z = {z_f:.2f})")
        elif v_f is not None:
            lines.append(f"- {name}: {v_f:.4f}")
        else:
            lines.append(f"- {name}: Not Available")

    user_prompt = (
        "Step 2 - Time-domain HRV analysis.\n"
        "Analyze the time-domain features using your clinical knowledge.\n\n"
        + "\n".join(lines)
        + "\n\n"
        "INTERPRETATION NOTES:\n"
        "- Normal RMSSD range: 20-50 ms (young healthy adults); <20 ms suggests reduced vagal tone.\n"
        "- Normal SDNN range: 50-100 ms (short-term); <50 ms indicates low HRV.\n"
        "- Normal pNN50 range: 5-25%; <5% indicates reduced parasympathetic activity.\n"
        "- Z-scores: 0 = population mean; ±1 = within 1 SD; ±2 = notable deviation.\n\n"
        "Provide a comprehensive analysis. Think through all aspects carefully before concluding."
    )

    system_prompt = (
        "You are interpreting time-domain HRV metrics for autonomic assessment.\n"
        "Essential constraints: Prioritize Z-scores for within-subject baseline comparisons. "
        "Integrate multiple indicators with clinical judgment."
    )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=None,
        max_new_tokens=INFERENCE_OUTPUT_LENGTH,
    )


def _run_step3_frequency(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step3: 頻域 / 呼吸特徵解析（擴展版：含 PSD 圖像 + 實際數據量化特徵）。
    支援 ENABLE_GUARDRAILS flag 控制呼吸干擾檢測與 system_prompt 約束。
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH, ENABLE_GUARDRAILS
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH
        ENABLE_GUARDRAILS = getattr(config, 'ENABLE_GUARDRAILS', True)

    raw = sample_data.get("raw_features") or {}
    img_feats = sample_data.get("image_features") or {}
    psd_data_feats = sample_data.get("psd_data_features") or {}
    edr_feats = sample_data.get("edr_features") or {}

    freq_names = [
        "ULF_peak",
        "LF_peak",
        "HF_peak",
        "ULF_ratio",
        "LF_ratio",
        "HF_ratio",
        "LF_HF",
    ]

    lines = [
        "Frequency-domain HRV features:",
        "",
        "=== UNIT REFERENCE ===",
        "- ULF_peak, LF_peak, HF_peak: Peak frequency in each band (Hz)",
        "  * ULF band: 0.00-0.04 Hz (very slow oscillations)",
        "  * LF band: 0.04-0.15 Hz (sympathetic + baroreflex)",
        "  * HF band: 0.15-0.40 Hz (parasympathetic/vagal, respiratory-coupled)",
        "- ULF_ratio, LF_ratio, HF_ratio: Proportion of total power in each band (0-1 scale)",
        "- LF_HF: LF/HF ratio (sympathovagal balance indicator, typical range: 0.5-2.0)",
        "- Power values below are in ms²/Hz (PSD units)",
        "",
        "=== MEASURED VALUES ===",
    ]
    for name in freq_names:
        v = raw.get(name)
        try:
            v_f = float(v) if v is not None else None
        except (TypeError, ValueError):
            v_f = None
        if v_f is not None:
            # 為不同類型的特徵添加適當的單位
            if "peak" in name.lower():
                lines.append(f"- {name}: {v_f:.4f} Hz")
            elif "ratio" in name.lower() and "LF_HF" not in name:
                lines.append(f"- {name}: {v_f:.4f} (proportion)")
            else:
                lines.append(f"- {name}: {v_f:.4f}")
        else:
            lines.append(f"- {name}: Not Available")

    # 圖像像素特徵
    psd_mean = _safe_float(img_feats.get("psd_img_mean"))
    psd_std = _safe_float(img_feats.get("psd_img_std"))
    psd_slope = _safe_float(img_feats.get("psd_row_slope"))

    # === 新增：實際數據量化特徵（避免圖像縮放失真） ===
    psd_freq_min = _safe_float(psd_data_feats.get("psd_freq_min"))
    psd_freq_max = _safe_float(psd_data_feats.get("psd_freq_max"))
    psd_freq_range = _safe_float(psd_data_feats.get("psd_freq_range"))
    psd_power_min = _safe_float(psd_data_feats.get("psd_power_min"))
    psd_power_max = _safe_float(psd_data_feats.get("psd_power_max"))
    psd_power_mean = _safe_float(psd_data_feats.get("psd_power_mean"))
    psd_power_std = _safe_float(psd_data_feats.get("psd_power_std"))
    psd_power_median = _safe_float(psd_data_feats.get("psd_power_median"))
    psd_peak_freq = _safe_float(psd_data_feats.get("psd_peak_freq"))
    psd_peak_power = _safe_float(psd_data_feats.get("psd_peak_power"))
    psd_total_power = _safe_float(psd_data_feats.get("psd_total_power"))
    psd_band_ulf_power = _safe_float(psd_data_feats.get("psd_band_ulf_power"))
    psd_band_lf_power = _safe_float(psd_data_feats.get("psd_band_lf_power"))
    psd_band_hf_power = _safe_float(psd_data_feats.get("psd_band_hf_power"))

    # === 新增：呼吸干擾檢測機制 (Respiratory Guardrails) ===
    # 提取關鍵頻率用於 RSA 干擾檢測
    edr_freq = _safe_float(edr_feats.get('respiratory_rate_hz'))
    edr_freq_bpm = _safe_float(edr_feats.get('respiratory_rate_bpm'))
    edr_snr = _safe_float(edr_feats.get('edr_snr'))
    hf_peak = _safe_float(raw.get('HF_peak'))
    hf_ratio = _safe_float(raw.get('HF_ratio'))
    lf_hf_ratio = _safe_float(raw.get('LF_HF'))
    
    # 從品質指標中獲取 RSA 重疊信息
    quality_feats = sample_data.get("quality_features") or {}
    hf_rsa_overlap = _safe_float(quality_feats.get("HF_RSA_overlap"))
    hf_reliability = _safe_float(quality_feats.get("HF_reliability"))
    ulf_dominance_flag = _safe_float(quality_feats.get("ULF_dominance_flag"))
    spectral_quality = _safe_float(quality_feats.get("spectral_quality"))
    
    # 生成呼吸干擾警告（受 ENABLE_GUARDRAILS 控制）
    respiratory_warnings = []
    respiratory_severity = "none"  # none / mild / moderate / severe
    
    # 只有在 ENABLE_GUARDRAILS 啟用時才進行呼吸干擾檢測
    if not ENABLE_GUARDRAILS:
        # Guardrails 關閉：跳過所有呼吸干擾檢測
        pass
    # 檢測 1: 呼吸頻率是否落入 HF 頻帶且與 HF Peak 接近
    elif edr_freq is not None and hf_peak is not None:
        # 呼吸頻率是否在 HF 頻帶 (0.15-0.4 Hz)
        if 0.15 <= edr_freq <= 0.40:
            freq_diff = abs(edr_freq - hf_peak)
            if freq_diff < 0.03:
                respiratory_severity = "severe"
                respiratory_warnings.append(
                    f"⚠️ CRITICAL RSA CONTAMINATION: HF peak ({hf_peak:.3f} Hz) aligns almost exactly "
                    f"with respiratory frequency ({edr_freq:.3f} Hz, diff={freq_diff:.3f} Hz).\n"
                    "The HF power is DOMINATED by Respiratory Sinus Arrhythmia (RSA), NOT pure vagal tone.\n"
                    "DO NOT interpret high HF as strong parasympathetic activity.\n"
                    "The LF/HF ratio is UNRELIABLE for sympathovagal balance assessment."
                )
            elif freq_diff < 0.05:
                respiratory_severity = "severe"
                respiratory_warnings.append(
                    f"⚠️ SEVERE RSA INTERFERENCE: HF peak ({hf_peak:.3f} Hz) closely aligns with "
                    f"respiratory frequency ({edr_freq:.3f} Hz, diff={freq_diff:.3f} Hz).\n"
                    "High HF power may be driven by RSA rather than pure vagal tone.\n"
                    "Interpret LF/HF ratio with EXTREME caution."
                )
            elif freq_diff < 0.08:
                respiratory_severity = "moderate"
                respiratory_warnings.append(
                    f"⚠️ MODERATE RSA INFLUENCE: HF peak ({hf_peak:.3f} Hz) is near "
                    f"respiratory frequency ({edr_freq:.3f} Hz, diff={freq_diff:.3f} Hz).\n"
                    "RSA likely contributes to HF power. Interpret with caution."
                )
    
    # 檢測 2-5: 只有在 ENABLE_GUARDRAILS 啟用時才執行
    if ENABLE_GUARDRAILS:
        # 檢測 2: 呼吸頻率異常（過快或過慢）
        if edr_freq_bpm is not None:
            if edr_freq_bpm > 24:  # > 0.4 Hz, 超出標準 HF 頻帶
                respiratory_warnings.append(
                    f"⚠️ RAPID BREATHING DETECTED: Respiratory rate = {edr_freq_bpm:.1f} bpm ({edr_freq:.3f} Hz).\n"
                    "This exceeds the standard HF band (0.15-0.4 Hz). Respiratory component may alias into VHF or cause artifacts."
                )
            elif edr_freq_bpm < 9:  # < 0.15 Hz, 落入 LF 頻帶
                respiratory_warnings.append(
                    f"⚠️ SLOW BREATHING DETECTED: Respiratory rate = {edr_freq_bpm:.1f} bpm ({edr_freq:.3f} Hz).\n"
                    "This falls into the LF band (0.04-0.15 Hz). LF power may be contaminated by respiratory modulation.\n"
                    "LF/HF ratio interpretation is compromised."
                )
                if respiratory_severity == "none":
                    respiratory_severity = "moderate"
        
        # 檢測 3: 使用預計算的品質指標
        if hf_rsa_overlap is not None and hf_rsa_overlap > 0.5:
            if "RSA" not in str(respiratory_warnings):  # 避免重複警告
                respiratory_warnings.append(
                    f"⚠️ HF-RSA OVERLAP INDICATOR: {hf_rsa_overlap:.2f} (high overlap detected by preprocessing).\n"
                    "HF power reliability is reduced."
                )
        
        # 檢測 4: ULF 主導警告（可能是去趨勢問題）
        if ulf_dominance_flag is not None and ulf_dominance_flag > 0.5:
            respiratory_warnings.append(
                f"⚠️ ULF DOMINANCE DETECTED: ULF_dominance_flag = {ulf_dominance_flag:.2f}.\n"
                "Low-frequency trends may dominate the spectrum. This could indicate:\n"
                "- Insufficient detrending in preprocessing\n"
                "- Non-stationary heart rate drift\n"
                "- Very slow autonomic oscillations\n"
                "LF and HF band powers may be underestimated relative to ULF."
            )
            if respiratory_severity in ["none", "mild"]:
                respiratory_severity = "moderate"
        
        # 檢測 5: EDR SNR 低警告
        if edr_snr is not None and edr_snr < 3.0:
            respiratory_warnings.append(
                f"Note: EDR signal quality is low (SNR = {edr_snr:.2f} dB).\n"
                "Respiratory frequency estimate may be unreliable."
            )

    lines.extend(
        [
            "",
            "Quantitative summary of PSD image (log-PSD vs frequency, 896x896 grayscale):",
            f"- Mean intensity: {psd_mean:.4f}" if psd_mean is not None else "- Mean intensity: NA",
            f"- Intensity std: {psd_std:.4f}" if psd_std is not None else "- Intensity std: NA",
            f"- Row-wise mean slope (top→bottom): {psd_slope:.4e}" if psd_slope is not None else "- Row-wise mean slope: NA",
            "",
            "=== Actual PSD Data Quantitative Features (from raw signals, immune to image scaling) ===",
            "Frequency Range:",
            f"- Frequency min: {psd_freq_min:.4f} Hz" if psd_freq_min is not None else "- Frequency min: NA",
            f"- Frequency max: {psd_freq_max:.4f} Hz" if psd_freq_max is not None else "- Frequency max: NA",
            f"- Frequency range: {psd_freq_range:.4f} Hz" if psd_freq_range is not None else "- Frequency range: NA",
            "",
            "Power Statistics:",
            f"- Power min: {psd_power_min:.6e}" if psd_power_min is not None else "- Power min: NA",
            f"- Power max: {psd_power_max:.6e}" if psd_power_max is not None else "- Power max: NA",
            f"- Power mean: {psd_power_mean:.6e}" if psd_power_mean is not None else "- Power mean: NA",
            f"- Power std: {psd_power_std:.6e}" if psd_power_std is not None else "- Power std: NA",
            f"- Power median: {psd_power_median:.6e}" if psd_power_median is not None else "- Power median: NA",
            "",
            "Peak Characteristics:",
            f"- Peak frequency: {psd_peak_freq:.4f} Hz" if psd_peak_freq is not None else "- Peak frequency: NA",
            f"- Peak power: {psd_peak_power:.6e}" if psd_peak_power is not None else "- Peak power: NA",
            f"- Total power: {psd_total_power:.6e}" if psd_total_power is not None else "- Total power: NA",
            "",
            "Band Power (from actual PSD data):",
            f"- ULF band power (0.00-0.04 Hz): {psd_band_ulf_power:.6e}" if psd_band_ulf_power is not None else "- ULF band power: NA",
            f"- LF band power (0.04-0.15 Hz): {psd_band_lf_power:.6e}" if psd_band_lf_power is not None else "- LF band power: NA",
            f"- HF band power (0.15-0.40 Hz): {psd_band_hf_power:.6e}" if psd_band_hf_power is not None else "- HF band power: NA",
            "",
            "=== EDR (ECG-derived Respiration) Features ===",
            "Respiratory Rate (estimated from R-peak amplitude modulation):",
            "NOTE: EDR peak power is in arbitrary units (depends on ECG amplitude scaling).",
            "      Values range from 10^1 to 10^4 - do NOT interpret absolute magnitude as 'high' or 'low'.",
            "      Focus on EDR SNR (>3 dB = good) and frequency alignment with HF band.",
            f"- Respiratory rate: {edr_freq_bpm:.2f} bpm" if edr_freq_bpm is not None else "- Respiratory rate: NA",
            f"- Respiratory frequency: {edr_freq:.4f} Hz" if edr_freq is not None else "- Respiratory frequency: NA",
            f"- EDR peak frequency: {_safe_float(edr_feats.get('edr_peak_freq_hz')):.4f} Hz" if _safe_float(edr_feats.get('edr_peak_freq_hz')) is not None else "- EDR peak frequency: NA",
            f"- EDR peak power: {_safe_float(edr_feats.get('edr_peak_power')):.6e} (arbitrary units)" if _safe_float(edr_feats.get('edr_peak_power')) is not None else "- EDR peak power: NA",
            f"- EDR SNR: {edr_snr:.2f} dB (>3 dB = reliable)" if edr_snr is not None else "- EDR SNR: NA",
            "",
            "=== Spectral Quality Indicators ===",
            f"- HF-RSA overlap: {hf_rsa_overlap:.3f}" if hf_rsa_overlap is not None else "- HF-RSA overlap: NA",
            f"- HF reliability: {hf_reliability:.3f}" if hf_reliability is not None else "- HF reliability: NA",
            f"- ULF dominance flag: {ulf_dominance_flag:.3f}" if ulf_dominance_flag is not None else "- ULF dominance flag: NA",
            f"- Overall spectral quality: {spectral_quality:.3f}" if spectral_quality is not None else "- Spectral quality: NA",
        ]
    )
    
    # 構建呼吸干擾警告區塊
    respiratory_warning_block = ""
    if respiratory_warnings:
        respiratory_warning_block = (
            "\n\n=== RESPIRATORY INTERFERENCE ASSESSMENT ===\n"
            f"Severity Level: {respiratory_severity.upper()}\n\n"
            + "\n\n".join(respiratory_warnings)
        )

    user_prompt = (
        "Step 3 - Frequency-domain HRV and respiratory-related features.\n"
        "Analyze the PSD characteristics using your clinical knowledge.\n\n"
        + "\n".join(lines)
        + respiratory_warning_block
        + "\n\n"
        "IMPORTANT INTERPRETATION GUIDELINES:\n"
        "1. ALWAYS check the respiratory frequency vs HF peak alignment FIRST.\n"
        "2. If respiratory frequency ≈ HF peak, HF power reflects RSA, NOT pure vagal tone.\n"
        "3. LF/HF ratio is ONLY valid when respiratory rate is stable within 0.15-0.4 Hz band.\n"
        "4. Slow breathing (<9 bpm) contaminates LF; fast breathing (>24 bpm) may alias.\n"
        "5. Check ULF dominance flag for potential detrending issues.\n\n"
        "Provide a comprehensive analysis. Think through all aspects carefully before concluding."
    )

    # 根據呼吸干擾嚴重程度調整 system_prompt（受 ENABLE_GUARDRAILS 控制）
    if not ENABLE_GUARDRAILS:
        # Guardrails 關閉：使用無約束的基礎 system_prompt
        system_prompt = (
            "You are analyzing frequency-domain HRV and respiratory modulation patterns.\n"
            "Essential constraints:\n"
            "1. Prioritize quantitative PSD metrics over image features.\n"
            "2. Provide balanced interpretation of frequency-domain findings."
        )
    elif respiratory_severity == "severe":
        system_prompt = (
            "You are analyzing frequency-domain HRV and respiratory modulation patterns.\n"
            "CRITICAL: RSA contamination is detected in this sample.\n"
            "Essential constraints:\n"
            "1. DO NOT interpret high HF power as strong parasympathetic activity.\n"
            "2. DO NOT use LF/HF ratio for sympathovagal balance assessment.\n"
            "3. The HF band is dominated by respiratory modulation (RSA).\n"
            "4. Focus on absolute power values and time-domain metrics instead.\n"
            "5. Acknowledge spectral limitations explicitly in your analysis."
        )
    elif respiratory_severity == "moderate":
        system_prompt = (
            "You are analyzing frequency-domain HRV and respiratory modulation patterns.\n"
            "CAUTION: Respiratory modulation may influence HF band power.\n"
            "Essential constraints:\n"
            "1. Interpret HF power with awareness of potential RSA contribution.\n"
            "2. Use LF/HF ratio with methodological caution.\n"
            "3. Cross-validate frequency findings with time-domain metrics.\n"
            "4. Acknowledge spectral limitations."
        )
    else:
        system_prompt = (
            "You are analyzing frequency-domain HRV and respiratory modulation patterns.\n"
            "No significant respiratory interference detected.\n"
            "Essential constraints:\n"
            "1. Prioritize quantitative PSD metrics over image features.\n"
            "2. Assess LF/HF ratios with standard methodological caution.\n"
            "3. Acknowledge any spectral limitations."
        )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=sample_data.get("psd_image"),
        max_new_tokens=int(2 * INFERENCE_OUTPUT_LENGTH),
    )


def _run_step4_poincare_complexity(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step4: Poincaré 幾何 / 複雜度（擴展版：SD1/SD2, SampEn, DFA + 實際數據量化特徵）。
    支援 ENABLE_GUARDRAILS flag 控制非線性指標可信度評估與 system_prompt 約束。
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH, ENABLE_GUARDRAILS
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH
        ENABLE_GUARDRAILS = getattr(config, 'ENABLE_GUARDRAILS', True)

    raw = sample_data.get("raw_features") or {}
    zscores = sample_data.get("zscore_features") or {}
    img_feats = sample_data.get("image_features") or {}
    poincare_data_feats = sample_data.get("poincare_data_features") or {}

    names = [
        "SD1_ms",
        "SD2_ms",
        "SD1_SD2",
        "SampEn",
        "DFA_alpha",
        "SDNN_index_ms",
    ]

    lines = [
        "Nonlinear / geometric HRV features:",
        "",
        "=== UNIT REFERENCE ===",
        "- SD1_ms: Poincaré plot minor axis (milliseconds) - reflects short-term variability (vagal)",
        "  * Normal range: 10-50 ms; <10 ms suggests reduced vagal modulation",
        "- SD2_ms: Poincaré plot major axis (milliseconds) - reflects long-term variability",
        "  * Normal range: 30-100 ms; low values indicate reduced HRV",
        "- SD1_SD2: SD1/SD2 ratio (unitless) - balance of short vs long-term variability",
        "  * Normal range: 0.2-0.5; <0.2 suggests sympathetic dominance",
        "- SampEn: Sample Entropy (unitless, typically 0-2.5)",
        "  * Higher = more complex/irregular; 0.8-1.5 is typical for healthy adults",
        "  * <0.8 suggests reduced complexity; >2.0 may indicate noise or artifact",
        "- DFA_alpha: Detrended Fluctuation Analysis scaling exponent (unitless)",
        "  * Normal range: 0.7-1.0; >1.0 suggests reduced complexity or trending",
        "- SDNN_index_ms: Mean of 5-min SDNN segments (milliseconds)",
        "",
        "=== MEASURED VALUES ===",
    ]
    for name in names:
        v = raw.get(name)
        zv = zscores.get(f"{name}_zscore")
        try:
            v_f = float(v) if v is not None else None
        except (TypeError, ValueError):
            v_f = None
        try:
            z_f = float(zv) if zv is not None else None
        except (TypeError, ValueError):
            z_f = None

        if v_f is not None and z_f is not None:
            lines.append(f"- {name}: {v_f:.4f} (z = {z_f:.2f})")
        elif v_f is not None:
            lines.append(f"- {name}: {v_f:.4f}")
        else:
            lines.append(f"- {name}: Not Available")

    # 圖像像素特徵
    p_mean = _safe_float(img_feats.get("poincare_img_mean"))
    p_std = _safe_float(img_feats.get("poincare_img_std"))
    p_slope = _safe_float(img_feats.get("poincare_row_slope"))

    # === 新增：實際數據量化特徵（避免圖像縮放失真） ===
    poincare_rri_x_min = _safe_float(poincare_data_feats.get("poincare_rri_x_min"))
    poincare_rri_x_max = _safe_float(poincare_data_feats.get("poincare_rri_x_max"))
    poincare_rri_x_mean = _safe_float(poincare_data_feats.get("poincare_rri_x_mean"))
    poincare_rri_x_std = _safe_float(poincare_data_feats.get("poincare_rri_x_std"))
    poincare_rri_y_min = _safe_float(poincare_data_feats.get("poincare_rri_y_min"))
    poincare_rri_y_max = _safe_float(poincare_data_feats.get("poincare_rri_y_max"))
    poincare_rri_y_mean = _safe_float(poincare_data_feats.get("poincare_rri_y_mean"))
    poincare_rri_y_std = _safe_float(poincare_data_feats.get("poincare_rri_y_std"))
    poincare_rri_range_x = _safe_float(poincare_data_feats.get("poincare_rri_range_x"))
    poincare_rri_range_y = _safe_float(poincare_data_feats.get("poincare_rri_range_y"))
    poincare_rri_ratio_xy = _safe_float(poincare_data_feats.get("poincare_rri_ratio_xy"))
    poincare_density_center_x = _safe_float(poincare_data_feats.get("poincare_density_center_x"))
    poincare_density_center_y = _safe_float(poincare_data_feats.get("poincare_density_center_y"))
    poincare_scatter_count = _safe_float(poincare_data_feats.get("poincare_scatter_count"))
    poincare_rri_y_mean = _safe_float(poincare_data_feats.get("poincare_rri_y_mean"))
    poincare_rri_y_std = _safe_float(poincare_data_feats.get("poincare_rri_y_std"))
    poincare_rri_range_x = _safe_float(poincare_data_feats.get("poincare_rri_range_x"))
    poincare_rri_range_y = _safe_float(poincare_data_feats.get("poincare_rri_range_y"))
    poincare_rri_ratio_xy = _safe_float(poincare_data_feats.get("poincare_rri_ratio_xy"))
    poincare_density_center_x = _safe_float(poincare_data_feats.get("poincare_density_center_x"))
    poincare_density_center_y = _safe_float(poincare_data_feats.get("poincare_density_center_y"))
    poincare_scatter_count = _safe_float(poincare_data_feats.get("poincare_scatter_count"))

    # === 非線性指標信賴度加權 (Confidence Weighting, 受 ENABLE_GUARDRAILS 控制) ===
    # 根據數據長度和品質指標，調整對非線性指標的解讀信心
    quality_feats = sample_data.get("quality_features") or {}
    sampen_confidence = _safe_float(quality_feats.get("SampEn_confidence"))
    sampen_reliability = quality_feats.get("SampEn_reliability", "")
    nonlinear_quality = _safe_float(quality_feats.get("nonlinear_quality"))
    
    # 根據 Scatter Count 生成信賴度警告（受 ENABLE_GUARDRAILS 控制）
    confidence_warning = ""
    data_quality_level = "adequate"  # adequate / marginal / insufficient
    sampen_warning = ""
    
    if not ENABLE_GUARDRAILS:
        # Guardrails 關閉：跳過所有可信度檢測，使用預設值
        data_quality_level = "adequate"
        confidence_warning = ""
        sampen_warning = ""
    elif poincare_scatter_count is not None:
        point_count = int(poincare_scatter_count)
        if point_count < 100:
            data_quality_level = "insufficient"
            confidence_warning = (
                f"⚠️ CRITICAL DATA LIMITATION: The Poincaré plot contains only {point_count} points.\n"
                "Nonlinear metrics (SampEn, DFA_alpha) are UNRELIABLE on such short time series.\n"
                "- SampEn requires at least 100-200 points for stable estimation (m=2).\n"
                "- DFA_alpha is highly sensitive to data length.\n"
                "DO NOT interpret high complexity values as definitive cognitive states.\n"
                "Rely primarily on time-domain metrics (MeanHR, RMSSD) for this sample."
            )
        elif point_count < 200:
            data_quality_level = "marginal"
            confidence_warning = (
                f"⚠️ DATA LIMITATION: The Poincaré plot contains only {point_count} points.\n"
                "Nonlinear metrics (SampEn, DFA_alpha) may be UNSTABLE on this short time series.\n"
                "- Consider SampEn with LOW confidence; wide confidence intervals expected.\n"
                "- Cross-validate with time-domain metrics (RMSSD, MeanHR) before concluding.\n"
                "Do not over-interpret high complexity values as definitive flow states."
            )
        elif point_count < 300:
            data_quality_level = "marginal"
            confidence_warning = (
                f"Note: The Poincaré plot contains {point_count} points (borderline adequate).\n"
                "Nonlinear metrics should be interpreted with MODERATE confidence.\n"
                "Corroborate SampEn/DFA findings with MeanHR and time-domain evidence."
            )
        else:
            data_quality_level = "adequate"
            confidence_warning = (
                f"Data quality: {point_count} points (adequate for nonlinear analysis).\n"
                "SampEn and DFA_alpha can be interpreted with reasonable confidence."
            )
        
        # 整合 SampEn 置信度指標（只在 ENABLE_GUARDRAILS 時）
        if sampen_confidence is not None:
            if sampen_confidence < 0.4:
                sampen_warning = (
                    f"⚠️ SampEn confidence: {sampen_confidence:.2f} (LOW - {sampen_reliability}).\n"
                    "SampEn value may be unreliable; prioritize other metrics."
                )
            elif sampen_confidence < 0.7:
                sampen_warning = (
                    f"SampEn confidence: {sampen_confidence:.2f} (MEDIUM - {sampen_reliability}).\n"
                    "Interpret SampEn with caution."
                )
            else:
                sampen_warning = f"SampEn confidence: {sampen_confidence:.2f} (HIGH - {sampen_reliability})."

    lines.extend(
        [
            "",
            "Quantitative summary of Poincaré image (RRn vs RRn+1, 896x896 grayscale):",
            f"- Mean intensity: {p_mean:.4f}" if p_mean is not None else "- Mean intensity: NA",
            f"- Intensity std: {p_std:.4f}" if p_std is not None else "- Intensity std: NA",
            f"- Row-wise mean slope (top→bottom): {p_slope:.4e}" if p_slope is not None else "- Row-wise mean slope: NA",
            "",
            "=== Actual Poincaré Data Quantitative Features (from raw RRI signals, immune to image scaling) ===",
            "X-axis (RRn) Statistics:",
            f"- X-axis min: {poincare_rri_x_min:.2f} ms" if poincare_rri_x_min is not None else "- X-axis min: NA",
            f"- X-axis max: {poincare_rri_x_max:.2f} ms" if poincare_rri_x_max is not None else "- X-axis max: NA",
            f"- X-axis mean: {poincare_rri_x_mean:.2f} ms" if poincare_rri_x_mean is not None else "- X-axis mean: NA",
            f"- X-axis std: {poincare_rri_x_std:.2f} ms" if poincare_rri_x_std is not None else "- X-axis std: NA",
            f"- X-axis range: {poincare_rri_range_x:.2f} ms" if poincare_rri_range_x is not None else "- X-axis range: NA",
            "",
            "Y-axis (RRn+1) Statistics:",
            f"- Y-axis min: {poincare_rri_y_min:.2f} ms" if poincare_rri_y_min is not None else "- Y-axis min: NA",
            f"- Y-axis max: {poincare_rri_y_max:.2f} ms" if poincare_rri_y_max is not None else "- Y-axis max: NA",
            f"- Y-axis mean: {poincare_rri_y_mean:.2f} ms" if poincare_rri_y_mean is not None else "- Y-axis mean: NA",
            f"- Y-axis std: {poincare_rri_y_std:.2f} ms" if poincare_rri_y_std is not None else "- Y-axis std: NA",
            f"- Y-axis range: {poincare_rri_range_y:.2f} ms" if poincare_rri_range_y is not None else "- Y-axis range: NA",
            "",
            "Geometric Characteristics:",
            f"- X/Y range ratio: {poincare_rri_ratio_xy:.4f}" if poincare_rri_ratio_xy is not None else "- X/Y range ratio: NA",
            f"- Density center X: {poincare_density_center_x:.2f} ms" if poincare_density_center_x is not None else "- Density center X: NA",
            f"- Density center Y: {poincare_density_center_y:.2f} ms" if poincare_density_center_y is not None else "- Density center Y: NA",
            f"- Scatter point count: {int(poincare_scatter_count) if poincare_scatter_count is not None else 'NA'}",
            f"- Data quality level: {data_quality_level.upper()}",
        ]
    )
    
    # 添加信賴度警告區塊
    confidence_block = ""
    if confidence_warning or sampen_warning:
        confidence_block = (
            "\n\n=== NONLINEAR METRIC CONFIDENCE ASSESSMENT ===\n"
            + (confidence_warning + "\n" if confidence_warning else "")
            + (sampen_warning if sampen_warning else "")
        )

    user_prompt = (
        "Step 4 - Poincaré geometry and HRV complexity.\n"
        "Analyze morphology and complexity features using your clinical knowledge.\n\n"
        + "\n".join(lines)
        + confidence_block
        + "\n\n"
        "IMPORTANT INTERPRETATION GUIDELINES:\n"
        "1. Check the 'Scatter point count' FIRST to assess data adequacy.\n"
        "2. If data quality is INSUFFICIENT (<100 points), do NOT draw conclusions from SampEn/DFA.\n"
        "3. If data quality is MARGINAL (100-300 points), interpret nonlinear metrics with caution.\n"
        "4. Always cross-validate complexity findings with time-domain metrics (RMSSD, MeanHR).\n"
        "5. High SampEn alone does NOT definitively indicate flow states on short data.\n\n"
        "Provide a comprehensive analysis. Think through all aspects carefully before concluding."
    )

    # 根據數據品質調整 system_prompt 的語氣（受 ENABLE_GUARDRAILS 控制）
    if not ENABLE_GUARDRAILS:
        # Guardrails 關閉：使用無約束的基礎 system_prompt
        system_prompt = (
            "You are evaluating HRV complexity and nonlinear dynamics.\n"
            "Essential constraints:\n"
            "1. Prioritize quantitative Poincaré metrics over image features.\n"
            "2. Provide balanced interpretation of complexity findings."
        )
    elif data_quality_level == "insufficient":
        system_prompt = (
            "You are evaluating HRV complexity and nonlinear dynamics.\n"
            "CRITICAL: This sample has INSUFFICIENT data points for reliable nonlinear analysis.\n"
            "Essential constraints:\n"
            "1. DO NOT interpret SampEn or DFA_alpha values as meaningful.\n"
            "2. Focus on geometric features (SD1/SD2 ratio) and time-domain metrics.\n"
            "3. Explicitly state that nonlinear conclusions cannot be drawn.\n"
            "4. Prioritize quantitative Poincaré metrics over image features."
        )
    elif data_quality_level == "marginal":
        system_prompt = (
            "You are evaluating HRV complexity and nonlinear dynamics.\n"
            "CAUTION: This sample has MARGINAL data points; nonlinear metrics may be unstable.\n"
            "Essential constraints:\n"
            "1. Interpret SampEn and DFA_alpha with LOW to MEDIUM confidence only.\n"
            "2. Require corroboration from MeanHR and RMSSD before concluding.\n"
            "3. Distinguish adaptive flexibility from pathological complexity.\n"
            "4. Prioritize quantitative Poincaré metrics over image features."
        )
    else:
        system_prompt = (
            "You are evaluating HRV complexity and nonlinear dynamics.\n"
            "Data quality is ADEQUATE for nonlinear analysis.\n"
            "Essential constraints:\n"
            "1. Prioritize quantitative Poincaré metrics over image features.\n"
            "2. Distinguish adaptive flexibility from pathological complexity.\n"
            "3. Cross-validate with time-domain metrics for robustness."
        )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=sample_data.get("image"),
        max_new_tokens=INFERENCE_OUTPUT_LENGTH,
    )


def _run_step5_baseline_delta(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step5: Baseline delta / relative-change 特徵摘要（擴展版：含 Delta Z-score）
    
    重要改進：
    - 使用 Delta Z-score（基於個人 Baseline，與 Delta 方向一致）
    - 避免傳統 Z-score 與 Delta 方向矛盾的問題
    - 可透過 ENABLE_DELTA_ZSCORE 開關控制是否使用 Delta Z-score
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH, ENABLE_DELTA_ZSCORE
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH
        ENABLE_DELTA_ZSCORE = getattr(config, 'ENABLE_DELTA_ZSCORE', True)

    delta_features = sample_data.get("delta_features") or {}
    delta_zscores = sample_data.get("delta_zscores") or {}
    
    if not delta_features:
        return (
            "Step 5 - Baseline delta analysis.\n"
            "Status: Missing delta features\n"
            "Notes: Delta features relative to baseline are not available for this trial."
        )

    abs_lines = []
    pct_lines = []
    baseline_lines = []
    zscore_lines = []

    for name, value in delta_features.items():
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        pretty_name = name.replace("Delta_", "")

        if name.endswith("_pct"):
            line = f"- {pretty_name.replace('_pct', '')}: {value_f:.1f}% change"
            pct_lines.append(line)
        elif name.startswith("Baseline_"):
            line = f"- {pretty_name.replace('Baseline_', '')}: {value_f:.3f}"
            baseline_lines.append(line)
        else:
            # 檢查是否有對應的 Delta Z-score（僅在 ENABLE_DELTA_ZSCORE=True 時使用）
            if ENABLE_DELTA_ZSCORE:
                delta_zscore_key = f"Delta_{pretty_name}_zscore"
                delta_z = _safe_float(delta_zscores.get(delta_zscore_key))

                if delta_z is not None:
                    line = f"- {pretty_name}: {value_f:.3f} (Delta z = {delta_z:.2f})"
                    zscore_lines.append(f"- {pretty_name}: Delta z = {delta_z:.2f}")
                else:
                    line = f"- {pretty_name}: {value_f:.3f}"
            else:
                line = f"- {pretty_name}: {value_f:.3f}"
            abs_lines.append(line)

    if not abs_lines:
        abs_lines.append("- Absolute delta metrics were unavailable.")
    if not pct_lines:
        pct_lines.append("- Percentage delta metrics were unavailable.")

    # 構建 Delta Z-score 說明區塊（僅在 ENABLE_DELTA_ZSCORE=True 時）
    zscore_block = ""
    if ENABLE_DELTA_ZSCORE and zscore_lines:
        zscore_block = (
            "\n\nDelta Z-scores (based on individual baseline, direction-consistent with Delta):\n"
            "IMPORTANT: Use Delta Z-scores for interpretation - they are direction-consistent with Delta values.\n"
            "- Delta > 0 with Delta_z > 0 = increase relative to baseline (consistent)\n"
            "- Delta < 0 with Delta_z < 0 = decrease relative to baseline (consistent)\n"
            + "\n".join(zscore_lines)
        )

    # 根據 ENABLE_DELTA_ZSCORE 調整 user_prompt
    if ENABLE_DELTA_ZSCORE:
        user_prompt = (
            "Step 5 - Baseline delta analysis.\n"
            "Analyze the delta HRV metrics relative to baseline using your clinical knowledge.\n\n"
            "Absolute deltas (with Delta Z-scores where available):\n"
            + "\n".join(abs_lines)
            + "\n\nPercentage deltas:\n"
            + "\n".join(pct_lines)
            + zscore_block
            + "\n\n"
            "CRITICAL: When interpreting changes, prioritize Delta Z-scores over traditional Z-scores.\n"
            "Delta Z-scores are calculated from the distribution of Delta values, ensuring consistency:\n"
            "- If Delta > 0 (increased from baseline), Delta_zscore will tend to be positive.\n"
            "- If Delta < 0 (decreased from baseline), Delta_zscore will tend to be negative.\n"
            "This avoids the common confusion where traditional Z-score > 0 but Delta < 0.\n\n"
            "Provide a comprehensive analysis. Think through all aspects carefully before concluding."
        )
        system_prompt = (
            "You are assessing HRV changes relative to individual baseline.\n"
            "Essential constraints:\n"
            "1. PRIORITIZE Delta Z-scores over traditional Z-scores for within-subject comparisons.\n"
            "2. Delta Z-scores are direction-consistent with Delta values (no sign contradictions).\n"
            "3. Consider both absolute changes and percentage deviations.\n"
            "4. Classify changes with clinical reasoning."
        )
    else:
        # ENABLE_DELTA_ZSCORE=False：使用傳統分析方式，不強調 Delta Z-score
        user_prompt = (
            "Step 5 - Baseline delta analysis.\n"
            "Analyze the delta HRV metrics relative to baseline using your clinical knowledge.\n\n"
            "Absolute deltas:\n"
            + "\n".join(abs_lines)
            + "\n\nPercentage deltas:\n"
            + "\n".join(pct_lines)
            + "\n\n"
            "Apply your expertise to describe:\n"
            "- Vagal tone direction (increased/decreased/mixed/unknown)\n"
            "- Arousal direction (increased/decreased/mixed/unknown)\n"
            "- Integration of multiple indicators\n"
            "- Resolution of contradictions if present\n"
            "- Overall shift assessment\n"
            "- Confidence level in your assessment"
        )
        system_prompt = (
            "You are analyzing delta HRV features relative to baseline.\n"
            "Essential constraints:\n"
            "1. Focus on absolute and percentage changes from baseline.\n"
            "2. Consider the direction and magnitude of changes.\n"
            "3. Classify changes with clinical reasoning."
        )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=None,
        max_new_tokens=INFERENCE_OUTPUT_LENGTH,
    )


def _run_step6_within_subject_profile(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step6: 同 subject 跨試次基線比較（within-subject profile）。
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH

    # baseline 欄位名稱規則：<feat>_baseline_mean / <feat>_baseline_std
    baseline_features = {
        k: v
        for k, v in sample_data.items()
        if isinstance(k, str) and (k.endswith("_baseline_mean") or k.endswith("_baseline_std"))
    }

    zscores = sample_data.get("zscore_features") or {}

    feats = ["RMSSD_ms", "SDNN_ms", "MeanHR_bpm", "SampEn", "DFA_alpha"]

    lines = ["Within-subject baseline statistics (per subject across all trials):"]
    for name in feats:
        mean_key = f"{name}_baseline_mean"
        std_key = f"{name}_baseline_std"
        m = _safe_float(baseline_features.get(mean_key))
        s = _safe_float(baseline_features.get(std_key))
        z = _safe_float(zscores.get(f"{name}_zscore"))

        if m is not None and s is not None:
            base_str = f"baseline mean={m:.4f}, baseline SD={s:.4f}"
        elif m is not None:
            base_str = f"baseline mean={m:.4f}, baseline SD=NA"
        else:
            base_str = "baseline NA"

        if z is not None:
            lines.append(f"- {name}: {base_str}, current trial z={z:.2f}")
        else:
            lines.append(f"- {name}: {base_str}, current trial z=NA")

    user_prompt = (
        "Step 6 - Within-subject baseline comparison.\n"
        "Analyze the baseline statistics and z-scores using your clinical knowledge.\n\n"
        + "\n".join(lines)
        + "\n\n"
        "Provide a comprehensive analysis. Think through all aspects carefully before concluding."
    )

    system_prompt = (
        "You are performing individualized HRV baseline profiling.\n"
        "Essential constraints: Prioritize subject-specific baseline over population norms. "
        "Distinguish between transient changes and baseline shifts."
    )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=None,
        max_new_tokens=INFERENCE_OUTPUT_LENGTH,
    )


def _run_step7_multimodal_eeg(base_model, processor, sample_data: Dict, config=None) -> str:
    """
    Step7: EEG 特徵整合（若有）
    """
    if config is None:
        from config import INFERENCE_OUTPUT_LENGTH, ENABLE_EEG_ANALYSIS
    else:
        INFERENCE_OUTPUT_LENGTH = config.INFERENCE_OUTPUT_LENGTH
        ENABLE_EEG_ANALYSIS = config.ENABLE_EEG_ANALYSIS

    eeg_features = sample_data.get("eeg_features") or {}
    if not eeg_features:
        return (
            "Step 7 - Multimodal EEG feature integration.\n"
            "EEG Status: Not provided\n"
            "Findings: EEG-derived features were not provided or EEG processing is disabled for this dataset.\n"
            "Alignment with HRV: Not available\n"
            "Confidence: Low"
        )

    global_feats = {
        name: eeg_features[name]
        for name in eeg_features
        if name.startswith("EEG_global") or name.startswith("EEG_frontal")
    }
    channel_feats = {
        name: eeg_features[name]
        for name in eeg_features
        if "EEG_ch" in name
    }

    def _fmt_lines(items, limit=10):
        lines = []
        count = 0
        for name, val in items.items():
            if val is None:
                continue
            try:
                val_f = float(val)
            except (TypeError, ValueError):
                continue
            lines.append(f"- {name}: {val_f:.4f}")
            count += 1
            if count >= limit:
                break
        if not lines:
            lines.append("- (no stable metrics available)")
        return lines

    global_lines = _fmt_lines(global_feats, limit=10)
    channel_lines = _fmt_lines(channel_feats, limit=12)

    user_prompt = (
        "Step 7 - Multimodal EEG feature integration.\n"
        "Analyze EEG-derived insights and their alignment with HRV findings using your clinical knowledge.\n\n"
        "=== CRITICAL: EEG POWER VALUES ARE DEVICE-SPECIFIC ===\n"
        "⚠️ DO NOT use words like 'extremely high/low' or 'abnormal' for absolute power values!\n"
        "⚠️ This dataset (DREAMER) uses consumer-grade EEG with different scaling than clinical EEG.\n\n"
        "EXPECTED RANGES FOR THIS DATASET (DREAMER, Emotiv EPOC):\n"
        "- Delta power: 10 - 3000+ (units are arbitrary, device-dependent)\n"
        "- Theta power: 5 - 1000+ (units are arbitrary, device-dependent)\n"
        "- Alpha power: 10 - 500+ (units are arbitrary, device-dependent)\n"
        "- Beta power: 5 - 1000+ (units are arbitrary, device-dependent)\n"
        "- Gamma power: 1 - 100+ (units are arbitrary, device-dependent)\n\n"
        "IMPORTANT INTERPRETATION RULES:\n"
        "1. ❌ DO NOT say 'extremely high' or 'abnormally elevated' for ANY absolute power value.\n"
        "2. ✅ DO focus on RATIOS: alpha/beta ratio is the KEY metric for arousal assessment.\n"
        "3. ✅ DO compare RELATIVE differences between channels, not absolute magnitudes.\n"
        "4. ✅ Alpha/Beta ratio >1 → relaxation; ratio <0.5 → alertness/stress.\n"
        "5. High delta in awake subjects: consider possible artifact, but do not call it 'extremely high'.\n\n"
        "Global / regional metrics:\n"
        + "\n".join(global_lines)
        + "\n\nChannel-level excerpts:\n"
        + "\n".join(channel_lines)
        + "\n\n"
        "YOUR TASK:\n"
        "1. Calculate and interpret alpha/beta RATIOS for arousal assessment.\n"
        "2. Look for RELATIVE patterns (which channels show higher/lower values).\n"
        "3. Cross-validate with HRV findings (low alpha/beta + elevated HR + low HRV → stress pattern).\n"
        "4. AVOID absolute judgments like 'extremely high' - these are meaningless without device calibration.\n\n"
        "Provide a balanced analysis focusing on RATIOS and RELATIVE patterns."
    )

    system_prompt = (
        "You are integrating EEG spectral features with HRV analysis.\n"
        "CRITICAL CONSTRAINTS - FOLLOW STRICTLY:\n"
        "1. NEVER use 'extremely high/low', 'abnormally elevated', or similar absolute judgments for EEG power.\n"
        "2. EEG power values are DEVICE-SPECIFIC and CANNOT be compared to clinical norms.\n"
        "3. ONLY interpret RATIOS (alpha/beta) and RELATIVE channel differences.\n"
        "4. Alpha/Beta ratio is the ONLY reliable arousal indicator: >1 = relaxed, <0.5 = alert/stressed.\n"
        "5. If you see high delta values, mention 'possible artifact or device scaling' - not 'extremely high'.\n"
        "6. Assess multimodal convergence: does alpha/beta ratio align with HRV stress indicators?"
    )

    return _run_llm_simple(
        base_model,
        processor,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image=None,
        max_new_tokens=INFERENCE_OUTPUT_LENGTH,
    )
