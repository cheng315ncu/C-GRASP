#!/usr/bin/env python3
"""
LLM + PIKE-RAG 整合版本 - 主執行檔案
重構後的主程式，使用模組化的架構
"""

import torch
import os
import sys
import time
import csv
import re
from typing import Dict, List
from tqdm import tqdm

# 添加 PIKE-RAG 路徑
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 導入 PIKE-RAG 醫療知識檢索模組
from pikerag_medical_integration import (
    create_hrv_analysis_query,
    METADATA_LIST_DELIMITER,
)

# 導入自定義模組
from config import *
from dataset import HRVEmotionDataset
from rag_system import initialize_rag_system
from inference_steps import (
    _run_step1_quality,
    _run_step2_time_domain,
    _run_step3_frequency,
    _run_step4_poincare_complexity,
    _run_step5_baseline_delta,
    _run_step6_within_subject_profile,
    _run_step7_multimodal_eeg,
)
from utils import (
    clean_catch,
    enforce_structured_output,
    detect_zscore_conflicts,
    build_clinical_summary,
    validate_zscore_usage,
    validate_output_format,
    _build_demographics_text,
    _safe_float,
    _parse_metadata_list,
)
from system_prompt import system_prompt_template
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoModel


def main():
    """
    主執行函數（多模態分步推理 + 可選 PIKE-RAG + 可選 EEG）

    工作流程:
    1. 載入 LLM 模型
    2. 初始化 RAG 系統（如果啟用）
    3. 載入 HRV 數據集
    4. 對每個樣本:
       a. 檢索相關臨床知識（如果 RAG 啟用）
       b. Step1: 訊號品質與前處理審查
       c. Step2: 時域 HRV 分析
       d. Step3: 頻域 / 呼吸特徵分析
       e. Step4: Poincaré 幾何與複雜度
       f. Step5: Baseline Delta 差異特徵分析
       g. Step6: Within-subject 基線分析
       h. Step7: EEG 多模態分析（如果啟用且資料可用）
       i. Step8: 使用統合模板產出最終報告（整合所有前序步驟）
    """
    model_name = MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID.split(os.sep)[-1]
    print("="*80)
    print(f"{model_name} 多模態 HRV 分步推理系統啟動")
    print("="*80)

    # 允許使用者從命令列決定是否啟用 RAG
    # 使用方式示例：
    #   python main.py --rag       # 強制啟用 RAG
    #   python main.py --no-rag    # 強制關閉 RAG
    #   （未給參數時沿用檔案內 ENABLE_RAG 預設值）
    rag_flag = None
    for arg in sys.argv[1:]:
        if arg == "--rag":
            rag_flag = True
        elif arg == "--no-rag":
            rag_flag = False

    # 步驟 1: 載入模型
    print("\n[步驟 1/4] 正在載入基礎模型...")
    base_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
    )

    # base_model = AutoModel.from_pretrained(
    #     MODEL_ID,
    #     dtype=torch.bfloat16,
    #     device_map="auto",
    #     quantization_config=quantization_config,
    #     trust_remote_code=True,
    #     # attn_implementation="flash_attention_2",
    # )
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    print("✓ 模型載入完成")

    # 步驟 2: 初始化 RAG 系統
    print("\n[步驟 2/4] 初始化 RAG 知識檢索系統...")
    rag_retriever = initialize_rag_system(enable=rag_flag)

    # 步驟 3: 載入數據集
    print(f"\n[步驟 3/4] 正在從 {CSV_PATH} 載入數據集...")
    dreamer_dataset = HRVEmotionDataset(csv_path=CSV_PATH)
    print(f"✓ 數據集載入完成，共 {len(dreamer_dataset)} 個樣本")

    # 確保輸出目錄存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 步驟 4: 分析樣本
    print(f"\n[步驟 4/4] 開始分析樣本...")
    print("="*80 + "\n")

    start_index = 0 # Start index for testing
    end_index = int(TRAILS * SUBJECTS) if TEST else int(len(dreamer_dataset)) # for testing

    analysis_results = []
    timing_results = []  # 統一收集所有樣本的時間數據

    # 追蹤前一個 subject，用於檢測 subject 變化
    previous_subject = None

    # 遍歷數據集進行分析
    for i in tqdm(range(start_index, end_index), desc="分析進度"):
        sample_data = dreamer_dataset[i]
        # 確保 Z-score 皆為 float，避免出現 -(-0.87) 等格式
        zscore_features = sample_data.get("zscore_features") or {}
        sample_data["zscore_features"] = {
            key: _safe_float(value) for key, value in zscore_features.items()
        }
        # 格式化 subject 和 trial 為 S01, T01 格式
        subject_raw = sample_data.get('subject')
        trial_raw = sample_data.get('trial')

        # 將數字格式化為 S01, T01 等格式
        if isinstance(subject_raw, (int, float)):
            subject = f"S{int(subject_raw):02d}"
        else:
            subject = str(subject_raw) if subject_raw is not None else 'unknown'

        if isinstance(trial_raw, (int, float)):
            trial = f"T{int(trial_raw):02d}"
        else:
            trial = str(trial_raw) if trial_raw is not None else 'unknown'

        # 檢測 subject 變化：如果 subject 改變且不是第一個樣本，清理內存
        if previous_subject is not None and subject != previous_subject:
            print(f"\n[{previous_subject}] ✓ 所有 trials 處理完成，清理 CUDA 緩存和內存...")
            clean_catch()
            print(f"[{previous_subject}] ✓ 清理完成\n")

        previous_subject = subject

        subject_dir = os.path.join(OUTPUT_DIR, subject)
        os.makedirs(subject_dir, exist_ok=True)

        output_file_path = os.path.join(subject_dir, f"{trial}.txt")

        # 如果報告已存在，則跳過，節省時間
        if os.path.exists(output_file_path):
            print(f"[{subject} {trial}] ⏭️  跳過（報告已存在）")
            continue

        # 初始化時間記錄字典
        print(f"\n[{subject} {trial}] {'='*60}")
        print(f"[{subject} {trial}] 開始處理樣本 {i+1}/{len(dreamer_dataset)}")
        timing_data = {
            'rag_retrieval': 0.0,
            'step1_quality': 0.0,
            'step2_time_domain': 0.0,
            'step3_frequency': 0.0,
            'step4_poincare': 0.0,
            'step5_baseline_delta': 0.0,
            'step6_within_subject': 0.0,
            'step7_eeg': 0.0,
            'step8_integration': 0.0,
            'total': 0.0
        }
        overall_start_time = time.time()

        # ★★★ RAG 增強：檢索相關臨床知識（預處理步驟）★★★
        clinical_knowledge_text = ""
        retrieved_docs = None  # 初始化，避免未定義錯誤
        clinical_summary = ""
        if rag_retriever is not None:
            try:
                print(f"\n[{subject} {trial}] [RAG] 開始檢索臨床知識...")
                rag_start_time = time.time()
                # 1. 根據 HRV 數據創建檢索查詢（使用批判性方法學視角）
                query = create_hrv_analysis_query(
                    sample_data,
                    use_critical_approach=True,
                    verbosity=RAG_QUERY_VERBOSITY
                )

                # 2. 從向量數據庫檢索相關文件
                retrieved_docs = rag_retriever.retrieve_relevant_knowledge(
                    query=query,
                    k=RAG_TOP_K,
                    score_threshold=RAG_SCORE_THRESHOLD
                )

                # 3. 格式化檢索到的知識
                if retrieved_docs:
                    print(f"[{subject} {trial}] [RAG] ✓ 檢索到 {len(retrieved_docs)} 個相關文件")
                    clinical_summary = build_clinical_summary(retrieved_docs)
                    for idx, doc_dict in enumerate(retrieved_docs):
                        metadata = doc_dict['metadata']
                        source = metadata.get('source_file', metadata.get('source', '未知來源'))
                        source_filename = os.path.basename(source) if source and source != '未知來源' else str(source)
                        adjusted_score = doc_dict['score']
                        raw_score = _safe_float(metadata.get('raw_score'))
                        domain_weight = _safe_float(metadata.get('domain_weight'))
                        score_repr = f"{adjusted_score:.3f}"
                        if raw_score is not None and domain_weight is not None:
                            score_repr += f" (原始 {raw_score:.3f} × 加權 {domain_weight:.2f})"
                        elif raw_score is not None:
                            score_repr += f" (原始 {raw_score:.3f})"

                        topic_labels = _parse_metadata_list(metadata, 'hrv_topics')[:2]
                        metric_labels = _parse_metadata_list(metadata, 'hrv_metrics')[:2]
                        info_chunks = []
                        if topic_labels:
                            info_chunks.append(f"主題: {', '.join(topic_labels)}")
                        if metric_labels:
                            info_chunks.append(f"指標: {', '.join(metric_labels)}")
                        extra_info = f" | {' | '.join(info_chunks)}" if info_chunks else ""
                        print(f"  - [{idx+1}] {source_filename} (相關性: {score_repr}){extra_info}")

                        key_points = _parse_metadata_list(metadata, 'key_points')
                        if key_points:
                            print(f"    -> 關鍵觀察: {key_points[0]}")
                        content_preview = doc_dict['content'][:80].replace('\n', ' ')
                        if content_preview:
                            print(f"    -> 節錄: {content_preview}...")

                    clinical_knowledge_text = rag_retriever.format_retrieved_knowledge(
                        retrieved_docs,
                        max_length=6144
                    )
                    if clinical_summary:
                        clinical_knowledge_text = f"{clinical_summary}\n\n{clinical_knowledge_text}"
                else:
                    print(f"[{subject} {trial}] [RAG] ⚠️ 未檢索到符合閾值的文件")
                    clinical_knowledge_text = "（未檢索到高度相關的臨床知識...）"

                timing_data['rag_retrieval'] = time.time() - rag_start_time
                print(f"[{subject} {trial}] [RAG] ✓ 完成 ({timing_data['rag_retrieval']:.2f} 秒)")

            except Exception as e:
                print(f"[{subject} {trial}] [RAG] ✗ 檢索失敗: {e}")
                clinical_knowledge_text = "（RAG 檢索暫時不可用）"
        else:
            print(f"[{subject} {trial}] [RAG] 跳過（RAG 功能未啟用）")
            clinical_knowledge_text = "（RAG 功能未啟用）"
            clinical_summary = ""

        # ---------- Step1–Step7：子報告 ----------
        print(f"[{subject} {trial}] [Step 1] 開始訊號品質分析...")
        step_start_time = time.time()
        step1_raw = _run_step1_quality(base_model, processor, sample_data)
        step1_text, step1_parsed = enforce_structured_output(step1_raw, "Step 1")
        timing_data['step1_quality'] = time.time() - step_start_time
        print(f"[{subject} {trial}] [Step 1] ✓ 完成 ({timing_data['step1_quality']:.2f} 秒)")

        print(f"[{subject} {trial}] [Step 2] 開始時域 HRV 分析...")
        step_start_time = time.time()
        step2_raw = _run_step2_time_domain(base_model, processor, sample_data)
        step2_text, step2_parsed = enforce_structured_output(step2_raw, "Step 2")
        timing_data['step2_time_domain'] = time.time() - step_start_time
        print(f"[{subject} {trial}] [Step 2] ✓ 完成 ({timing_data['step2_time_domain']:.2f} 秒)")

        print(f"[{subject} {trial}] [Step 3] 開始頻域 HRV 分析...")
        step_start_time = time.time()
        step3_raw = _run_step3_frequency(base_model, processor, sample_data)
        step3_text, step3_parsed = enforce_structured_output(step3_raw, "Step 3")
        timing_data['step3_frequency'] = time.time() - step_start_time
        print(f"[{subject} {trial}] [Step 3] ✓ 完成 ({timing_data['step3_frequency']:.2f} 秒)")

        print(f"[{subject} {trial}] [Step 4] 開始 Poincaré/複雜度分析...")
        step_start_time = time.time()
        step4_raw = _run_step4_poincare_complexity(base_model, processor, sample_data)
        step4_text, step4_parsed = enforce_structured_output(step4_raw, "Step 4")
        timing_data['step4_poincare'] = time.time() - step_start_time
        print(f"[{subject} {trial}] [Step 4] ✓ 完成 ({timing_data['step4_poincare']:.2f} 秒)")

        print(f"[{subject} {trial}] [Step 5] 開始 Baseline Delta 分析...")
        step_start_time = time.time()
        has_delta_features = bool(sample_data.get("delta_features"))
        step5_raw = _run_step5_baseline_delta(base_model, processor, sample_data)
        step5_text, step5_parsed = enforce_structured_output(step5_raw, "Step 5")
        timing_data['step5_baseline_delta'] = time.time() - step_start_time
        if not has_delta_features:
            print(f"[{subject} {trial}] [Step 5] ⚠️  跳過（無 Delta 特徵數據）")
        else:
            print(f"[{subject} {trial}] [Step 5] ✓ 完成 ({timing_data['step5_baseline_delta']:.2f} 秒)")

        print(f"[{subject} {trial}] [Step 6] 開始 Within-subject 基線分析...")
        step_start_time = time.time()
        step6_raw = _run_step6_within_subject_profile(base_model, processor, sample_data)
        step6_text, step6_parsed = enforce_structured_output(step6_raw, "Step 6")
        timing_data['step6_within_subject'] = time.time() - step_start_time
        print(f"[{subject} {trial}] [Step 6] ✓ 完成 ({timing_data['step6_within_subject']:.2f} 秒)")

        print(f"[{subject} {trial}] [Step 7] 開始 EEG 多模態分析...")
        step_start_time = time.time()
        eeg_features = sample_data.get("eeg_features") or {}
        has_eeg_features = any(
            v is not None
            for v in eeg_features.values()
        )
        if ENABLE_EEG_ANALYSIS:
            if has_eeg_features:
                step7_raw = _run_step7_multimodal_eeg(base_model, processor, sample_data)
            else:
                step7_raw = (
                    "Step 7 - Multimodal EEG feature integration.\n"
                    "EEG Status: Not provided\n"
                    "Findings: EEG analysis enabled but no EEG-derived features were found for this sample.\n"
                    "Alignment with HRV: Not available\n"
                    "Confidence: Low"
                )
                print(f"[{subject} {trial}] [Step 7] ⚠️ EEG 分析已啟用但未找到特徵")
        else:
            step7_raw = (
                "Step 7 - Multimodal EEG feature integration.\n"
                "EEG Status: Disabled\n"
                "Findings: EEG analysis disabled (ENABLE_EEG_ANALYSIS=False).\n"
                "Alignment with HRV: Not available\n"
                "Confidence: Low"
            )
            print(f"[{subject} {trial}] [Step 7] 跳過（EEG 分析未啟用）")

        step7_text, step7_parsed = enforce_structured_output(step7_raw, "Step 7")
        timing_data['step7_eeg'] = time.time() - step_start_time
        print(f"[{subject} {trial}] [Step 7] ✓ 完成 ({timing_data['step7_eeg']:.2f} 秒)")

        rag_summary_text = clinical_summary or clinical_knowledge_text
        step2_metrics = (step2_parsed or {}).get("metrics_zscores") if isinstance(step2_parsed, dict) else {}
        step6_metrics = (step6_parsed or {}).get("z_scores") if isinstance(step6_parsed, dict) else {}
        if not isinstance(step2_metrics, dict):
            step2_metrics = {}
        if not isinstance(step6_metrics, dict):
            step6_metrics = {}
        data_conflicts: List[str] = []
        
        # 檢測傳統 Z-score 與 Delta 之間的矛盾（僅在 ENABLE_DELTA_ZSCORE=True 時）
        delta_features = sample_data.get("delta_features") or {}
        delta_zscores = sample_data.get("delta_zscores") or {}
        zscore_features = sample_data.get("zscore_features") or {}
        
        if ENABLE_DELTA_ZSCORE:
            # 檢查重要特徵的 Z-score vs Delta 一致性
            important_features = ["RMSSD_ms", "SDNN_ms", "MeanHR_bpm", "SampEn", "DFA_alpha"]
            for feat in important_features:
                delta_key = f"Delta_{feat}"
                trad_zscore_key = f"{feat}_zscore"
                delta_zscore_key = f"Delta_{feat}_zscore"
                
                delta_val = _safe_float(delta_features.get(delta_key))
                trad_z = _safe_float(zscore_features.get(trad_zscore_key))
                delta_z = _safe_float(delta_zscores.get(delta_zscore_key))
                
                # 檢查傳統 Z-score 與 Delta 方向是否矛盾
                if delta_val is not None and trad_z is not None:
                    # 如果 Delta > 0 但傳統 Z < 0，或 Delta < 0 但傳統 Z > 0
                    if (delta_val > 0 and trad_z < -0.5) or (delta_val < 0 and trad_z > 0.5):
                        # 這是預期中的矛盾，建議使用 Delta Z-score
                        if delta_z is not None:
                            data_conflicts.append(
                                f"Z-SCORE_DELTA_MISMATCH ({feat}): Traditional z={trad_z:.2f} vs Delta={delta_val:.2f}. "
                                f"Recommend using Delta_zscore={delta_z:.2f} for consistency."
                            )
                        else:
                            data_conflicts.append(
                                f"Z-SCORE_DELTA_MISMATCH ({feat}): Traditional z={trad_z:.2f} vs Delta={delta_val:.2f}. "
                                f"Use Delta value for interpretation."
                            )
        
        # 原有的 Step 2 vs Step 6 衝突檢測
        if step2_metrics and step6_metrics:
            conflict_keys = detect_zscore_conflicts(step2_metrics, step6_metrics)
            if conflict_keys:
                data_conflicts.append(
                    "DATA_CONFLICT: Step 2 and Step 6 reported opposite z-score signs for "
                    + ", ".join(conflict_keys)
                )

        # ---------- Step8：最終統合推理（整合所有前序步驟） ----------
        print(f"[{subject} {trial}] [Step 8] 開始統合推理（整合所有步驟）...")
        step8_start_time = time.time()
        # 1. 建立個體背景資訊
        demographics_text = _build_demographics_text(sample_data)

        # 2. 建立 HRV 特徵文本，結合原始值與 Z-score
        # 若 ENABLE_DELTA_ZSCORE=True，同時顯示傳統和 Delta Z-score；否則只顯示傳統 Z-score
        hrv_features_text = "Heart Rate Variability (HRV) Features:\n"
        delta_zscores = sample_data.get("delta_zscores") or {}
        
        for name, value in sample_data["raw_features"].items():
            if value is not None:
                try:
                    value_float = float(value)
                    # 檢查是否有對應的傳統 Z-score
                    zscore_key = f"{name}_zscore"
                    z_str = ""
                    delta_z_str = ""
                    
                    if zscore_key in sample_data["zscore_features"]:
                        z_value = sample_data["zscore_features"][zscore_key]
                        if z_value is not None:
                            try:
                                z_value_float = float(z_value)
                                z_str = f"z = {z_value_float:.2f}"
                            except (ValueError, TypeError):
                                pass
                    
                    # 僅在 ENABLE_DELTA_ZSCORE=True 時檢查 Delta Z-score
                    if ENABLE_DELTA_ZSCORE:
                        delta_zscore_key = f"Delta_{name}_zscore"
                        if delta_zscore_key in delta_zscores:
                            delta_z_value = delta_zscores[delta_zscore_key]
                            if delta_z_value is not None:
                                try:
                                    delta_z_float = float(delta_z_value)
                                    delta_z_str = f"Δz = {delta_z_float:.2f}"
                                except (ValueError, TypeError):
                                    pass
                    
                    # 組合顯示
                    if z_str and delta_z_str:
                        hrv_features_text += f"- {name}: {value_float:.4f} ({z_str}, {delta_z_str})\n"
                    elif z_str:
                        hrv_features_text += f"- {name}: {value_float:.4f} ({z_str})\n"
                    elif delta_z_str:
                        hrv_features_text += f"- {name}: {value_float:.4f} ({delta_z_str})\n"
                    else:
                        hrv_features_text += f"- {name}: {value_float:.4f}\n"
                except (ValueError, TypeError):
                    hrv_features_text += f"- {name}: {value}\n"
            else:
                hrv_features_text += f"- {name}: Not Available\n"
        
        # 添加 Z-score 解釋說明（根據 ENABLE_DELTA_ZSCORE 調整）
        if ENABLE_DELTA_ZSCORE:
            hrv_features_text += (
                "\nZ-score Legend:\n"
                "- z = Traditional Z-score (compared to population mean)\n"
                "- Δz = Delta Z-score (based on individual baseline, direction-consistent with Delta)\n"
                "- Use Δz for within-subject change interpretation to avoid sign contradictions\n"
            )
        else:
            hrv_features_text += (
                "\nZ-score Legend:\n"
                "- z = Traditional Z-score (compared to population mean)\n"
            )

        # 3. 組合 Step1–Step8 子報告摘要
        step_summaries_text = (
            "Step 1 - Signal Quality Report:\n"
            + step1_text
            + "\n\nStep 2 - Time-domain HRV Report:\n"
            + step2_text
            + "\n\nStep 3 - Frequency-domain HRV Report:\n"
            + step3_text
            + "\n\nStep 4 - Poincaré / Complexity Report:\n"
            + step4_text
            + "\n\nStep 5 - Baseline Delta Report:\n"
            + step5_text
            + "\n\nStep 6 - Within-subject Baseline Report:\n"
            + step6_text
            + "\n\nStep 7 - EEG / Multimodal Report:\n"
            + step7_text
            + "\n\nStep 8 - Clinical Knowledge / RAG Summary:\n"
            + (rag_summary_text or "No external clinical knowledge was retrieved for this case.")
        )
        conflict_block = ""
        if data_conflicts:
            conflict_lines = "\n".join(f"- {msg}" for msg in data_conflicts)
            conflict_block = f"DATA_QUALITY_WARNINGS:\n{conflict_lines}\n\n"

        # 4. 構建 Step8（最終統合推理）專用的 system & user prompt
        system_prompt = system_prompt_template.format(
            clinical_knowledge=clinical_knowledge_text
        )

        user_prompt_text = (
            f"{demographics_text}\n\n"
            f"{hrv_features_text}\n"
            "Below are intermediate analyses from previous steps. "
            "Use them to complement your reasoning, but critically evaluate them. "
            "You may reference them in any order that makes sense for your analysis. "
            "If they contain contradictions or uncertainties, acknowledge them and apply your clinical judgment:\n\n"
            f"{step_summaries_text}\n\n"
            f"{conflict_block}"
            "**Concise chain-of-thought (keep it short):**\n"
            "- Start with data quality and any DATA_QUALITY_WARNINGS; gate decisions if critical.\n"
            "- Integrate Steps 1-7 into 3-5 sentences that drive valence/arousal + learning-state classification; do not restate full step texts.\n"
            "- Reconcile conflicts explicitly (e.g., Step2 vs Step6 z-scores), and note missing/low-confidence inputs.\n"
            "- Keep the <think> section under ~150 words to control latency while ensuring completeness.\n\n"
            "**Literature Citation Guidelines:**\n"
            "- When citing retrieved clinical literature, clearly specify which indicators each citation supports.\n"
            "- If literature conclusions conflict with your analysis based on individual Z-scores, acknowledge the discrepancy and explain your reasoning.\n\n"
            "**OUTPUT REQUIREMENTS:**\n"
            "You MUST provide your answer in the exact format specified in the system instructions:\n"
            "1. Use <think> and <answer> tags as required\n"
            "2. In the <answer> section, provide all 6 required fields with clear, explicit values\n"
            "3. At the very end, you MUST append exactly these three lines:\n"
            "   State: [HVHA|HVLA|LVHA|LVLA]\n"
            "   Learning: [Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused]\n"
            "   Confidence: [High|Medium|Low]\n"
            "4. Each value must be exactly one of the specified options (no variations, no additional text)\n\n"
            "Now perform your clinical reasoning and provide the final assessment with explicit, extractable answers."
        )

        poincare_image = sample_data["image"]

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt_text}]},
        ]

        if poincare_image:
            messages[1]["content"].append({"type": "image", "image": poincare_image})

        # --- Step8（最終統合推理）模型推理與輸出 ---
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(base_model.device)

        input_len = inputs["input_ids"].shape[-1]

        # 準備額外終止符號，避免無限生成
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
            from utils import SafeLogitsProcessor, LogitsProcessorList
            logits_processor = LogitsProcessorList([SafeLogitsProcessor()])
            generation = base_model.generate(
                **inputs,
                max_new_tokens=SUMMARY_OUTPUT_LENGTH,
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                # no_repeat_ngram_size=3,  # 已移除以避免 Unicode 亂碼
                repetition_penalty=1.05,  # 略微增加重複懲罰以減少重複
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=eos_ids,
                logits_processor=logits_processor,
            )
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)
        timing_data['step8_integration'] = time.time() - step8_start_time
        print(f"[{subject} {trial}] [Step 8] ✓ 完成 ({timing_data['step8_integration']:.2f} 秒)")

        # ★★★ 數值幻覺檢測與修復（優先處理）★★★
        def detect_and_flag_numerical_hallucinations(text: str, sample_data: Dict) -> str:
            """
            檢測並標記數值幻覺，在文本中添加警告標記。
            不直接修改數值（避免誤判），而是標記可疑數值。
            """
            warnings = []

            # 提取輸入數據中的實際數值（用於驗證）
            raw_features = sample_data.get("raw_features", {})
            zscore_features = sample_data.get("zscore_features", {})

            # 定義生理合理性範圍
            PHYSIOLOGICAL_RANGES = {
                'RRI': (300, 2000),  # ms
                'RMSSD': (5, 200),   # ms
                'SDNN': (10, 300),   # ms
                'MeanHR': (40, 200), # bpm
                'SampEn': (0.1, 3.0),
                'DFA_alpha': (0.3, 2.0),
            }

            # 檢測科學記號濫用（過大的指數）
            scientific_pattern = r'(\d+\.?\d*)\s*[eE]\s*[+\-]?\s*(\d+)'
            for match in re.finditer(scientific_pattern, text):
                base = float(match.group(1))
                exp = int(match.group(2))
                value = base * (10 ** exp)

                # 檢查是否為不合理的科學記號
                if abs(exp) > 6:  # 指數絕對值 > 6 通常不合理
                    warnings.append(f"⚠️ 可疑科學記號: {match.group(0)} (值 = {value:.2e})")
                elif value > 1e6 or value < -1e6:  # 絕對值 > 1e6
                    warnings.append(f"⚠️ 可疑大數值: {match.group(0)} (值 = {value:.2e})")

            # 檢測明顯不合理的數值（不使用科學記號的大數）
            # MeanHR > 200 bpm 或 < 40 bpm
            hr_pattern = r'(?:MeanHR|Mean HR|heart rate|HR)[\s:]*(\d{3,})'
            for match in re.finditer(hr_pattern, text, re.IGNORECASE):
                hr_value = int(match.group(1))
                if hr_value > 200 or hr_value < 40:
                    warnings.append(f"⚠️ 不合理的 HR 值: {hr_value} bpm (應在 40-200 bpm)")

            # 檢測 RRI 範圍異常
            rri_range_pattern = r'RRI\s+range[\s:]*(\d+\.?\d*)\s*ms'
            for match in re.finditer(rri_range_pattern, text, re.IGNORECASE):
                rri_value = float(match.group(1))
                if rri_value > 5000:  # RRI range > 5 秒不合理
                    warnings.append(f"⚠️ 不合理的 RRI range: {rri_value} ms (應 < 5000 ms)")

            # 檢測 Z-score 格式錯誤（缺小數點）
            zscore_bad_pattern = r'z\s*[=:]\s*[-\+]?(\d{2,})\b'
            for match in re.finditer(zscore_bad_pattern, text, re.IGNORECASE):
                z_value = match.group(1)
                if len(z_value) >= 2 and not '.' in z_value:  # 兩位數以上且沒有小數點
                    warnings.append(f"⚠️ Z-score 格式錯誤: z = {match.group(0)} (應包含小數點，如 z = -0.59)")

            # 檢測使用逗號作為小數點
            comma_decimal_pattern = r'\d+,\d+'
            for match in re.finditer(comma_decimal_pattern, text):
                warnings.append(f"⚠️ 使用逗號作為小數點: {match.group(0)} (應使用點號，如 0.99)")

            # 如果有警告，在文本開頭添加警告區塊
            if warnings:
                warning_block = "\n".join([f"[數值驗證警告] {w}" for w in warnings])
                text = f"{warning_block}\n\n{text}"
                print(f"[{subject} {trial}] ⚠️  檢測到 {len(warnings)} 個數值問題")

            return text

        # 在清理之前先檢測數值幻覺
        decoded = detect_and_flag_numerical_hallucinations(decoded, sample_data)

        # ★★★ 清理異常輸出：移除 HTML 標籤、重複文本、JSON 格式等 ★★★

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
                matches = list(re.finditer(pattern, text))
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
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    if match.start() < earliest_pos:
                        earliest_pos = match.start()

            return earliest_pos if earliest_pos < len(text) else -1

        # 0.5. 檢測並截斷異常的單字符重複模式（如 >>>>>, <<<<<, (((((, ))))), ::::: 等）
        def detect_single_char_repetition(text: str) -> int:
            """檢測連續重複的單個字符或符號組合，返回異常開始位置"""
            # 檢測連續重複的單個字符（至少10個連續相同字符）
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
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    if match.start() < earliest_pos:
                        earliest_pos = match.start()

            # 檢測異常的符號組合模式（如 ><><><, ()()(), :;:; 等）
            combo_patterns = [
                r'(><){5,}',      # ><><><><><
                r'(\(\)){5,}',    # ()()()()()
                r'(\[\]){5,}',    # [][][][][]
                r'(:;){5,}',      # :;:;:;:;:;
                r'(::){5,}',      # ::::::
            ]

            for pattern in combo_patterns:
                matches = list(re.finditer(pattern, text))
                for match in matches:
                    if match.start() < earliest_pos:
                        earliest_pos = match.start()

            return earliest_pos if earliest_pos < len(text) else -1

        def detect_character_repetition(text: str, min_pattern_len: int = 3, min_repeats: int = 5) -> int:
            """
            檢測連續重複的字符模式（如 'allowallowallow', 'lawlawlaw'）。
            返回異常開始位置，若未檢測到則返回 -1。
            """
            earliest_pos = len(text)
            # 檢測重複的短字符串模式
            for pattern_len in range(min_pattern_len, 20):
                pattern = rf'(.{{{pattern_len}}})\1{{{min_repeats},}}'
                for match in re.finditer(pattern, text):
                    if match.start() < earliest_pos:
                        earliest_pos = match.start()
            return earliest_pos if earliest_pos < len(text) else -1

        # 先檢測單字符重複（更早截斷）
        single_char_pos = detect_single_char_repetition(decoded)
        if single_char_pos >= 0:
            decoded = decoded[:single_char_pos].rstrip()
            print(f"[{subject} {trial}] ⚠️  檢測到單字符重複模式，已在位置 {single_char_pos} 截斷")

        # 再檢測字符級別的重複模式
        char_repeat_pos = detect_character_repetition(decoded)
        if char_repeat_pos > 0:
            decoded = decoded[:char_repeat_pos].rstrip()
            print(f"[{subject} {trial}] ⚠️  檢測到字符重複模式，已在位置 {char_repeat_pos} 截斷")

        # 1. 移除 JSON 代碼塊標記（但保留內容）
        decoded = re.sub(r'```json\s*', '', decoded, flags=re.IGNORECASE)
        decoded = re.sub(r'```\s*', '', decoded)

        # 2. 移除 HTML 標籤（但保留我們需要的 XML 標籤如 <answer>, <think> 等）
        # 先標記需要保留的標籤
        preserved_tags = ['answer', 'think', 'redacted_reasoning']
        for tag in preserved_tags:
            decoded = re.sub(f'<{tag}[^>]*>', f'<PRESERVE_{tag}>', decoded, flags=re.IGNORECASE)
            decoded = re.sub(f'</{tag}[^>]*>', f'</PRESERVE_{tag}>', decoded, flags=re.IGNORECASE)

        # 移除所有其他 HTML/XML 標籤
        decoded = re.sub(r'<[^>]+>', '', decoded)

        # 恢復保留的標籤
        for tag in preserved_tags:
            decoded = decoded.replace(f'<PRESERVE_{tag}>', f'<{tag}>')
            decoded = decoded.replace(f'</PRESERVE_{tag}>', f'</{tag}>')

        # 3. 檢測並移除重複的異常文本模式（增強版：包含字符級別重複檢測）
        def remove_repetitive_text(text: str) -> str:
            """移除重複出現的異常文本模式"""
            # 檢測連續重複的單詞（如 "executedterminating executedterminating ..."）
            # 匹配連續出現 3 次以上的相同單詞
            pattern = r'\b(\w+)(?:\s+\1){2,}\b'
            while re.search(pattern, text):
                text = re.sub(pattern, r'\1', text)

            # 檢測字符級別的重複（如 "allowallowallow", "lawlawlaw"）
            # 匹配至少5個字符的重複模式，重複3次以上
            char_repeat_pattern = r'(\w{5,}?)(\1{2,})'
            def remove_char_repeat(m):
                return m.group(1)  # 只保留第一次出現
            text = re.sub(char_repeat_pattern, remove_char_repeat, text)

            # 檢測特定的異常重複模式
            abnormal_patterns = [
                r'executedterminating[^\s]*\s*',
                r'verification[^\s]*\s*',
                r'terminating[^\s]*\s*',
                r'allow\s+allow',  # 檢測 "allow allow" 重複
                r'law\s+law',  # 檢測 "law law" 重複
                r'LawnLawnLawn',  # 檢測連續的 "LawnLawnLawn"
            ]
            for pattern in abnormal_patterns:
                # 如果該模式連續出現超過 3 次，移除後續重複
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                if len(matches) > 3:
                    # 保留前 2 次出現，移除後續
                    last_pos = matches[1].end() if len(matches) > 1 else 0
                    if last_pos > 0:
                        # 找到最後一次正常出現的位置後，移除所有後續重複
                        remaining = text[last_pos:]
                        for p in abnormal_patterns:
                            remaining = re.sub(f'({p}){{3,}}', '', remaining, flags=re.IGNORECASE)
                        text = text[:last_pos] + remaining

            return text

        decoded = remove_repetitive_text(decoded)

        # 4. 強力截斷：在檢測到異常重複模式時立即截斷
        # 檢測 "executedterminating" 等異常模式的首次出現位置
        abnormal_markers = [
            r'executedterminating',
            r'Termination Sequence Executed',
            r'verifier_result',
            r'Verified\.verified\.verified',
            r'allow\s+allow\s+allow',  # 檢測連續的 "allow allow allow"
            r'law\s+law\s+law',  # 檢測連續的 "law law law"
            r'LawnLawnLawn',  # 檢測連續的 "LawnLawnLawn"
        ]
        earliest_abnormal_pos = len(decoded)
        for marker in abnormal_markers:
            match = re.search(marker, decoded, re.IGNORECASE)
            if match and match.start() < earliest_abnormal_pos:
                earliest_abnormal_pos = match.start()

        # 如果找到異常標記，在該位置截斷
        if earliest_abnormal_pos < len(decoded):
            decoded = decoded[:earliest_abnormal_pos].rstrip()
            print(f"[{subject} {trial}] ⚠️  檢測到異常輸出模式，已在位置 {earliest_abnormal_pos} 截斷")

        # 5. 移除驗證相關的重複文本（在截斷後再次清理）
        verification_patterns = [
            r'verifier_result[^\n]*',
            r'Verified\.verified\.verified[^\n]*',
            r'verification[^\s]*\s+passed[^\n]*',
            r'Termination Sequence Executed[^\n]*',
            r'executedterminating[^\s]*',
        ]
        for pattern in verification_patterns:
            decoded = re.sub(pattern, '', decoded, flags=re.IGNORECASE)

        # 6. 移除多餘的空白行（超過 2 個連續換行）
        decoded = re.sub(r'\n{3,}', '\n\n', decoded)

        # 7. 移除開頭和結尾的空白
        decoded = decoded.strip()

        # 7.5. 修復轉義字符和正則表達式模式問題
        # 移除 Markdown 轉義字符（但保留實際需要的轉義）
        decoded = re.sub(r'\\([_\*\[\]\(\)])', r'\1', decoded)  # 移除下劃線、星號等的轉義
        decoded = re.sub(r'\\([\.\-])', r'\1', decoded)  # 移除點和連字符的轉義

        # 移除正則表達式模式（如 \d{2}, \d{} 等）
        decoded = re.sub(r'\\d\{\d*\}', '', decoded)  # 移除 \d{2}, \d{} 等
        decoded = re.sub(r'\\d', '', decoded)  # 移除剩餘的 \d

        # 修復常見的 HRV 指標名稱中的空格插入
        hrv_indicators = [
            (r'R\s+M\s+S\s+S\s+D', 'RMSSD'),
            (r'S\s+D\s+N\s+N', 'SDNN'),
            (r'N\s+N\s*5\s*0', 'NN50'),
            (r'p\s*N\s*N\s*5\s*0', 'pNN50'),
            (r'M\s+e\s*a\s*n\s*H\s*R', 'MeanHR'),
            (r'S\s+D\s*H\s*R', 'SDHR'),
            (r'S\s+a\s*m\s*p\s*E\s*n', 'SampEn'),
            (r'D\s+F\s*A', 'DFA'),
            (r'R\s+M\s+S\s+S\s+D\s*_\s*m\s*s', 'RMSSD_ms'),
            (r'S\s+D\s+N\s+N\s*_\s*m\s*s', 'SDNN_ms'),
            (r'M\s+e\s*a\s*n\s*H\s*R\s*_\s*b\s*p\s*m', 'MeanHR_bpm'),
            (r'S\s+a\s*m\s*p\s*E\s*n\s*_\s*z\s*s\s*c\s*o\s*r\s*e', 'SampEn_zscore'),
            (r'D\s+F\s*A\s*_\s*a\s*l\s*p\s*h\s*a', 'DFA_alpha'),
            (r'p\s*N\s*N\s*5\s*0\s*_\s*z\s*s\s*c\s*o\s*r\s*e', 'pNN50_zscore'),
        ]
        for pattern, replacement in hrv_indicators:
            decoded = re.sub(pattern, replacement, decoded, flags=re.IGNORECASE)

        # 修復常見詞彙中的隨機空格插入（更通用的方法）
        def fix_random_spaces(text: str) -> str:
            """修復隨機插入的空格"""
            # 修復常見的詞彙模式
            common_words = [
                (r'I\s+n\s+f\s+e\s+r\s+r\s+e\s+d', 'Inferred'),
                (r'P\s+s\s+y\s+c\s+h\s+o\s+p\s+h\s+y\s+s\s+i\s+o\s+l\s+o\s+g\s+i\s+c\s+a\s+l', 'Psychophysiological'),
                (r'A\s+f\s+f\s+e\s+c\s+t\s+i\s+v\s+e', 'Affective'),
                (r'C\s+o\s+n\s+f\s+i\s+d\s+e\s+n\s+c\s+e', 'Confidence'),
                (r'R\s+a\s+t\s+i\s+o\s+n\s+a\s+l\s+e', 'Rationale'),
                (r'E\s+v\s+i\s+d\s+e\s+n\s+c\s+e', 'Evidence'),
                (r'L\s+i\s+m\s+i\s+t\s+a\s+t\s+i\s+o\s+n\s+s', 'Limitations'),
                (r'V\s+a\s+g\s+a\s+l', 'Vagal'),
                (r'A\s+r\s+o\s+u\s+s\s+a\s+l', 'Arousal'),
                (r'S\s+t\s+a\s+t\s+e', 'State'),
                (r'L\s+e\s+a\s+r\s+n\s+i\s+n\s+g', 'Learning'),
                (r'C\s+o\s+r\s+r\s+e\s+l\s+a\s+t\s+e', 'Correlate'),
            ]
            for pattern, replacement in common_words:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return text

        decoded = fix_random_spaces(decoded)

        # 修復單詞中間的斷字空格問題（更全面的方法）
        def fix_word_breaking_spaces(text: str) -> str:
            """修復單詞中間被錯誤插入的空格"""
            # 常見的醫學/HRV術語詞典（按長度排序，先匹配長的）
            medical_terms = [
                # 複合詞和短語
                (r'z\s*s\s*c\s*o\s*r\s*e', 'zscore'),
                (r'z\s*-\s*s\s*c\s*o\s*r\s*e', 'z-score'),
                (r's\s*h\s*o\s*r\s*t\s*-\s*t\s*e\s*r\s*m', 'short-term'),
                (r'l\s*o\s*n\s*g\s*-\s*t\s*e\s*r\s*m', 'long-term'),
                (r'b\s*e\s*a\s*t\s*-\s*t\s*o\s*-\s*b\s*e\s*a\s*t', 'beat-to-beat'),
                (r'r\s*r\s*\s*i\s*n\s*t\s*e\s*r\s*v\s*a\s*l', 'rr interval'),
                (r'h\s*e\s*a\s*r\s*t\s*\s*r\s*a\s*t\s*e', 'heart rate'),
                (r'h\s*e\s*a\s*r\s*t\s*r\s*a\s*t\s*e', 'heartrate'),
                # 長單詞
                (r'p\s*a\s*r\s*a\s*s\s*y\s*m\s*p\s*a\s*t\s*h\s*e\s*t\s*i\s*c', 'parasympathetic'),
                (r'p\s*s\s*y\s*c\s*h\s*o\s*p\s*h\s*y\s*s\s*i\s*o\s*l\s*o\s*g\s*i\s*c\s*a\s*l', 'psychophysiological'),
                (r'c\s*a\s*r\s*d\s*i\s*o\s*v\s*a\s*s\s*c\s*u\s*l\s*a\s*r', 'cardiovascular'),
                (r'r\s*e\s*s\s*p\s*i\s*r\s*a\s*t\s*o\s*r\s*y', 'respiratory'),
                (r'a\s*u\s*t\s*o\s*n\s*o\s*m\s*i\s*c', 'autonomic'),
                (r'c\s*o\s*m\s*p\s*l\s*e\s*x\s*i\s*t\s*y', 'complexity'),
                (r'v\s*a\s*r\s*i\s*a\s*b\s*i\s*l\s*i\s*t\s*y', 'variability'),
                (r'a\s*d\s*a\s*p\s*t\s*a\s*b\s*i\s*l\s*i\s*t\s*y', 'adaptability'),
                (r'f\s*l\s*e\s*x\s*i\s*b\s*i\s*l\s*i\s*t\s*y', 'flexibility'),
                (r'r\s*e\s*g\s*u\s*l\s*a\s*t\s*o\s*r\s*y', 'regulatory'),
                (r'm\s*o\s*d\s*u\s*l\s*a\s*t\s*i\s*o\s*n', 'modulation'),
                (r'a\s*r\s*r\s*h\s*y\s*t\s*h\s*m\s*i\s*a', 'arrhythmia'),
                (r'h\s*o\s*m\s*e\s*o\s*s\s*t\s*a\s*s\s*i\s*s', 'homeostasis'),
                (r'a\s*d\s*a\s*p\s*t\s*a\s*t\s*i\s*o\s*n', 'adaptation'),
                (r'c\s*o\s*m\s*p\s*e\s*n\s*s\s*a\s*t\s*i\s*o\s*n', 'compensation'),
                (r'c\s*o\s*n\s*t\s*r\s*a\s*d\s*i\s*c\s*t\s*i\s*o\s*n', 'contradiction'),
                (r'd\s*i\s*s\s*c\s*r\s*e\s*p\s*a\s*n\s*c\s*y', 'discrepancy'),
                (r'c\s*o\s*r\s*r\s*o\s*b\s*o\s*r\s*a\s*t\s*i\s*o\s*n', 'corroboration'),
                (r'c\s*h\s*a\s*r\s*a\s*c\s*t\s*e\s*r\s*i\s*z\s*i\s*n\s*g', 'characterizing'),
                (r'd\s*e\s*m\s*o\s*n\s*s\s*t\s*r\s*a\s*t\s*i\s*n\s*g', 'demonstrating'),
                (r'm\s*e\s*t\s*h\s*o\s*d\s*o\s*l\s*o\s*g\s*i\s*c\s*a\s*l', 'methodological'),
                (r'e\s*x\s*p\s*e\s*r\s*i\s*m\s*e\s*n\s*t\s*a\s*l', 'experimental'),
                (r's\s*t\s*a\s*t\s*i\s*s\s*t\s*i\s*c\s*a\s*l', 'statistical'),
                (r'q\s*u\s*a\s*n\s*t\s*i\s*t\s*a\s*t\s*i\s*v\s*e', 'quantitative'),
                (r'q\s*u\s*a\s*l\s*i\s*t\s*a\s*t\s*i\s*v\s*e', 'qualitative'),
                (r'i\s*n\s*t\s*e\s*r\s*p\s*r\s*e\s*t\s*a\s*t\s*i\s*o\s*n', 'interpretation'),
                (r'e\s*v\s*a\s*l\s*u\s*a\s*t\s*i\s*o\s*n', 'evaluation'),
                (r'm\s*e\s*a\s*s\s*u\s*r\s*e\s*m\s*e\s*n\s*t', 'measurement'),
                (r'c\s*a\s*l\s*c\s*u\s*l\s*a\s*t\s*i\s*o\s*n', 'calculation'),
                (r'e\s*s\s*t\s*i\s*m\s*a\s*t\s*i\s*o\s*n', 'estimation'),
                (r'a\s*p\s*p\s*r\s*o\s*x\s*i\s*m\s*a\s*t\s*i\s*o\s*n', 'approximation'),
                (r'c\s*o\s*m\s*p\s*a\s*r\s*i\s*s\s*o\s*n', 'comparison'),
                (r'c\s*o\s*r\s*r\s*e\s*l\s*a\s*t\s*i\s*o\s*n', 'correlation'),
                (r'a\s*s\s*s\s*o\s*c\s*i\s*a\s*t\s*i\s*o\s*n', 'association'),
                (r'r\s*e\s*l\s*a\s*t\s*i\s*o\s*n\s*s\s*h\s*i\s*p', 'relationship'),
                (r'i\s*n\s*t\s*e\s*r\s*a\s*c\s*t\s*i\s*o\s*n', 'interaction'),
                (r'd\s*y\s*n\s*a\s*m\s*i\s*c\s*s', 'dynamics'),
                (r'f\s*r\s*e\s*q\s*u\s*e\s*n\s*c\s*y', 'frequency'),
                (r's\s*p\s*e\s*c\s*t\s*r\s*a\s*l', 'spectral'),
                (r'g\s*e\s*o\s*m\s*e\s*t\s*r\s*i\s*c', 'geometric'),
                (r'n\s*o\s*n\s*l\s*i\s*n\s*e\s*a\s*r', 'nonlinear'),
                (r'f\s*l\s*u\s*c\s*t\s*u\s*a\s*t\s*i\s*o\s*n', 'fluctuation'),
                (r'd\s*e\s*t\s*r\s*e\s*n\s*d\s*e\s*d', 'detrended'),
                (r'o\s*s\s*c\s*i\s*l\s*l\s*a\s*t\s*i\s*o\s*n', 'oscillation'),
                (r'r\s*h\s*y\s*t\s*h\s*m', 'rhythm'),
                (r'p\s*a\s*t\s*t\s*e\s*r\s*n', 'pattern'),
                (r's\s*e\s*q\s*u\s*e\s*n\s*c\s*e', 'sequence'),
                (r'i\s*n\s*t\s*e\s*r\s*v\s*a\s*l', 'interval'),
                (r's\s*u\s*c\s*c\s*e\s*s\s*s\s*i\s*v\s*e', 'successive'),
                (r'd\s*i\s*f\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s', 'differences'),
                (r's\s*t\s*a\s*n\s*d\s*a\s*r\s*d', 'standard'),
                (r'd\s*e\s*v\s*i\s*a\s*t\s*i\s*o\s*n', 'deviation'),
                (r'p\s*e\s*r\s*c\s*e\s*n\s*t\s*a\s*g\s*e', 'percentage'),
                (r'p\s*r\s*o\s*p\s*o\s*r\s*t\s*i\s*o\s*n', 'proportion'),
                (r'c\s*o\s*e\s*f\s*f\s*i\s*c\s*i\s*e\s*n\s*t', 'coefficient'),
                (r'e\s*x\s*p\s*o\s*n\s*e\s*n\s*t', 'exponent'),
                (r'p\s*a\s*r\s*a\s*m\s*e\s*t\s*e\s*r', 'parameter'),
                (r'v\s*a\s*r\s*i\s*a\s*b\s*l\s*e', 'variable'),
                (r'f\s*e\s*a\s*t\s*u\s*r\s*e', 'feature'),
                (r'a\s*t\s*t\s*r\s*i\s*b\s*u\s*t\s*e', 'attribute'),
                (r'c\s*h\s*a\s*r\s*a\s*c\s*t\s*e\s*r\s*i\s*s\s*t\s*i\s*c', 'characteristic'),
                (r'p\s*r\s*o\s*p\s*e\s*r\s*t\s*y', 'property'),
                (r'd\s*i\s*m\s*e\s*n\s*s\s*i\s*o\s*n', 'dimension'),
                (r'a\s*s\s*p\s*e\s*c\s*t', 'aspect'),
                (r'c\s*o\s*m\s*p\s*o\s*n\s*e\s*n\s*t', 'component'),
                (r'e\s*l\s*e\s*m\s*e\s*n\s*t', 'element'),
                (r'f\s*a\s*c\s*t\s*o\s*r', 'factor'),
                (r'd\s*e\s*t\s*e\s*r\s*m\s*i\s*n\s*a\s*n\s*t', 'determinant'),
                (r'p\s*r\s*e\s*d\s*i\s*c\s*t\s*o\s*r', 'predictor'),
                (r'o\s*u\s*t\s*c\s*o\s*m\s*e', 'outcome'),
                (r'c\s*r\s*i\s*t\s*e\s*r\s*i\s*o\s*n', 'criterion'),
                (r'd\s*e\s*p\s*e\s*n\s*d\s*e\s*n\s*t', 'dependent'),
                (r'i\s*n\s*d\s*e\s*p\s*e\s*n\s*d\s*e\s*n\s*t', 'independent'),
                (r'c\s*o\s*v\s*a\s*r\s*i\s*a\s*t\s*e', 'covariate'),
                (r'c\s*o\s*n\s*f\s*o\s*u\s*n\s*d\s*e\s*r', 'confounder'),
                (r'm\s*e\s*d\s*i\s*a\s*t\s*o\s*r', 'mediator'),
                (r'm\s*o\s*d\s*e\s*r\s*a\s*t\s*o\s*r', 'moderator'),
                (r'c\s*a\s*u\s*s\s*a\s*l', 'causal'),
                (r'r\s*e\s*g\s*r\s*e\s*s\s*s\s*i\s*o\s*n', 'regression'),
                (r'p\s*r\s*e\s*d\s*i\s*c\s*t\s*i\s*o\s*n', 'prediction'),
                (r'f\s*o\s*r\s*e\s*c\s*a\s*s\s*t\s*i\s*n\s*g', 'forecasting'),
                (r'p\s*r\s*o\s*j\s*e\s*c\s*t\s*i\s*o\s*n', 'projection'),
                (r'e\s*x\s*t\s*r\s*a\s*p\s*o\s*l\s*a\s*t\s*i\s*o\s*n', 'extrapolation'),
                (r'i\s*n\s*t\s*e\s*r\s*p\s*o\s*l\s*a\s*t\s*i\s*o\s*n', 'interpolation'),
                (r'c\s*a\s*l\s*i\s*b\s*r\s*a\s*t\s*i\s*o\s*n', 'calibration'),
                (r'v\s*a\s*l\s*i\s*d\s*a\s*t\s*i\s*o\s*n', 'validation'),
                (r'v\s*e\s*r\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*o\s*n', 'verification'),
                (r'c\s*o\s*n\s*f\s*i\s*r\s*m\s*a\s*t\s*i\s*o\s*n', 'confirmation'),
                (r'r\s*e\s*p\s*l\s*i\s*c\s*a\s*t\s*i\s*o\s*n', 'replication'),
                (r'r\s*e\s*p\s*r\s*o\s*d\s*u\s*c\s*t\s*i\s*o\s*n', 'reproduction'),
                (r'r\s*e\s*l\s*i\s*a\s*b\s*i\s*l\s*i\s*t\s*y', 'reliability'),
                (r'v\s*a\s*l\s*i\s*d\s*i\s*t\s*y', 'validity'),
                (r'a\s*c\s*c\s*u\s*r\s*a\s*c\s*y', 'accuracy'),
                (r'p\s*r\s*e\s*c\s*i\s*s\s*i\s*o\s*n', 'precision'),
                (r's\s*e\s*n\s*s\s*i\s*t\s*i\s*v\s*i\s*t\s*y', 'sensitivity'),
                (r's\s*p\s*e\s*c\s*i\s*f\s*i\s*c\s*i\s*t\s*y', 'specificity'),
                (r's\s*i\s*g\s*n\s*i\s*f\s*i\s*c\s*a\s*n\s*c\s*e', 'significance'),
                (r'h\s*y\s*p\s*o\s*t\s*h\s*e\s*s\s*i\s*s', 'hypothesis'),
                (r'a\s*l\s*t\s*e\s*r\s*n\s*a\s*t\s*i\s*v\s*e', 'alternative'),
                (r's\s*t\s*a\s*t\s*i\s*s\s*t\s*i\s*c\s*a\s*l', 'statistical'),
                (r'c\s*l\s*i\s*n\s*i\s*c\s*a\s*l', 'clinical'),
                (r'p\s*r\s*a\s*c\s*t\s*i\s*c\s*a\s*l', 'practical'),
                (r'm\s*e\s*a\s*n\s*i\s*n\s*g\s*f\s*u\s*l', 'meaningful'),
                (r's\s*u\s*b\s*s\s*t\s*a\s*n\s*t\s*i\s*a\s*l', 'substantial'),
                (r's\s*u\s*b\s*s\s*t\s*a\s*n\s*t\s*i\s*v\s*e', 'substantive'),
                (r'c\s*o\s*n\s*s\s*i\s*d\s*e\s*r\s*a\s*b\s*l\s*e', 'considerable'),
                (r'n\s*o\s*t\s*a\s*b\s*l\s*e', 'notable'),
                (r'r\s*e\s*m\s*a\s*r\s*k\s*a\s*b\s*l\s*e', 'remarkable'),
                (r's\s*t\s*r\s*i\s*k\s*i\s*n\s*g', 'striking'),
                (r'p\s*r\s*o\s*n\s*o\s*u\s*n\s*c\s*e\s*d', 'pronounced'),
                (r'm\s*a\s*r\s*k\s*e\s*d', 'marked'),
                (r'm\s*o\s*d\s*e\s*r\s*a\s*t\s*e', 'moderate'),
                (r'm\s*i\s*l\s*d', 'mild'),
                (r's\s*u\s*b\s*t\s*l\s*e', 'subtle'),
                (r'm\s*i\s*n\s*i\s*m\s*a\s*l', 'minimal'),
                (r'n\s*e\s*g\s*l\s*i\s*g\s*i\s*b\s*l\s*e', 'negligible'),
                (r't\s*r\s*i\s*v\s*i\s*a\s*l', 'trivial'),
                (r'i\s*n\s*s\s*i\s*g\s*n\s*i\s*f\s*i\s*c\s*a\s*n\s*t', 'insignificant'),
                (r'b\s*o\s*r\s*d\s*e\s*r\s*l\s*i\s*n\s*e', 'borderline'),
                (r'e\s*q\s*u\s*i\s*v\s*o\s*c\s*a\s*l', 'equivocal'),
                (r'a\s*m\s*b\s*i\s*g\s*u\s*o\s*u\s*s', 'ambiguous'),
                (r'u\s*n\s*c\s*l\s*e\s*a\s*r', 'unclear'),
                (r'u\s*n\s*c\s*e\s*r\s*t\s*a\s*i\s*n', 'uncertain'),
                (r'd\s*o\s*u\s*b\s*t\s*f\s*u\s*l', 'doubtful'),
                (r'q\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*a\s*b\s*l\s*e', 'questionable'),
                (r'd\s*e\s*b\s*a\s*t\s*a\s*b\s*l\s*e', 'debatable'),
                (r'c\s*o\s*n\s*t\s*r\s*o\s*v\s*e\s*r\s*s\s*i\s*a\s*l', 'controversial'),
                (r'd\s*i\s*s\s*p\s*u\s*t\s*e\s*d', 'disputed'),
                (r'c\s*o\s*n\s*t\s*e\s*s\s*t\s*e\s*d', 'contested'),
                (r'c\s*h\s*a\s*l\s*l\s*e\s*n\s*g\s*e\s*d', 'challenged'),
                (r'c\s*r\s*i\s*t\s*i\s*c\s*i\s*z\s*e\s*d', 'criticized'),
                (r'q\s*u\s*e\s*s\s*t\s*i\s*o\s*n\s*e\s*d', 'questioned'),
                (r's\s*c\s*r\s*u\s*t\s*i\s*n\s*i\s*z\s*e\s*d', 'scrutinized'),
                (r'e\s*x\s*a\s*m\s*i\s*n\s*e\s*d', 'examined'),
                (r'i\s*n\s*v\s*e\s*s\s*t\s*i\s*g\s*a\s*t\s*e\s*d', 'investigated'),
                (r's\s*t\s*u\s*d\s*i\s*e\s*d', 'studied'),
                (r'a\s*n\s*a\s*l\s*y\s*z\s*e\s*d', 'analyzed'),
                (r'e\s*v\s*a\s*l\s*u\s*a\s*t\s*e\s*d', 'evaluated'),
                (r'a\s*s\s*s\s*e\s*s\s*s\s*e\s*d', 'assessed'),
                (r'm\s*e\s*a\s*s\s*u\s*r\s*e\s*d', 'measured'),
                (r'q\s*u\s*a\s*n\s*t\s*i\s*f\s*i\s*e\s*d', 'quantified'),
                (r'c\s*h\s*a\s*r\s*a\s*c\s*t\s*e\s*r\s*i\s*z\s*e\s*d', 'characterized'),
                (r't\s*y\s*p\s*i\s*f\s*i\s*e\s*d', 'typified'),
                (r'e\s*x\s*e\s*m\s*p\s*l\s*i\s*f\s*i\s*e\s*d', 'exemplified'),
                (r'e\s*m\s*b\s*o\s*d\s*i\s*e\s*d', 'embodied'),
                (r'p\s*e\s*r\s*s\s*o\s*n\s*i\s*f\s*i\s*e\s*d', 'personified'),
                (r'i\s*n\s*c\s*a\s*r\s*n\s*a\s*t\s*e\s*d', 'incarnated'),
                (r'm\s*a\s*t\s*e\s*r\s*i\s*a\s*l\s*i\s*z\s*e\s*d', 'materialized'),
                (r'r\s*e\s*a\s*l\s*i\s*z\s*e\s*d', 'realized'),
                (r'a\s*c\s*t\s*u\s*a\s*l\s*i\s*z\s*e\s*d', 'actualized'),
                (r'c\s*o\s*n\s*c\s*r\s*e\s*t\s*i\s*z\s*e\s*d', 'concretized'),
                (r's\s*u\s*b\s*s\s*t\s*a\s*n\s*t\s*i\s*a\s*t\s*e\s*d', 'substantiated'),
                (r'i\s*n\s*s\s*t\s*a\s*n\s*t\s*i\s*a\s*t\s*e\s*d', 'instantiated'),
                # 中等長度單詞
                (r'b\s*a\s*s\s*e\s*l\s*i\s*n\s*e', 'baseline'),
                (r'a\s*s\s*s\s*e\s*s\s*s\s*m\s*e\s*n\s*t', 'assessment'),
                (r'i\s*n\s*d\s*i\s*c\s*a\s*t\s*o\s*r', 'indicator'),
                (r'c\s*o\s*r\s*r\s*e\s*l\s*a\s*t\s*e', 'correlate'),
                (r'c\s*o\s*n\s*f\s*i\s*d\s*e\s*n\s*c\s*e', 'confidence'),
                (r'r\s*a\s*t\s*i\s*o\s*n\s*a\s*l\s*e', 'rationale'),
                (r'e\s*v\s*i\s*d\s*e\s*n\s*c\s*e', 'evidence'),
                (r'l\s*i\s*m\s*i\s*t\s*a\s*t\s*i\s*o\s*n\s*s', 'limitations'),
                (r'i\s*n\s*f\s*e\s*r\s*e\s*n\s*c\s*e', 'inference'),
                (r'h\s*y\s*p\s*o\s*t\s*h\s*e\s*s\s*i\s*s', 'hypothesis'),
                (r'f\s*o\s*r\s*m\s*a\s*t\s*i\s*o\s*n', 'formation'),
                (r'r\s*e\s*f\s*i\s*n\s*e\s*m\s*e\s*n\s*t', 'refinement'),
                (r'd\s*i\s*s\s*a\s*m\s*b\s*i\s*g\s*u\s*a\s*t\s*i\s*o\s*n', 'disambiguation'),
                (r'i\s*n\s*t\s*e\s*g\s*r\s*a\s*t\s*i\s*o\s*n', 'integration'),
                (r's\s*y\s*n\s*t\s*h\s*e\s*s\s*i\s*s', 'synthesis'),
                (r'a\s*n\s*o\s*m\s*a\s*l\s*y', 'anomaly'),
                (r'c\s*o\s*n\s*f\s*o\s*u\s*n\s*d', 'confound'),
                (r'p\s*r\s*e\s*l\s*i\s*m\s*i\s*n\s*a\s*r\s*y', 'preliminary'),
                (r't\s*e\s*n\s*t\s*a\s*t\s*i\s*v\s*e', 'tentative'),
                (r'c\s*l\s*a\s*s\s*s\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*o\s*n', 'classification'),
                (r'c\s*o\s*n\s*c\s*l\s*u\s*s\s*i\s*o\s*n', 'conclusion'),
                (r'c\s*o\s*n\s*s\s*i\s*s\s*t\s*e\s*n\s*t', 'consistent'),
                (r'r\s*e\s*d\s*u\s*c\s*e\s*d', 'reduced'),
                (r'e\s*l\s*e\s*v\s*a\s*t\s*e\s*d', 'elevated'),
                (r'i\s*n\s*c\s*r\s*e\s*a\s*s\s*e\s*d', 'increased'),
                (r'd\s*e\s*c\s*r\s*e\s*a\s*s\s*e\s*d', 'decreased'),
                (r'e\s*n\s*h\s*a\s*n\s*c\s*e\s*d', 'enhanced'),
                (r's\s*u\s*p\s*p\s*r\s*e\s*s\s*s\s*e\s*d', 'suppressed'),
                (r'a\s*c\s*t\s*i\s*v\s*a\s*t\s*e\s*d', 'activated'),
                (r's\s*t\s*r\s*e\s*s\s*s\s*e\s*d', 'stressed'),
                (r'a\s*n\s*x\s*i\s*o\s*u\s*s', 'anxious'),
                (r'e\s*n\s*g\s*a\s*g\s*e\s*d', 'engaged'),
                (r'c\s*u\s*r\s*i\s*o\s*u\s*s', 'curious'),
                (r'f\s*o\s*c\s*u\s*s\s*e\s*d', 'focused'),
                (r'f\s*l\s*o\s*w', 'flow'),
                (r'd\s*i\s*s\s*e\s*n\s*g\s*a\s*g\s*e\s*d', 'disengaged'),
                (r'c\s*o\s*n\s*f\s*u\s*s\s*e\s*d', 'confused'),
                (r'l\s*e\s*a\s*r\s*n\s*i\s*n\s*g', 'learning'),
                (r's\s*t\s*a\s*t\s*e', 'state'),
                (r's\s*h\s*o\s*r\s*t', 'short'),
                (r'l\s*o\s*n\s*g', 'long'),
                (r't\s*e\s*r\s*m', 'term'),
                (r'r\s*e\s*g\s*u\s*l\s*a\s*t\s*o\s*r\s*y', 'regulation'),
                (r'c\s*o\s*n\s*t\s*r\s*o\s*l', 'control'),
                (r'f\s*u\s*n\s*c\s*t\s*i\s*o\s*n', 'function'),
                (r's\s*y\s*s\s*t\s*e\s*m', 'system'),
                (r'n\s*e\s*r\s*v\s*o\s*u\s*s', 'nervous'),
                (r'r\s*e\s*s\s*p\s*o\s*n\s*s\s*e', 'response'),
                (r'r\s*e\s*a\s*c\s*t\s*i\s*o\s*n', 'reaction'),
                (r'b\s*a\s*l\s*a\s*n\s*c\s*e', 'balance'),
                (r'd\s*o\s*m\s*i\s*n\s*a\s*n\s*c\s*e', 'dominance'),
                (r'i\s*n\s*f\s*l\s*u\s*e\s*n\s*c\s*e', 'influence'),
                (r'a\s*c\s*t\s*i\s*v\s*i\s*t\s*y', 'activity'),
                (r'm\s*o\s*d\s*u\s*l\s*a\s*t\s*i\s*o\s*n', 'modulation'),
                (r'r\s*e\s*g\s*u\s*l\s*a\s*t\s*i\s*o\s*n', 'regulation'),
                (r'c\s*o\s*n\s*t\s*r\s*o\s*l', 'control'),
                (r'f\s*u\s*n\s*c\s*t\s*i\s*o\s*n', 'function'),
                (r's\s*y\s*s\s*t\s*e\s*m', 'system'),
                (r'n\s*e\s*r\s*v\s*o\s*u\s*s', 'nervous'),
                (r'r\s*e\s*s\s*p\s*o\s*n\s*s\s*e', 'response'),
                (r'r\s*e\s*a\s*c\s*t\s*i\s*o\s*n', 'reaction'),
                (r'b\s*a\s*l\s*a\s*n\s*c\s*e', 'balance'),
                (r'd\s*o\s*m\s*i\s*n\s*a\s*n\s*c\s*e', 'dominance'),
                (r'i\s*n\s*f\s*l\s*u\s*e\s*n\s*c\s*e', 'influence'),
                (r'a\s*c\s*t\s*i\s*v\s*i\s*t\s*y', 'activity'),
                # 短單詞
                (r'v\s*a\s*g\s*a\s*l', 'vagal'),
                (r'a\s*r\s*o\s*u\s*s\s*a\s*l', 'arousal'),
                (r'c\s*o\s*m\s*p\s*a\s*r\s*e\s*d', 'compared'),
                (r'r\s*e\s*d\s*u\s*c\s*e\s*d', 'reduced'),
                (r'i\s*n\s*d\s*i\s*c\s*a\s*t\s*e\s*s', 'indicates'),
                (r's\s*u\s*g\s*g\s*e\s*s\s*t\s*s', 'suggests'),
                (r's\s*u\s*p\s*p\s*o\s*r\s*t\s*s', 'supports'),
                (r'a\s*l\s*i\s*g\s*n\s*s', 'aligns'),
                (r'w\s*i\s*t\s*h', 'with'),
                (r'o\s*v\s*e\s*r', 'over'),
                (r'r\s*a\s*l\s*l', 'overall'),
                (r'c\s*o\s*n\s*f\s*i\s*r\s*m\s*s', 'confirms'),
                (r'f\s*u\s*r\s*t\s*h\s*e\s*r', 'further'),
                (r'v\s*i\s*e\s*w', 'view'),
                (r'c\s*o\s*n\s*c\s*l\s*u\s*s\s*i\s*o\s*n', 'conclusion'),
                (r'c\s*o\s*m\s*b\s*i\s*n\s*i\s*n\s*g', 'combining'),
                (r'p\s*o\s*i\s*n\s*t\s*s', 'points'),
                (r't\s*o\s*w\s*a\s*r\s*d\s*s', 'towards'),
                (r't\s*e\s*s\s*t', 'test'),
                (r'm\s*i\s*l\s*d\s*l\s*y', 'mildly'),
                (r'r\s*e\s*l\s*a\s*t\s*i\s*v\s*e', 'relative'),
                (r'c\s*o\s*n\s*s\s*i\s*s\s*t\s*e\s*n\s*t', 'consistent'),
                (r'c\s*a\s*t\s*e\s*g\s*o\s*r\s*y', 'category'),
                (r'r\s*e\s*v\s*e\s*a\s*l\s*s', 'reveals'),
                (r's\s*u\s*b\s*s\s*t\s*a\s*n\s*t\s*i\s*a\s*l\s*l\s*y', 'substantially'),
                (r'e\s*n\s*h\s*a\s*n\s*c\s*e\s*d', 'enhanced'),
                (r'r\s*a\s*n\s*g\s*e', 'range'),
                (r'o\s*r\s*d\s*e\s*r', 'order'),
                (r'f\s*i\s*n\s*d\s*i\s*n\s*g', 'finding'),
                (r'c\s*o\s*n\s*t\s*r\s*a\s*d\s*i\s*c\s*t\s*s', 'contradicts'),
                (r'e\s*x\s*p\s*e\s*c\s*t\s*a\s*t\s*i\s*o\s*n\s*s', 'expectations'),
                (r's\s*i\s*m\s*p\s*l\s*e', 'simple'),
                (r's\s*t\s*r\s*e\s*s\s*s', 'stress'),
                (r'r\s*e\s*s\s*p\s*o\s*n\s*s\s*e', 'response'),
                (r'w\s*h\s*i\s*c\s*h', 'which'),
                (r'o\s*f\s*t\s*e\s*n', 'often'),
                (r'd\s*e\s*c\s*r\s*e\s*a\s*s\s*e\s*s', 'decreases'),
                (r'i\s*n\s*t\s*e\s*g\s*r\s*a\s*t\s*i\s*o\s*n', 'integration'),
                (r'p\s*r\s*i\s*o\s*r\s*i\s*t\s*i\s*z\s*i\s*n\s*g', 'prioritizing'),
                (r'a\s*c\s*c\s*o\s*r\s*d\s*i\s*n\s*g', 'according'),
                (r'i\s*n\s*s\s*t\s*r\s*u\s*c\s*t\s*i\s*o\s*n\s*s', 'instructions'),
                (r's\s*t\s*r\s*e\s*n\s*g\s*t\s*h\s*e\s*n\s*s', 'strengthens'),
                (r'c\s*l\s*a\s*s\s*s\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*o\s*n', 'classification'),
                (r'h\s*o\s*w\s*e\s*v\s*e\s*r', 'however'),
                (r's\s*u\s*g\s*g\s*e\s*s\s*t\s*s', 'suggests'),
                (r'r\s*e\s*s\s*i\s*l\s*i\s*e\s*n\s*c\s*e', 'resilience'),
                (r'a\s*d\s*a\s*p\s*t\s*a\s*b\s*i\s*l\s*i\s*t\s*y', 'adaptability'),
                (r'm\s*a\s*i\s*n\s*t\s*a\s*i\s*n\s*e\s*d', 'maintained'),
                (r'd\s*e\s*s\s*p\s*i\s*t\s*e', 'despite'),
                (r'a\s*c\s*u\s*t\s*e', 'acute'),
                (r'c\s*h\s*a\s*n\s*g\s*e\s*s', 'changes'),
                (r'r\s*e\s*f\s*l\s*e\s*c\s*t\s*e\s*d', 'reflected'),
                (r'm\s*e\s*a\s*s\s*u\s*r\s*e\s*s', 'measures'),
                (r'c\s*r\s*e\s*a\s*t\s*e\s*s', 'creates'),
                (r'n\s*u\s*a\s*n\s*c\s*e\s*d', 'nuanced'),
                (r'p\s*i\s*c\s*t\s*u\s*r\s*e', 'picture'),
                (r'w\s*h\s*e\s*r\s*e', 'where'),
                (r'm\s*a\s*y', 'may'),
                (r'b\s*e', 'be'),
                (r'i\s*m\s*p\s*a\s*i\s*r\s*e\s*d', 'impaired'),
                (r'b\s*u\s*t', 'but'),
                (r's\s*y\s*s\s*t\s*e\s*m\s*i\s*c', 'systemic'),
                (r's\s*t\s*a\s*b\s*i\s*l\s*i\s*t\s*y', 'stability'),
                (r'p\s*e\s*r\s*s\s*i\s*s\s*t\s*s', 'persists'),
                (r's\s*y\s*n\s*t\s*h\s*e\s*s\s*i\s*s', 'synthesis'),
                (r'n\s*e\s*g\s*a\s*t\s*i\s*v\s*e', 'negative'),
                (r's\s*c\s*o\s*r\s*e\s*s', 'scores'),
                (r'p\s*o\s*i\s*n\s*t', 'point'),
                (r'w\s*a\s*r\s*d', 'ward'),
                (r'd\s*e\s*c\s*r\s*e\s*a\s*s\s*e\s*d', 'decreased'),
                (r'p\s*o\s*s\s*i\s*t\s*i\s*v\s*e', 'positive'),
                (r's\s*c\s*o\s*r\s*e', 'score'),
                (r'h\s*i\s*g\s*h\s*e\s*r', 'higher'),
                (r'v\s*i\s*s\s*u\s*a\s*l\s*i\s*z\s*e\s*s', 'visualizes'),
                (r'e\s*l\s*o\s*n\s*g\s*a\s*t\s*e\s*d', 'elongated'),
                (r's\s*h\s*a\s*p\s*e', 'shape'),
                (r'i\s*n\s*d\s*i\s*c\s*a\s*t\s*i\s*n\s*g', 'indicating'),
                (r'p\s*r\s*e\s*d\s*o\s*m\s*i\s*n\s*a\s*n\s*t\s*l\s*y', 'predominantly'),
                (r'v\s*e\s*r\s*t\s*i\s*c\s*a\s*l', 'vertical'),
                (r'f\s*l\s*u\s*c\s*t\s*u\s*a\s*t\s*i\s*o\s*n\s*s', 'fluctuations'),
                (r'l\s*e\s*s\s*s', 'less'),
                (r'v\s*a\s*r\s*i\s*a\s*b\s*i\s*l\s*i\s*t\s*y', 'variability'),
                (r'o\s*v\s*e\s*r\s*l\s*a\s*i\s*d', 'overlaid'),
                (r'b\s*r\s*o\s*a\s*d\s*e\s*r', 'broader'),
                (r't\s*e\s*m\s*p\s*o\s*r\s*a\s*l', 'temporal'),
                (r'd\s*r\s*i\s*f\s*t\s*s', 'drifts'),
                (r'r\s*a\s*t\s*i\s*o', 'ratio'),
                (r'e\s*s\s*t\s*i\s*m\s*a\s*t\s*e\s*d', 'estimated'),
                (r'f\s*r\s*o\s*m', 'from'),
                (r'g\s*r\s*a\s*p\s*h', 'graph'),
                (r'a\s*p\s*p\s*r\s*o\s*x\s*i\s*m\s*a\s*t\s*e\s*l\s*y', 'approximately'),
                (r'e\s*m\s*p\s*h\s*a\s*s\s*i\s*z\s*i\s*n\s*g', 'emphasizing'),
                (r'l\s*i\s*m\s*i\s*t\s*e\s*d', 'limited'),
                (r'd\s*y\s*n\s*a\s*m\s*i\s*c\s*s', 'dynamics'),
                (r'e\s*x\s*c\s*e\s*l\s*l\s*e\s*n\s*t', 'excellent'),
                (r's\s*i\s*g\s*n\s*a\s*l', 'signal'),
                (r'q\s*u\s*a\s*l\s*i\s*t\s*y', 'quality'),
                (r'm\s*i\s*n\s*i\s*m\s*i\s*z\s*e\s*s', 'minimizes'),
                (r'a\s*r\s*t\s*e\s*f\s*a\s*c\s*t', 'artifact'),
                (r'c\s*o\s*n\s*c\s*e\s*r\s*n\s*s', 'concerns'),
                (r'l\s*a\s*c\s*k', 'lack'),
                (r'd\s*a\s*t\s*a', 'data'),
                (r'p\s*r\s*e\s*v\s*e\s*n\s*t\s*s', 'prevents'),
                (r'f\s*u\s*l\s*l', 'full'),
                (r'i\s*n\s*t\s*e\s*r\s*p\s*r\s*e\s*t\s*a\s*t\s*i\s*o\s*n', 'interpretation'),
                (r'e\s*s\s*p\s*e\s*c\s*i\s*a\s*l\s*l\s*y', 'especially'),
                (r's\s*u\s*b\s*j\s*e\s*c\s*t', 'subject'),
                (r'a\s*g\s*e', 'age'),
                (r'y\s*e\s*a\s*r\s*s', 'years'),
                (r'p\s*l\s*a\s*c\s*e\s*s', 'places'),
                (r'y\s*o\s*u\s*n\s*g', 'young'),
                (r'a\s*d\s*u\s*l\s*t', 'adult'),
                (r'p\s*h\s*a\s*s\s*e', 'phase'),
                (r't\s*y\s*p\s*i\s*c\s*a\s*l\s*l\s*y', 'typically'),
                (r'e\s*x\s*h\s*i\s*b\s*i\s*t\s*s', 'exhibits'),
                (r'h\s*i\s*g\s*h\s*e\s*r', 'higher'),
                (r't\s*h\s*a\s*n', 'than'),
                (r'o\s*l\s*d\s*e\s*r', 'older'),
                (r'p\s*o\s*p\s*u\s*l\s*a\s*t\s*i\s*o\s*n\s*s', 'populations'),
                (r'c\s*o\s*n\s*f\s*l\s*i\s*c\s*t\s*i\s*n\s*g', 'conflicting'),
                (r'f\s*i\s*n\s*d\s*i\s*n\s*g\s*s', 'findings'),
                (r'b\s*e\s*t\s*w\s*e\s*e\s*n', 'between'),
                (r'c\s*o\s*m\s*p\s*l\s*e\s*x\s*i\s*t\s*y', 'complexity'),
                (r's\s*c\s*a\s*l\s*i\s*n\s*g', 'scaling'),
                (r'r\s*e\s*q\s*u\s*i\s*r\s*e', 'require'),
                (r'c\s*a\s*u\s*t\s*i\s*o\s*u\s*s', 'cautious'),
                (r'i\s*n\s*t\s*e\s*r\s*p\s*r\s*e\s*t\s*a\s*t\s*i\s*o\s*n', 'interpretation'),
                (r'i\s*n\s*t\s*e\s*g\s*r\s*a\s*t\s*i\s*n\s*g', 'integrating'),
                (r'm\s*o\s*r\s*p\s*h\s*o\s*l\s*o\s*g\s*y', 'morphology'),
                (r'm\s*e\s*t\s*r\s*i\s*c\s*s', 'metrics'),
                (r'l\s*e\s*a\s*d\s*s', 'leads'),
                (r'c\s*l\s*a\s*s\s*s\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*o\s*n', 'classification'),
                (r'd\s*o\s*m\s*i\s*n\s*a\s*t\s*e\s*s', 'dominates'),
                (r'w\s*h\s*i\s*l\s*e', 'while'),
                (r'a\s*p\s*p\s*e\s*a\s*r\s*s', 'appears'),
                (r'p\s*r\s*e\s*s\s*e\s*r\s*v\s*e\s*d', 'preserved'),
                (r'i\s*m\s*m\s*e\s*d\s*i\s*a\s*t\s*e', 'immediate'),
                (r'p\s*h\s*y\s*s\s*i\s*o\s*l\s*o\s*g\s*i\s*c\s*a\s*l', 'physiological'),
                (r'p\s*r\s*o\s*f\s*i\s*l\s*e', 'profile'),
                (r'r\s*e\s*f\s*l\s*e\s*c\s*t\s*s', 'reflects'),
                (r'a\s*c\s*t\s*i\s*v\s*a\s*t\s*e\s*d', 'activated'),
                (r'a\s*m\s*b\s*i\s*g\s*u\s*i\s*t\s*i\s*e\s*s', 'ambiguities'),
                (r'p\s*e\s*r\s*s\s*i\s*s\s*t\s*e\s*n\s*c\s*e', 'persistence'),
                (r'd\s*u\s*r\s*i\s*n\s*g', 'during'),
                (r'p\s*e\s*r\s*i\s*o\s*d', 'period'),
                (r'a\s*p\s*p\s*a\s*r\s*e\s*n\s*t', 'apparent'),
                (r'r\s*e\s*m\s*a\s*i\s*n\s*s', 'remains'),
                (r'p\s*u\s*z\s*z\s*l\s*i\s*n\s*g', 'puzzling'),
                (r'r\s*e\s*p\s*r\s*e\s*s\s*e\s*n\s*t', 'represent'),
                (r'a\s*d\s*a\s*p\s*t\s*i\s*v\s*e', 'adaptive'),
                (r'm\s*e\s*c\s*h\s*a\s*n\s*i\s*s\s*m\s*s', 'mechanisms'),
                (r'i\s*n\s*h\s*e\s*r\s*e\s*n\s*t', 'inherent'),
                (r'i\s*n\s*d\s*i\s*v\s*i\s*d\s*u\s*a\s*l', 'individual'),
                (r'd\s*i\s*f\s*f\s*e\s*r\s*e\s*n\s*c\s*e\s*s', 'differences'),
                (r'p\s*r\s*o\s*v\s*i\s*d\s*e', 'provide'),
                (r's\s*t\s*r\s*o\s*n\s*g', 'strong'),
                (r'b\s*u\s*t\s*t\s*r\s*e\s*s\s*s\s*e\s*d', 'buttressed'),
                (r'd\s*i\s*s\s*c\s*r\s*e\s*p\s*a\s*n\s*c\s*y', 'discrepancy'),
                (r't\s*e\s*m\s*p\s*e\s*r\s*s', 'tempers'),
                (r'c\s*e\s*r\s*t\s*a\s*i\s*n\s*t\s*y', 'certainty'),
                (r'c\s*o\s*r\s*r\s*e\s*l\s*a\s*t\s*e', 'correlate'),
                (r'm\s*a\s*p\s*s', 'maps'),
                (r'b\s*e\s*h\s*a\s*v\s*i\s*o\s*r\s*a\s*l', 'behavioral'),
                (r'm\s*a\s*n\s*i\s*f\s*e\s*s\s*t\s*a\s*t\s*i\s*o\s*n\s*s', 'manifestations'),
                (r'i\s*n\s*c\s*l\s*u\s*d\s*e', 'include'),
                (r'd\s*i\s*f\s*f\s*i\s*c\s*u\s*l\s*t\s*y', 'difficulty'),
                (r'f\s*o\s*c\s*u\s*s\s*i\s*n\s*g', 'focusing'),
                (r'i\s*n\s*c\s*r\s*e\s*a\s*s\s*e\s*d', 'increased'),
                (r'd\s*i\s*s\s*t\s*r\s*a\s*c\s*t\s*i\s*o\s*n', 'distraction'),
                (r'i\s*m\s*p\s*a\s*i\s*r\s*e\s*d', 'impaired'),
                (r'c\s*o\s*g\s*n\s*i\s*t\s*i\s*v\s*e', 'cognitive'),
                (r'p\s*e\s*r\s*f\s*o\s*r\s*m\s*a\s*n\s*c\s*e', 'performance'),
                (r'd\s*u\s*e', 'due'),
                (r'd\s*i\s*v\s*i\s*d\s*e\s*d', 'divided'),
                (r'a\s*t\s*t\s*e\s*n\s*t\s*i\s*o\s*n', 'attention'),
                (r'a\s*m\s*o\s*n\s*g', 'among'),
                (r'e\s*m\s*o\s*t\s*i\s*o\s*n\s*a\s*l', 'emotional'),
                (r'r\s*e\s*s\s*p\s*o\s*n\s*s\s*e\s*s', 'responses'),
                (r't\s*a\s*s\s*k\s*s', 'tasks'),
                (r's\s*p\s*e\s*c\s*u\s*l\s*a\s*t\s*i\s*v\s*e', 'speculative'),
                (r'f\s*l\s*a\s*g', 'flag'),
                (r'r\s*e\s*q\s*u\s*i\s*r\s*e\s*s', 'requires'),
                (r'k\s*n\s*o\s*w\s*l\s*e\s*d\s*g\s*e', 'knowledge'),
                (r'e\s*n\s*v\s*i\s*r\s*o\s*n\s*m\s*e\s*n\s*t\s*a\s*l', 'environmental'),
                (r'c\s*o\s*n\s*d\s*i\s*t\s*i\s*o\s*n\s*s', 'conditions'),
                (r'p\s*r\s*e\s*s\s*e\s*n\s*t\s*a\s*t\s*i\s*o\s*n', 'presentation'),
                (r's\s*e\s*s\s*s\s*i\s*o\s*n', 'session'),
                (r'v\s*a\s*l\s*i\s*d\s*a\s*t\s*i\s*o\s*n', 'validation'),
                (r'r\s*u\s*l\s*e', 'rule'),
                (r'a\s*u\s*d\s*i\s*t', 'audit'),
                (r's\s*t\s*r\s*i\s*c\s*t\s*l\s*y', 'strictly'),
                (r'f\s*o\s*l\s*l\s*o\s*w\s*e\s*d', 'followed'),
                (r'p\s*r\s*i\s*o\s*r\s*i\s*t\s*y', 'priority'),
                (r'r\s*u\s*l\s*e\s*s', 'rules'),
                (r'p\s*r\s*i\s*m\s*a\s*r\s*y', 'primary'),
                (r'b\s*e\s*f\s*o\s*r\s*e', 'before'),
                (r'f\s*e\s*a\s*t\s*u\s*r\s*e\s*s', 'features'),
                (r'a\s*c\s*c\s*o\s*u\s*n\s*t\s*e\s*d', 'accounted'),
                (r'p\s*o\s*s\s*s\s*i\s*b\s*l\s*e', 'possible'),
                (r'a\s*v\s*o\s*i\s*d\s*e\s*d', 'avoided'),
                (r'a\s*b\s*s\s*o\s*l\s*u\s*t\s*e', 'absolute'),
                (r'c\s*u\s*t\s*o\s*f\s*f\s*s', 'cutoffs'),
                (r'e\s*x\s*c\s*e\s*p\s*t', 'except'),
                (r'w\s*h\s*e\s*r\s*e', 'where'),
                (r'n\s*e\s*c\s*e\s*s\s*s\s*a\s*r\s*y', 'necessary'),
                (r'g\s*i\s*v\s*e\s*n', 'given'),
                (r'c\s*o\s*m\s*p\s*l\s*i\s*a\s*n\s*c\s*e', 'compliance'),
                (r'a\s*d\s*h\s*e\s*r\s*e\s*d', 'adhered'),
                (r'p\s*r\s*e\s*c\s*i\s*s\s*e\s*l\s*y', 'precisely'),
                (r't\s*e\s*m\s*p\s*l\s*a\s*t\s*e', 'template'),
                (r's\s*t\s*r\s*u\s*c\s*t\s*u\s*r\s*e', 'structure'),
                (r'i\s*n\s*c\s*l\s*u\s*d\s*i\s*n\s*g', 'including'),
                (r'n\s*u\s*m\s*b\s*e\s*r\s*e\s*d', 'numbered'),
                (r's\s*e\s*c\s*t\s*i\s*o\s*n\s*s', 'sections'),
                (r'b\s*o\s*l\s*d', 'bold'),
                (r'h\s*e\s*a\s*d\s*e\s*r\s*s', 'headers'),
                (r'b\s*u\s*l\s*l\s*e\s*t\s*s', 'bullets'),
                (r'm\s*a\s*i\s*n\s*t\s*a\s*i\s*n\s*e\s*d', 'maintained'),
                (r'r\s*e\s*q\s*u\s*i\s*r\s*e\s*d', 'required'),
                (r't\s*e\s*r\s*m\s*i\s*n\s*o\s*l\s*o\s*g\s*y', 'terminology'),
                (r'a\s*d\s*j\s*u\s*s\s*t\s*m\s*e\s*n\s*t\s*s', 'adjustments'),
                (r'p\s*h\s*a\s*s\s*e', 'phase'),
                (r'l\s*o\s*c\s*k\s*i\s*n\s*g', 'locking'),
                (r'w\s*o\s*u\s*l\s*d', 'would'),
                (r'c\s*l\s*a\s*r\s*i\s*f\s*y', 'clarify'),
                (r'a\s*n\s*a\s*l\s*y\s*s\s*i\s*s', 'analysis'),
                (r'l\s*o\s*n\s*g\s*i\s*t\s*u\s*d\s*i\s*n\s*a\s*l', 'longitudinal'),
                (r't\s*r\s*a\s*c\s*k\s*i\s*n\s*g', 'tracking'),
                (r'c\s*o\s*u\s*l\s*d', 'could'),
                (r'd\s*e\s*t\s*e\s*r\s*m\s*i\s*n\s*e', 'determine'),
                (r't\s*r\s*a\s*n\s*s\s*i\s*e\s*n\s*c\s*e', 'transience'),
                (r'v\s*e\s*r\s*s\s*u\s*s', 'versus'),
                (r'c\s*h\s*r\s*o\s*n\s*i\s*c\s*i\s*t\s*y', 'chronicity'),
                (r'p\s*a\s*t\s*t\s*e\s*r\s*n\s*s', 'patterns'),
            ]

            # 按長度排序，先匹配長的（避免短詞覆蓋長詞的一部分）
            medical_terms.sort(key=lambda x: len(x[1]), reverse=True)

            for pattern, replacement in medical_terms:
                # 使用單詞邊界確保完整匹配
                full_pattern = r'\b' + pattern + r'\b'
                text = re.sub(full_pattern, replacement, text, flags=re.IGNORECASE)

            return text

        decoded = fix_word_breaking_spaces(decoded)

        # 修復單個字母之間的空格（但保留單詞之間的空格）
        # 只修復明顯是錯誤的空格插入（連續的單字母+空格模式）
        def fix_single_char_spaces(text: str) -> str:
            """修復單個字符之間的空格插入"""
            lines = text.split('\n')
            fixed_lines = []
            for line in lines:
                # 如果一行中有太多單字母+空格的模式，可能是錯誤的空格插入
                # 但我們要小心，不要破壞正常的文本
                # 只修復明顯的模式，如 "R M S S D" -> "RMSSD"
                fixed_line = line
                # 修復 "z = -0. 83" -> "z = -0.83" 這樣的模式
                fixed_line = re.sub(r'(-?\d+\.)\s+(\d+)', r'\1\2', fixed_line)
                # 修復 "z = -\d{2}. 89" -> "z = -0.89" 這樣的模式（但先移除 \d{2}）
                fixed_line = re.sub(r'z\s*=\s*-?\s*\\d\{[^}]*\}\s*\.\s*(\d+)', r'z = -0.\1', fixed_line, flags=re.IGNORECASE)
                fixed_lines.append(fixed_line)
            return '\n'.join(fixed_lines)

        decoded = fix_single_char_spaces(decoded)

        # 8. 規範化標籤：修復常見的 < answer> / </ answer> 錯誤
        decoded = re.sub(r'<\s*answer\s*>', '<answer>', decoded, flags=re.IGNORECASE)
        decoded = re.sub(r'</\s*answer\s*>', '</answer>', decoded, flags=re.IGNORECASE)
        decoded = re.sub(r'<\s*think\s*>', '<think>', decoded, flags=re.IGNORECASE)
        decoded = re.sub(r'</\s*think\s*>', '</think>', decoded, flags=re.IGNORECASE)
        decoded = decoded.replace("\\<", "<").replace("\\>", ">")

        # 9. 在清理後立即修復缺失的 XML 標籤（增強版）
        def fix_missing_xml_tags(text: str) -> str:
            """如果找不到 <think> 或 <answer> 標籤，嘗試根據關鍵字位置自動注入"""
            # 檢查標籤是否存在（不區分大小寫）
            has_think = bool(re.search(r'<think>', text, re.IGNORECASE))
            has_answer = bool(re.search(r'<answer>', text, re.IGNORECASE))

            if has_think and has_answer:
                # 確保標籤正確關閉
                has_think_close = bool(re.search(r'</think>', text, re.IGNORECASE))
                has_answer_close = bool(re.search(r'</answer>', text, re.IGNORECASE))

                if not has_think_close:
                    # 在 <answer> 之前插入 </think>
                    answer_pos = re.search(r'<answer>', text, re.IGNORECASE)
                    if answer_pos:
                        text = text[:answer_pos.start()] + "\n</think>\n\n" + text[answer_pos.start():]

                if not has_answer_close:
                    # 在文本末尾或 State: 行之前插入 </answer>
                    state_match = re.search(r'^State:\s*', text, re.MULTILINE)
                    if state_match:
                        text = text[:state_match.start()] + "\n</answer>" + text[state_match.start():]
                    else:
                        text = text.rstrip() + "\n</answer>"

                return text

            # 尋找 think 區塊開始標記（多種可能的格式）
            think_start_markers = [
                "**1. Data Ingestion",
                "**1. Data Extraction",
                "**1. Preliminary",
                "**1. ",
            ]
            think_start_pos = -1
            think_marker = None
            for marker in think_start_markers:
                pos = text.find(marker)
                if pos >= 0:
                    think_start_pos = pos
                    think_marker = marker
                    break

            # 尋找 answer 區塊開始標記（多種可能的格式）
            answer_start_markers = [
                "**1. Inferred Psychophysiological State",
                "**1. Inferred",
                "Inferred Psychophysiological State",
                "State:",
            ]
            answer_start_pos = -1
            answer_marker = None
            for marker in answer_start_markers:
                pos = text.find(marker)
                if pos >= 0 and (answer_start_pos < 0 or pos < answer_start_pos):
                    answer_start_pos = pos
                    answer_marker = marker

            # 如果找不到 answer 標記，嘗試尋找 State: 行
            if answer_start_pos < 0:
                state_match = re.search(r'^State:\s*', text, re.MULTILINE)
                if state_match:
                    answer_start_pos = state_match.start()
                    answer_marker = "State:"

            # 如果連 answer 標記都找不到，可能格式太亂，放棄修復
            if answer_start_pos < 0:
                return text

            new_text = text

            # 如果缺少 <answer> 標籤
            if not has_answer and answer_start_pos >= 0:
                # 在 Answer 區塊開始前插入 <answer>
                parts = new_text.split(answer_marker, 1)
                if len(parts) == 2:
                    pre_answer = parts[0].rstrip()
                    post_answer = answer_marker + parts[1]

                    # 如果 pre_answer 尾部沒有 </think>，且我們假設前面是 think，則加上 </think>
                    if not re.search(r'</think>', pre_answer, re.IGNORECASE):
                        if has_think:
                            pre_answer += "\n</think>"
                        elif think_start_pos >= 0:
                            pre_answer += "\n</think>"

                    # 在 State: 行之前或文本末尾插入 </answer>
                    state_match = re.search(r'^State:\s*', post_answer, re.MULTILINE)
                    if state_match:
                        answer_end_pos = state_match.start()
                        new_text = f"{pre_answer}\n\n<answer>\n{post_answer[:answer_end_pos].rstrip()}\n</answer>\n{post_answer[answer_end_pos:]}"
                    else:
                        new_text = f"{pre_answer}\n\n<answer>\n{post_answer}\n</answer>"

            # 如果缺少 <think> 標籤
            if not has_think:
                # 在 Think 區塊開始前插入 <think>
                if think_start_pos >= 0 and think_marker:
                    parts = new_text.split(think_marker, 1)
                    if len(parts) == 2:
                        new_text = f"{parts[0]}<think>\n{think_marker}{parts[1]}"
                elif new_text.strip().startswith("**1."):
                    new_text = f"<think>\n{new_text}"
                elif answer_start_pos > 0:
                    # 如果找不到 think 標記，但在 answer 之前有內容，假設前面是 think
                    new_text = f"<think>\n{new_text}"

            # 清理可能產生的多餘標籤（例如重複的標籤）
            # 確保只有一個 <think> 和一個 <answer>
            think_tags = list(re.finditer(r'<think>', new_text, re.IGNORECASE))
            if len(think_tags) > 1:
                # 只保留第一個
                for tag in think_tags[1:]:
                    new_text = new_text[:tag.start()] + new_text[tag.end():]

            answer_tags = list(re.finditer(r'<answer>', new_text, re.IGNORECASE))
            if len(answer_tags) > 1:
                # 只保留第一個
                for tag in answer_tags[1:]:
                    new_text = new_text[:tag.start()] + new_text[tag.end():]

            return new_text

        # 在清理後立即修復 XML 標籤
        decoded = fix_missing_xml_tags(decoded)

        # 規範化 <think> 步驟標頭：將 "**Step N:" 改為 "**N. " 以通過步驟計數
        def normalize_think_steps(text: str) -> str:
            def repl(m):
                num = m.group(1)
                rest = m.group(2) or ''
                return f"**{num}. {rest}".rstrip()
            return re.sub(r"\*\*\s*Step\s*([1-7])\s*:\s*(.*?)\s*(?=\n)", repl, text, flags=re.IGNORECASE|re.DOTALL)

        decoded = re.sub(
            r"(<think>)([\s\S]*?)(</think>)",
            lambda m: m.group(1) + normalize_think_steps(m.group(2)) + m.group(3),
            decoded,
            flags=re.IGNORECASE,
        )

        # 確保 <answer> 區塊包含六個精確欄位標籤；若缺則在開頭注入標籤行
        required_labels = [
            "**1. Inferred Psychophysiological State:**",
            "**2. [REVISED] Inferred Affective State / Correlate:**",
            "**3. [REVISED] Inferred Learning State Correlate:**",
            "**4. [REVISED] Confidence Level:**",
            "**5. [REVISED] Key Rationale and Evidence:**",
            "**6. [REVISED] Notes on Input Limitations:**",
        ]

        def ensure_answer_labels(text: str) -> str:
            m = re.search(r"<answer>([\s\S]*?)</answer>", text, flags=re.IGNORECASE)
            if not m:
                return text
            content = m.group(1).strip()

            # 檢查內容是否為空或只有標籤沒有實際內容
            # 如果內容只有標籤標題而沒有實際描述，嘗試從其他地方提取
            has_actual_content = False
            for label in required_labels:
                # 檢查標籤後面是否有實際內容（不只是換行或空白）
                label_pos = content.find(label)
                if label_pos >= 0:
                    # 找到標籤後的下一個非空白內容
                    after_label = content[label_pos + len(label):].strip()
                    # 如果標籤後有內容（超過10個字符），認為有實際內容
                    if len(after_label) > 10:
                        has_actual_content = True
                        break

            # 如果沒有實際內容，嘗試從 <think> 中提取關鍵信息
            if not has_actual_content and len(content) < 200:
                # 嘗試從 redacted_reasoning 中提取狀態信息
                reasoning_match = re.search(r"<think>([\s\S]*?)</think>", text, flags=re.IGNORECASE)
                if reasoning_match:
                    reasoning_content = reasoning_match.group(1)

                    # 先清理 reasoning_content 中的空格插入問題
                    reasoning_content = re.sub(r'R\s+M\s+S\s+S\s+D', 'RMSSD', reasoning_content, flags=re.IGNORECASE)
                    reasoning_content = re.sub(r'S\s+D\s+N\s+N', 'SDNN', reasoning_content, flags=re.IGNORECASE)
                    reasoning_content = re.sub(r'M\s+e\s*a\s*n\s*H\s*R', 'MeanHR', reasoning_content, flags=re.IGNORECASE)
                    reasoning_content = re.sub(r'S\s+a\s*m\s*p\s*E\s*n', 'SampEn', reasoning_content, flags=re.IGNORECASE)
                    reasoning_content = re.sub(r'D\s+F\s*A', 'DFA', reasoning_content, flags=re.IGNORECASE)

                    # 提取狀態標籤（HVHA/HVLA/LVHA/LVLA）- 更寬鬆的匹配
                    state_match = re.search(r'\b(HVHA|HVLA|LVHA|LVLA)\b', reasoning_content, re.IGNORECASE)
                    if not state_match:
                        # 嘗試匹配帶空格的版本
                        state_match = re.search(r'(H\s*V\s*H\s*A|H\s*V\s*L\s*A|L\s*V\s*H\s*A|L\s*V\s*L\s*A)', reasoning_content, re.IGNORECASE)
                        if state_match:
                            state_str = state_match.group(1).replace(' ', '').upper()
                            state_match = type('obj', (object,), {'group': lambda self, n: state_str})()

                    # 提取學習狀態 - 更寬鬆的匹配
                    learning_match = re.search(r'(Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused)', reasoning_content, re.IGNORECASE)
                    if not learning_match:
                        # 嘗試匹配帶空格的版本
                        learning_patterns = [
                            r'A\s*n\s*x\s*i\s*o\s*u\s*s[^,]*S\s*t\s*r\s*e\s*s\s*s\s*e\s*d',
                            r'E\s*n\s*g\s*a\s*g\s*e\s*d[^,]*C\s*u\s*r\s*i\s*o\s*u\s*s',
                            r'F\s*o\s*c\s*u\s*s\s*e\s*d[^,]*F\s*l\s*o\s*w',
                            r'D\s*i\s*s\s*e\s*n\s*g\s*a\s*g\s*e\s*d[^,]*C\s*o\s*n\s*f\s*u\s*s\s*e\s*d',
                        ]
                        for pattern in learning_patterns:
                            match = re.search(pattern, reasoning_content, re.IGNORECASE)
                            if match:
                                # 簡化提取的結果
                                if 'nxious' in match.group(0).lower() or 'tressed' in match.group(0).lower():
                                    learning_match = type('obj', (object,), {'group': lambda self, n: 'Anxious/Stressed'})()
                                elif 'ngaged' in match.group(0).lower() or 'urious' in match.group(0).lower():
                                    learning_match = type('obj', (object,), {'group': lambda self, n: 'Engaged/Curious'})()
                                elif 'ocused' in match.group(0).lower() or 'low' in match.group(0).lower():
                                    learning_match = type('obj', (object,), {'group': lambda self, n: 'Focused/Flow'})()
                                elif 'isengaged' in match.group(0).lower() or 'onfused' in match.group(0).lower():
                                    learning_match = type('obj', (object,), {'group': lambda self, n: 'Disengaged/Confused'})()
                                break

                    # 提取信心水平 - 更寬鬆的匹配
                    conf_match = re.search(r'Confidence\s*(Level)?\s*[:：]\s*(High|Medium|Low)', reasoning_content, re.IGNORECASE)
                    if not conf_match:
                        # 嘗試匹配帶空格的版本
                        conf_match = re.search(r'C\s*o\s*n\s*f\s*i\s*d\s*e\s*n\s*c\s*e[^:]*[:：]\s*(H\s*i\s*g\s*h|M\s*e\s*d\s*i\s*u\s*m|L\s*o\s*w)', reasoning_content, re.IGNORECASE)
                        if conf_match:
                            conf_str = ''.join(re.findall(r'[HML][a-z]+', conf_match.group(1), re.IGNORECASE))
                            if conf_str:
                                conf_match = type('obj', (object,), {'group': lambda self, n: conf_str.capitalize() if n == 2 else None})()

                    # 構建基本內容
                    extracted_content = ""
                    if state_match:
                        state_val = state_match.group(1) if hasattr(state_match.group(1), 'upper') else state_match.group(1)
                        extracted_content += f"**1. Inferred Psychophysiological State:** {state_val}\n\n"
                    if learning_match:
                        learning_val = learning_match.group(1) if hasattr(learning_match.group(1), 'split') else learning_match.group(1)
                        extracted_content += f"**2. [REVISED] Inferred Affective State / Correlate:** {learning_val}\n\n"
                        extracted_content += f"**3. [REVISED] Inferred Learning State Correlate:** {learning_val}\n\n"
                    if conf_match:
                        conf_val = conf_match.group(2) if conf_match.group(2) else conf_match.group(1)
                        extracted_content += f"**4. [REVISED] Confidence Level:** {conf_val}\n\n"

                    # 嘗試提取關鍵證據和限制說明
                    rationale_match = re.search(r'Key\s+Rationale[^:]*[:：]([^6]*?)(?:Notes|$)', reasoning_content, re.IGNORECASE | re.DOTALL)
                    if rationale_match:
                        rationale_text = rationale_match.group(1).strip()[:500]  # 限制長度
                        if len(rationale_text) > 20:
                            extracted_content += f"**5. [REVISED] Key Rationale and Evidence:** {rationale_text}\n\n"

                    notes_match = re.search(r'Notes\s+on\s+Input\s+Limitations[^:]*[:：](.*?)(?:$|</)', reasoning_content, re.IGNORECASE | re.DOTALL)
                    if notes_match:
                        notes_text = notes_match.group(1).strip()[:500]  # 限制長度
                        if len(notes_text) > 20:
                            extracted_content += f"**6. [REVISED] Notes on Input Limitations:** {notes_text}\n\n"

                    # 如果提取到了內容，使用它
                    if extracted_content:
                        # 保留原有的標籤（如果存在），但添加提取的內容
                        if content:
                            new_content = content + "\n\n" + extracted_content
                        else:
                            new_content = extracted_content
                        return text[: m.start(1)] + new_content + text[m.end(1):]

            # 正常處理：添加缺失的標籤
            missing = [label for label in required_labels if label not in content]
            if missing:
                injected = "\n".join(missing) + "\n"
                new_content = injected + content.lstrip('\n') if content else injected
                return text[: m.start(1)] + new_content + text[m.end(1):]
            return text

        decoded = ensure_answer_labels(decoded)

        # ★★★ 修正 B：從 <answer> 區塊提取狀態（後處理校正）★★★
        reasoned_state = None
        reasoned_learning = None

        # 從 <answer> 區塊提取狀態（在截斷之前）
        answer_match = re.search(r'<answer>([\s\S]*?)</answer>', decoded, re.IGNORECASE)
        if answer_match:
            answer_content = answer_match.group(1)

            # 尋找明確的狀態標籤（優先從 "Inferred Psychophysiological State" 提取）
            state_patterns = [
                r'Inferred\s+Psychophysiological\s+State[^:]*[:：]\s*(HVHA|HVLA|LVHA|LVLA)',
                r'\b(HVHA|HVLA|LVHA|LVLA)\b'
            ]

            for pattern in state_patterns:
                state_match = re.search(pattern, answer_content, re.IGNORECASE)
                if state_match:
                    reasoned_state = state_match.group(1).upper()
                    break

            # 尋找 Learning State（優先從 "Inferred Learning State Correlate" 提取）
            learning_patterns = [
                r'Inferred\s+Learning\s+State\s+Correlate[^:]*[:：]\s*(Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused)',
                r'\b(Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused)\b'
            ]

            for pattern in learning_patterns:
                learning_match = re.search(pattern, answer_content, re.IGNORECASE)
                if learning_match:
                    reasoned_learning = learning_match.group(1)
                    break

        # 在 </answer>（若存在）或任何舊有 State: 前截斷
        end_idx = decoded.find('</answer>')
        if end_idx != -1:
            decoded = decoded[: end_idx + len('</answer>')]
        else:
            m_state_line = re.search(r'^State:\s*.*$', decoded, flags=re.MULTILINE)
            if m_state_line:
                decoded = decoded[: m_state_line.start()]

        _, zscore_warning = validate_zscore_usage(
            decoded,
            sample_data.get('zscore_features')
        )

        # Learning 映射表（用於備用映射）
        learning_map = {
            'HVLA': 'Focused/Flow',
            'HVHA': 'Engaged/Curious',
            'LVHA': 'Anxious/Stressed',
            'LVLA': 'Disengaged/Confused',
        }

        # ★★★ 優先級提取策略：1) <answer> 區塊 > 2) 最後的 State:/Learning:/Confidence: 行 > 3) 文本搜索 > 4) 映射 ★★★

        # 優先使用從 <answer> 區塊提取的值
        final_state = reasoned_state
        final_learning = reasoned_learning
        final_conf = None

        # 如果 <answer> 中沒有提取到，嘗試從最後的 State:/Learning:/Confidence: 行提取
        if not final_state:
            final_state_match = re.search(r'^State:\s*(HVHA|HVLA|LVHA|LVLA)\s*$', decoded, re.MULTILINE | re.IGNORECASE)
            if final_state_match:
                final_state = final_state_match.group(1).upper()

        if not final_learning:
            final_learning_match = re.search(r'^Learning:\s*(Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused)\s*$', decoded, re.MULTILINE | re.IGNORECASE)
            if final_learning_match:
                final_learning = final_learning_match.group(1)

        final_conf_match = re.search(r'^Confidence:\s*(High|Medium|Low)\s*$', decoded, re.MULTILINE | re.IGNORECASE)
        if final_conf_match:
            final_conf = final_conf_match.group(1).capitalize()

        # 如果還是沒有，使用文本搜索作為備用
        if not final_state:
            state_match = re.search(r'\b(HVLA|HVHA|LVHA|LVLA)\b', decoded)
            final_state = state_match.group(1).upper() if state_match else None

        if not final_learning and final_state:
            # 如果沒有明確的 Learning，根據 State 映射
            final_learning = learning_map.get(final_state, None)

        if not final_conf:
            conf_match = re.search(r'Confidence\s*(Level)?\s*[:：]\s*(High|Medium|Low)', decoded, flags=re.IGNORECASE)
            final_conf = conf_match.group(2).capitalize() if conf_match else 'Medium'

        # 如果從 <answer> 提取到了值但與最後的行不一致，記錄警告
        if reasoned_state and final_state and reasoned_state != final_state:
            print(f"[{subject} {trial}] ⚠️  狀態不一致: <answer>={reasoned_state}, 最後行={final_state}，優先使用 <answer>")
            final_state = reasoned_state
            if reasoned_state in learning_map:
                final_learning = learning_map[reasoned_state]

        if reasoned_learning and final_learning and reasoned_learning != final_learning:
            print(f"[{subject} {trial}] ⚠️  學習狀態不一致: <answer>={reasoned_learning}, 最後行={final_learning}，優先使用 <answer>")
            final_learning = reasoned_learning

        # ★★★ 標籤一致性檢查（僅記錄，不強制修改）★★★
        # 保留模型原始判斷，讓研究者自行分析 State 與 Learning 的關係
        STATE_TO_LEARNING_MAP = {
            "HVHA": "Engaged/Curious",    # 高效價+高喚醒 = 投入/好奇
            "HVLA": "Focused/Flow",       # 高效價+低喚醒 = 專注/心流
            "LVHA": "Anxious/Stressed",   # 低效價+高喚醒 = 焦慮/壓力
            "LVLA": "Disengaged/Confused" # 低效價+低喚醒 = 脫離/困惑
        }
        
        # 僅記錄不一致，但保留模型原始判斷
        if final_state and final_state in STATE_TO_LEARNING_MAP:
            expected_learning = STATE_TO_LEARNING_MAP[final_state]
            if final_learning and final_learning.lower() != expected_learning.lower():
                print(f"[{subject} {trial}] ⚠️  標籤不一致提醒: State='{final_state}' 但 Learning='{final_learning}' (預期: {expected_learning})，保留模型原始判斷")
        
        # 如果只有 State 沒有 Learning，使用映射補全（這是必要的）
        if final_state and not final_learning:
            final_learning = STATE_TO_LEARNING_MAP.get(final_state, "Unknown")
            print(f"[{subject} {trial}] 📝 標籤補全: State '{final_state}' -> Learning '{final_learning}' (模型未輸出 Learning)")

        # 確保最後總是有明確的三行摘要（用於提取）
        # 如果提取到了狀態和學習標籤，確保最後有明確的三行
        if final_state and final_learning:
            # 移除可能存在的舊的 State:/Learning:/Confidence: 行
            decoded = re.sub(r'^State:\s*.*$', '', decoded, flags=re.MULTILINE)
            decoded = re.sub(r'^Learning:\s*.*$', '', decoded, flags=re.MULTILINE)
            decoded = re.sub(r'^Confidence:\s*.*$', '', decoded, flags=re.MULTILINE)
            # 清理多餘空行
            decoded = re.sub(r'\n{3,}', '\n\n', decoded)
            # 在最後追加明確的三行（格式：每行一個標籤，值精確匹配）
            decoded = decoded.rstrip('\n') + "\n\n" + \
                f"State: {final_state}\n" + \
                f"Learning: {final_learning}\n" + \
                f"Confidence: {final_conf}"

        # ★★★ 格式驗證 ★★★
        validation_result = validate_output_format(
            decoded, 
            subject, 
            trial
        )
        if zscore_warning:
            validation_result['warnings'].append(zscore_warning)

        # 構建驗證報告（已註釋，不再使用）
        validation_report = ""

        # 準備完整的報告內容（包含 Step1–Step8 子報告與最終結論）
        valence = sample_data.get('valence')
        arousal = sample_data.get('arousal')
        valence_str = f"{valence:.2f}" if valence is not None else "N/A"
        arousal_str = f"{arousal:.2f}" if arousal is not None else "N/A"

        # 從 MODEL_ID 提取簡潔的模型名稱
        # model_name = MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID.split(os.sep)[-1]
        
        report_content = (
            f"========== {model_name} {'+ PIKE-RAG' if rag_retriever else ''} 分析結果 ==========\n"
            f"原始情緒標籤 (供參考): Valence={int(valence) if valence is not None else 'N/A'}/5, Arousal={int(arousal) if arousal is not None else 'N/A'}/5\n"
            f"受試者: {subject}, 試驗: {trial}\n"
            f"========== 系統配置 ==========\n"
            f"RAG 增強: {'✅ 已啟用' if rag_retriever else '❌ 未啟用'}\n"
            f"Guardrails: {'✅ 已啟用' if ENABLE_GUARDRAILS else '❌ 未啟用'}\n"
            f"Delta Z-score: {'✅ 已啟用' if ENABLE_DELTA_ZSCORE else '❌ 未啟用'}\n"
            f"4-bit 量化: {'✅ 已啟用' if USE_4BIT else '❌ 未啟用'}\n"
            f"EEG 分析: {'✅ 已啟用' if ENABLE_EEG_ANALYSIS else '❌ 未啟用'}\n"
            f"RAG 檢索結果: {len(retrieved_docs) if retrieved_docs else 0} 個相關文件\n"
            "----------------------------------------\n"
            "Step 1 - Signal Quality Report:\n"
            f"{step1_text}\n\n"
            "Step 2 - Time-domain HRV Report:\n"
            f"{step2_text}\n\n"
            "Step 3 - Frequency-domain HRV Report:\n"
            f"{step3_text}\n\n"
            "Step 4 - Poincaré / Complexity Report:\n"
            f"{step4_text}\n\n"
            "Step 5 - Baseline Delta Report:\n"
            f"{step5_text}\n\n"
            "Step 6 - Within-subject Baseline Report:\n"
            f"{step6_text}\n\n"
            "Step 7 - EEG / Multimodal Report:\n"
            f"{step7_text}\n\n"
            "Step 8 - Clinical Knowledge / RAG Summary:\n"
            f"{rag_summary_text}\n\n"
            "===== Final Integrated Report (Template-constrained) =====\n"
            f"{decoded}\n"
            f"{validation_report}"
            "========================================\n"
        )

        with open(output_file_path, "w", encoding='utf-8') as f:
            f.write(report_content)

        # 計算總時間
        timing_data['total'] = time.time() - overall_start_time
        print(f"[{subject} {trial}] ✓ 所有步驟完成，總耗時: {timing_data['total']:.2f} 秒")

        # 收集時間數據到統一列表（用於後續統合分析）
        timing_entry = {
            "subject": subject,
            "trial": trial,
            "timing_data": timing_data  # 保存原始 timing_data 以便 CSV 輸出
        }
        timing_results.append(timing_entry)

    # 處理完最後一個 subject 後也清理一次
    if previous_subject is not None:
        print(f"\n[{previous_subject}] ✓ 所有 trials 處理完成，清理 CUDA 緩存和內存...")
        clean_catch()
        print(f"[{previous_subject}] ✓ 清理完成\n")

    # 保存時間數據為 CSV 文件
    print("\n" + "="*80)
    print("正在生成時間統計 CSV 報告...")
    timing_csv_path = os.path.join(OUTPUT_DIR, "timing_statistics.csv")

    # 定義 CSV 列名：第一列是 metadata，然後是各個步驟
    csv_columns = [
        "metadata",  # 第一列：subject_trial 作為 metadata
        "rag_retrieval",
        "step1_quality",
        "step2_time_domain",
        "step3_frequency",
        "step4_poincare",
        "step5_baseline_delta",
        "step6_within_subject",
        "step7_eeg",
        "step8_integration",
        "total"
    ]

    # 準備 CSV 數據
    csv_rows = []
    for entry in timing_results:
        subject_id = entry.get("subject", "unknown")
        trial_id = entry.get("trial", "unknown")
        metadata = f"{subject_id}_{trial_id}"  # 第一列作為 metadata

        # 直接從 timing_data 中提取時間數據
        timing_data_dict = entry.get("timing_data", {})

        # 構建 CSV 行數據
        row = {
            "metadata": metadata,
            "rag_retrieval": round(timing_data_dict.get("rag_retrieval", 0.0), 4),
            "step1_quality": round(timing_data_dict.get("step1_quality", 0.0), 4),
            "step2_time_domain": round(timing_data_dict.get("step2_time_domain", 0.0), 4),
            "step3_frequency": round(timing_data_dict.get("step3_frequency", 0.0), 4),
            "step4_poincare": round(timing_data_dict.get("step4_poincare", 0.0), 4),
            "step5_baseline_delta": round(timing_data_dict.get("step5_baseline_delta", 0.0), 4),
            "step6_within_subject": round(timing_data_dict.get("step6_within_subject", 0.0), 4),
            "step7_eeg": round(timing_data_dict.get("step7_eeg", 0.0), 4),
            "step8_integration": round(timing_data_dict.get("step8_integration", 0.0), 4),
            "total": round(timing_data_dict.get("total", 0.0), 4)
        }
        csv_rows.append(row)

    # 寫入 CSV 文件
    with open(timing_csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"✓ 時間統計 CSV 已保存至: {timing_csv_path}")

    print("\n" + "="*80)
    print("✓ 所有樣本分析完成！")
    print(f"✓ 報告已保存至: {OUTPUT_DIR}")
    print("="*80)


if __name__ == '__main__':
    main()
