#!/usr/bin/env python3
"""
MedGemma + PIKE-RAG 整合版本 - 數據集模組
包含 HRVEmotionDataset 類別定義
"""

import polars as pl
from PIL import Image
from torch.utils.data import Dataset
import os
from typing import Dict, Any, List


class HRVEmotionDataset(Dataset):
    """
    從 CSV 讀取 HRV 數據的 Dataset 類別
    """
    def __init__(self, csv_path):
        # 讀取最終 HRV 數據（含圖片路徑與影像特徵）
        self.data = pl.read_csv(csv_path)

        # 所有的 HRV 原始特徵欄位
        self.raw_feature_columns = [
            "MeanRR_ms",
            "SDNN_ms",
            "MeanHR_bpm",
            "SDHR_bpm",
            "RMSSD_ms",
            "NN50",
            "pNN50",
            "SDNN_index_ms",
            "ULF_peak",
            "LF_peak",
            "HF_peak",
            "ULF_ratio",
            "LF_ratio",
            "HF_ratio",
            "LF_HF",
            "SD1_ms",
            "SD2_ms",
            "SampEn",
            "DFA_alpha",
        ]

        # Baseline 差異特徵（Delta_ 前綴，含百分比欄位）
        self.delta_feature_columns = sorted(
            [
                col
                for col in self.data.columns
                if col.startswith("Delta_")
            ]
        )

        # EEG 特徵欄位（多通道 + 區域平均）
        self.eeg_feature_columns = sorted(
            [
                col
                for col in self.data.columns
                if col.startswith("EEG_")
            ]
        )

        # EDR (ECG-derived Respiration) 特徵欄位
        self.edr_feature_columns = sorted(
            [
                col
                for col in self.data.columns
                if col.startswith("respiratory_rate") or col.startswith("edr_")
            ]
        )

        # 關鍵的 Z-score 補充欄位
        # 包含兩種 Z-score：
        # 1. 傳統 Z-score（基於全域分佈）：{feat}_zscore
        # 2. Delta Z-score（基於個人 Baseline，與 Delta 方向一致）：Delta_{feat}_zscore
        self.zscore_columns = [col for col in self.data.columns if "_zscore" in col]
        
        # 分離傳統 Z-score 和 Delta Z-score（用於報告中的一致性檢查）
        self.traditional_zscore_columns = [
            col for col in self.zscore_columns 
            if not col.startswith("Delta_")
        ]
        self.delta_zscore_columns = [
            col for col in self.zscore_columns 
            if col.startswith("Delta_")
        ]

        # 訊號品質欄位（擴展版：包含新的品質指標）
        self.quality_columns = [
            # 原有品質指標
            "artifact_rate",
            "ectopy_count",
            "ectopy_flag",
            "valid_rr_ratio",
            "window_start_s",
            "window_end_s",
            "window_duration_s",
            # === 新增：綜合品質指標 ===
            "overall_quality",
            "data_length_quality",
            "artifact_quality",
            "nonlinear_quality",
            "respiratory_interference",
            "quality_flags_str",
            # === 新增：頻域品質指標 ===
            "HF_RSA_overlap",
            "HF_reliability",
            "ULF_dominance_flag",
            "spectral_quality",
            # === 新增：SampEn 品質指標 ===
            "SampEn_confidence",
            "SampEn_reliability",
        ]

        # 影像量化特徵欄位（圖像像素特徵）
        self.image_feature_columns = [
            "poincare_img_mean",
            "poincare_img_std",
            "poincare_row_slope",
            "signal_quality_img_mean",
            "signal_quality_img_std",
            "signal_quality_row_slope",
            "psd_img_mean",
            "psd_img_std",
            "psd_row_slope",
        ]

        # === 新增：從原始數據提取的量化特徵（避免圖像縮放失真） ===
        # Poincaré 圖的實際數據量化特徵
        self.poincare_data_feature_columns = [
            "poincare_rri_x_min",
            "poincare_rri_x_max",
            "poincare_rri_x_mean",
            "poincare_rri_x_std",
            "poincare_rri_y_min",
            "poincare_rri_y_max",
            "poincare_rri_y_mean",
            "poincare_rri_y_std",
            "poincare_rri_range_x",
            "poincare_rri_range_y",
            "poincare_rri_ratio_xy",
            "poincare_density_center_x",
            "poincare_density_center_y",
            "poincare_scatter_count",
        ]

        # PSD 圖的實際數據量化特徵
        self.psd_data_feature_columns = [
            "psd_freq_min",
            "psd_freq_max",
            "psd_freq_range",
            "psd_power_min",
            "psd_power_max",
            "psd_power_mean",
            "psd_power_std",
            "psd_power_median",
            "psd_peak_freq",
            "psd_peak_power",
            "psd_total_power",
            "psd_band_ulf_power",
            "psd_band_lf_power",
            "psd_band_hf_power",
        ]

        # Signal Quality panel 的實際數據量化特徵
        self.signal_quality_data_feature_columns = [
            "sq_ecg_amplitude_min",
            "sq_ecg_amplitude_max",
            "sq_ecg_amplitude_range",
            "sq_ecg_amplitude_mean",
            "sq_ecg_amplitude_std",
            "sq_rri_min",
            "sq_rri_max",
            "sq_rri_range",
            "sq_time_duration",
        ]

        # 為 Step5 準備 within-subject baseline 統計
        baseline_feature_cols = [
            "RMSSD_ms",
            "SDNN_ms",
            "MeanHR_bpm",
            "SampEn",
            "DFA_alpha",
        ]
        agg_exprs = []
        for col in baseline_feature_cols:
            if col in self.data.columns:
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_baseline_mean"))
                agg_exprs.append(pl.col(col).std().alias(f"{col}_baseline_std"))

        if agg_exprs and "subject" in self.data.columns:
            baseline_df = self.data.group_by("subject").agg(agg_exprs)
            # 將 baseline 統計併回原始資料
            self.data = self.data.join(baseline_df, on="subject", how="left")
            # 記下 baseline 欄位名稱，方便 __getitem__ 取出
            self.baseline_columns = [expr.meta.output_name for expr in agg_exprs]  # type: ignore[attr-defined]
        else:
            self.baseline_columns = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.row(idx, named=True)

        # 提取所有需要的數據
        raw_features = {name: sample.get(name) for name in self.raw_feature_columns if name in self.data.columns}
        zscore_features = {name: sample.get(name) for name in self.zscore_columns}
        quality_features = {name: sample.get(name) for name in self.quality_columns if name in self.data.columns}
        image_features = {name: sample.get(name) for name in self.image_feature_columns if name in self.data.columns}
        delta_features = {name: sample.get(name) for name in self.delta_feature_columns}
        eeg_features = {name: sample.get(name) for name in self.eeg_feature_columns}
        edr_features = {name: sample.get(name) for name in self.edr_feature_columns}
        baseline_features = {name: sample.get(name) for name in getattr(self, "baseline_columns", [])}

        # === 新增：提取從原始數據提取的量化特徵（避免圖像縮放失真） ===
        poincare_data_features = {name: sample.get(name) for name in self.poincare_data_feature_columns if name in self.data.columns}
        psd_data_features = {name: sample.get(name) for name in self.psd_data_feature_columns if name in self.data.columns}
        signal_quality_data_features = {name: sample.get(name) for name in self.signal_quality_data_feature_columns if name in self.data.columns}

        # 主要 Poincaré 圖
        image_path = sample.get("img_path")
        image = None
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = None

        # 訊號品質 panel 圖
        sq_image_path = sample.get("signal_quality_img_path")
        signal_quality_image = None
        if sq_image_path is not None:
            try:
                signal_quality_image = Image.open(sq_image_path).convert("RGB")
            except Exception:
                signal_quality_image = None

        # PSD 圖
        psd_image_path = sample.get("psd_img_path")
        psd_image = None
        if psd_image_path is not None:
            try:
                psd_image = Image.open(psd_image_path).convert("RGB")
            except Exception:
                psd_image = None

        # === 新增：分離傳統 Z-score 和 Delta Z-score ===
        traditional_zscores = {
            name: sample.get(name) 
            for name in self.traditional_zscore_columns 
            if name in self.data.columns
        }
        delta_zscores = {
            name: sample.get(name) 
            for name in self.delta_zscore_columns 
            if name in self.data.columns
        }

        return {
            "raw_features": raw_features,
            "zscore_features": zscore_features,  # 所有 Z-score（向後兼容）
            "traditional_zscores": traditional_zscores,  # 傳統 Z-score（基於全域分佈）
            "delta_zscores": delta_zscores,  # Delta Z-score（與 Delta 方向一致）
            "quality_features": quality_features,
            "image_features": image_features,
            "delta_features": delta_features,
            "eeg_features": eeg_features,
            "edr_features": edr_features,  # EDR (ECG-derived Respiration) 特徵
            "poincare_data_features": poincare_data_features,  # Poincaré 實際數據特徵
            "psd_data_features": psd_data_features,  # PSD 實際數據特徵
            "signal_quality_data_features": signal_quality_data_features,  # Signal Quality 實際數據特徵
            **baseline_features,
            "image": image,
            "image_path": image_path,
            "signal_quality_image": signal_quality_image,
            "signal_quality_img_path": sq_image_path,
            "psd_image": psd_image,
            "psd_img_path": psd_image_path,
            "valence": sample.get("valence"),
            "arousal": sample.get("arousal"),
            "subject": sample.get("subject"),
            "trial": sample.get("trial"),
            "age": sample.get("age"),
            "gender": sample.get("gender"),
        }
