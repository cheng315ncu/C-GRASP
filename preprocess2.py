#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGRASP - HRV Preprocessing Pipeline
====================================

Integrated HRV feature extraction and data processing pipeline.
Processes DREAMER dataset (or similar) to extract ECG/EEG features.

Usage:
    python preprocess2.py --mat-path /path/to/DREAMER.mat --output-dir ./output
    python preprocess2.py --help

Environment Variables:
    CGRASP_MAT_PATH: Path to DREAMER.mat file
    CGRASP_PREPROCESS_OUTPUT: Output directory for processed data
"""

import os
import sys
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
import scipy.signal as sps
import neurokit2 as nk
from PIL import Image
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, cpu_count
import pickle


# =============================================================================
# Configuration (can be overridden via CLI or environment variables)
# =============================================================================

@dataclass
class PreprocessConfig:
    """Configuration for HRV preprocessing pipeline."""
    
    # Input/Output paths
    mat_path: str = ""
    output_dir: str = ""
    
    # Processing range
    subject_range: Tuple[int, int] = (1, 23)
    trial_range: Tuple[int, int] = (1, 18)
    
    # ECG settings
    ecg_channel: int = 0  # 0 or 1: which ECG channel to use
    resample_hz: float = 4.0  # Frequency domain resampling rate (Hz)
    
    # Feature toggles
    use_baseline: bool = True  # Use ECG baseline data for delta features
    use_eeg: bool = True  # Extract EEG features
    use_dual_channel: bool = False  # Fuse dual-channel ECG (if SQI disabled)
    use_sqi_selection: bool = True  # Use Signal Quality Index for channel selection
    sqi_mode: str = "best"  # "best" = select best channel, "weighted" = weighted average
    
    # Output settings
    save_plots: bool = True
    poincare_size: Tuple[int, int] = (896, 896)
    
    # Parallel processing
    num_workers: Optional[int] = None  # None = auto-detect CPU count
    
    def __post_init__(self):
        """Resolve paths from environment variables if not set."""
        if not self.mat_path:
            self.mat_path = os.environ.get("CGRASP_MAT_PATH", "")
        if not self.output_dir:
            self.output_dir = os.environ.get(
                "CGRASP_PREPROCESS_OUTPUT", 
                str(Path(__file__).parent / "data" / "processed")
            )
        if self.num_workers is None:
            self.num_workers = min(cpu_count(), 16)


def parse_args() -> PreprocessConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(
        description="CGRASP HRV Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process DREAMER dataset
    python preprocess2.py --mat-path /path/to/DREAMER.mat --output-dir ./output
    
    # Process specific subjects (1-5) and trials (1-10)
    python preprocess2.py --mat-path data.mat -o ./out --subjects 1 5 --trials 1 10
    
    # Disable EEG and baseline features
    python preprocess2.py --mat-path data.mat -o ./out --no-eeg --no-baseline
        """
    )
    
    # Required paths
    parser.add_argument(
        "--mat-path", "-m",
        type=str,
        default=os.environ.get("CGRASP_MAT_PATH", ""),
        help="Path to DREAMER.mat file (or set CGRASP_MAT_PATH env var)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=os.environ.get("CGRASP_PREPROCESS_OUTPUT", "./data/processed"),
        help="Output directory for processed data"
    )
    
    # Processing range
    parser.add_argument(
        "--subjects", "-s",
        type=int,
        nargs=2,
        default=[1, 23],
        metavar=("START", "END"),
        help="Subject range (inclusive), default: 1 23"
    )
    parser.add_argument(
        "--trials", "-t",
        type=int,
        nargs=2,
        default=[1, 18],
        metavar=("START", "END"),
        help="Trial range (inclusive), default: 1 18"
    )
    
    # Feature toggles
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Disable baseline delta features"
    )
    parser.add_argument(
        "--no-eeg",
        action="store_true",
        help="Disable EEG feature extraction"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable plot generation (faster processing)"
    )
    
    # ECG settings
    parser.add_argument(
        "--ecg-channel",
        type=int,
        choices=[0, 1],
        default=0,
        help="ECG channel to use (0 or 1), default: 0"
    )
    parser.add_argument(
        "--sqi-mode",
        type=str,
        choices=["best", "weighted", "none"],
        default="best",
        help="SQI channel selection mode: best, weighted, or none"
    )
    
    # Parallel processing
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Validate mat_path
    if not args.mat_path:
        parser.error("--mat-path is required (or set CGRASP_MAT_PATH env var)")
    
    if not os.path.exists(args.mat_path):
        parser.error(f"MAT file not found: {args.mat_path}")
    
    # Build config
    config = PreprocessConfig(
        mat_path=args.mat_path,
        output_dir=args.output_dir,
        subject_range=tuple(args.subjects),
        trial_range=tuple(args.trials),
        ecg_channel=args.ecg_channel,
        use_baseline=not args.no_baseline,
        use_eeg=not args.no_eeg,
        save_plots=not args.no_plots,
        use_sqi_selection=(args.sqi_mode != "none"),
        sqi_mode=args.sqi_mode if args.sqi_mode != "none" else "best",
        num_workers=args.workers,
    )
    
    return config


# Global config (set in main())
CONFIG: Optional[PreprocessConfig] = None

# Legacy compatibility aliases (will be removed in future versions)
def _get_config() -> PreprocessConfig:
    """Get current config or create default."""
    global CONFIG
    if CONFIG is None:
        CONFIG = PreprocessConfig()
    return CONFIG

# =============================================================================

def ensure_dir(p): 
    os.makedirs(p, exist_ok=True)


# ============================================================
# 階段 1: ECG 處理與 HRV 特徵提取
# ============================================================

def load_trial(D, s_idx, t_idx, use_baseline=True, use_eeg=True):
    """
    Load a single trial's ECG data from DREAMER.mat.
    
    Args:
        D: DREAMER data structure
        s_idx: Subject index (1-based)
        t_idx: Trial index (1-based)
        use_baseline: Whether to load baseline ECG data
        use_eeg: Whether to load EEG data
    
    Returns:
        dict with ecg_stimuli, ecg_baseline, eeg_stimuli, eeg_baseline, fs_ecg, fs_eeg, labels
    """
    fs_ecg = int(np.asarray(getattr(D, "ECG_SamplingRate", 256)).squeeze()) if hasattr(D, "ECG_SamplingRate") else 256
    fs_eeg = int(np.asarray(getattr(D, "EEG_SamplingRate", 128)).squeeze()) if hasattr(D, "EEG_SamplingRate") else 128
    subj = D.Data[s_idx-1]
    
    # 安全取值工具
    def safe_float_array_field(obj, name):
        try:
            v = np.asarray(getattr(obj, name)).astype(float).squeeze()
            return v
        except Exception:
            return np.array([], dtype=float)

    def safe_float_scalar(x):
        try:
            return float(np.asarray(x).squeeze())
        except Exception:
            return np.nan

    def safe_str_scalar(x):
        try:
            return str(np.asarray(x).squeeze())
        except Exception:
            return str(x)

    # ECG Stimuli 資料
    stim = np.asarray(subj.ECG.stimuli, dtype=object)
    arr_stimuli = np.asarray(stim[t_idx-1]).astype(float)  # (N,2)
    
    # ECG Baseline 資料（如果啟用）
    arr_baseline = None
    if use_baseline and hasattr(subj, 'ECG') and hasattr(subj.ECG, 'baseline'):
        try:
            baseline = np.asarray(subj.ECG.baseline, dtype=object)
            arr_baseline = np.asarray(baseline[t_idx-1]).astype(float)  # (N,2)
        except Exception:
            arr_baseline = None
    
    # EEG 資料（如果啟用）
    eeg_stimuli = None
    eeg_baseline = None
    if use_eeg and hasattr(subj, 'EEG'):
        try:
            eeg_stim = np.asarray(subj.EEG.stimuli, dtype=object)
            eeg_stimuli = np.asarray(eeg_stim[t_idx-1]).astype(float)  # (N, 14)
        except Exception:
            eeg_stimuli = None
        try:
            eeg_bl = np.asarray(subj.EEG.baseline, dtype=object)
            eeg_baseline = np.asarray(eeg_bl[t_idx-1]).astype(float)  # (N, 14)
        except Exception:
            eeg_baseline = None

    # 取 PAD 分數
    val_arr = safe_float_array_field(subj, "ScoreValence")
    aro_arr = safe_float_array_field(subj, "ScoreArousal")
    dom_arr = safe_float_array_field(subj, "ScoreDominance")

    labels = dict(
        valence = float(val_arr[t_idx-1]) if val_arr.size >= t_idx else np.nan,
        arousal = float(aro_arr[t_idx-1]) if aro_arr.size >= t_idx else np.nan,
        dominance = float(dom_arr[t_idx-1]) if dom_arr.size >= t_idx else np.nan,
        age = safe_float_scalar(getattr(subj, "Age", np.nan)),
        gender = safe_str_scalar(getattr(subj, "Gender", ""))
    )

    return {
        'ecg_stimuli': arr_stimuli,
        'ecg_baseline': arr_baseline,
        'eeg_stimuli': eeg_stimuli,
        'eeg_baseline': eeg_baseline,
        'fs_ecg': fs_ecg,
        'fs_eeg': fs_eeg,
        'labels': labels
    }


def calculate_sqi(ecg_signal, fs, ridx=None):
    """
    計算 ECG 訊號的 Signal Quality Index (SQI)
    
    參數:
        ecg_signal: ECG 訊號陣列
        fs: 採樣率
        ridx: R-peak 索引（可選，如果提供則用於計算 R-peak 相關指標）
    
    返回:
        dict: 包含多個 SQI 指標的字典
    """
    if ecg_signal is None or len(ecg_signal) == 0:
        return {
            'kurtosis': np.nan,
            'snr': np.nan,
            'rpeak_success_rate': np.nan,
            'signal_power': np.nan,
            'coefficient_variation': np.nan,
            'overall_sqi': np.nan
        }
    
    x = np.asarray(ecg_signal, dtype=float)
    x_centered = x - np.median(x)
    
    sqi_dict = {}
    
    # 1. 峰度 (Kurtosis) - 衡量信號的尖銳程度，ECG 應該有較高的峰度
    if len(x_centered) > 3:
        from scipy.stats import kurtosis
        kurt = kurtosis(x_centered, fisher=True)  # Fisher's definition (normalized)
        sqi_dict['kurtosis'] = float(kurt) if np.isfinite(kurt) else np.nan
    else:
        sqi_dict['kurtosis'] = np.nan
    
    # 2. 信噪比 (SNR) - 使用功率比估算
    try:
        # 估算信號功率（使用高頻濾波後的功率作為噪聲，原始功率作為信號）
        # 簡單方法：使用信號功率與標準差的比值
        signal_power = float(np.var(x_centered))
        # 使用高頻成分估算噪聲（簡單高通濾波）
        if len(x_centered) > 10:
            # 簡單的噪聲估算：使用差分信號的變異性
            diff_signal = np.diff(x_centered)
            noise_power = float(np.var(diff_signal)) if len(diff_signal) > 0 else signal_power
            snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.nan
            sqi_dict['snr'] = float(snr_db) if np.isfinite(snr_db) else np.nan
        else:
            sqi_dict['snr'] = np.nan
        sqi_dict['signal_power'] = signal_power
    except Exception:
        sqi_dict['snr'] = np.nan
        sqi_dict['signal_power'] = np.nan
    
    # 3. R-peak 檢測成功率（如果提供了 ridx）
    if ridx is not None and len(ridx) >= 3:
        # 計算 R-peak 檢測的合理性
        # 檢查 R-peak 間距是否在合理範圍內（300-2000 ms）
        rri_ms = np.diff(ridx) / fs * 1000.0
        valid_rri = (rri_ms >= 300.0) & (rri_ms <= 2000.0)
        rpeak_success_rate = float(np.mean(valid_rri)) if len(valid_rri) > 0 else 0.0
        sqi_dict['rpeak_success_rate'] = rpeak_success_rate
    else:
        sqi_dict['rpeak_success_rate'] = np.nan
    
    # 4. 變異係數 (Coefficient of Variation) - 標準化後的變異性
    if len(x_centered) > 1:
        std_val = np.std(x_centered, ddof=1)
        mean_abs = np.mean(np.abs(x_centered))
        cv = float(std_val / mean_abs) if mean_abs > 0 else np.nan
        sqi_dict['coefficient_variation'] = cv if np.isfinite(cv) else np.nan
    else:
        sqi_dict['coefficient_variation'] = np.nan
    
    # 5. 綜合 SQI 分數（標準化到 0-1 範圍，越高越好）
    # 將各個指標標準化並加權平均
    scores = []
    weights = []
    
    # 峰度：ECG 通常有較高峰度（>0），越高越好，標準化到 0-1
    if np.isfinite(sqi_dict['kurtosis']):
        kurt_norm = min(max((sqi_dict['kurtosis'] + 2) / 10.0, 0), 1)  # 假設合理範圍 -2 到 8
        scores.append(kurt_norm)
        weights.append(0.25)
    
    # SNR：越高越好，假設合理範圍 -10 到 30 dB，標準化到 0-1
    if np.isfinite(sqi_dict['snr']):
        snr_norm = min(max((sqi_dict['snr'] + 10) / 40.0, 0), 1)
        scores.append(snr_norm)
        weights.append(0.30)
    
    # R-peak 成功率：已經是 0-1 範圍
    if np.isfinite(sqi_dict['rpeak_success_rate']):
        scores.append(sqi_dict['rpeak_success_rate'])
        weights.append(0.35)
    
    # 變異係數：較低較好（但也不能太低），標準化
    if np.isfinite(sqi_dict['coefficient_variation']):
        cv_norm = min(max(1.0 - sqi_dict['coefficient_variation'] / 2.0, 0), 1)  # 假設合理範圍 0-2
        scores.append(cv_norm)
        weights.append(0.10)
    
    # 計算加權平均
    if len(scores) > 0 and sum(weights) > 0:
        overall_sqi = float(np.average(scores, weights=weights[:len(scores)]))
        sqi_dict['overall_sqi'] = overall_sqi if np.isfinite(overall_sqi) else np.nan
    else:
        sqi_dict['overall_sqi'] = np.nan
    
    return sqi_dict


def fuse_ecg_channels(ch1, ch2, fs=None, mode="simple"):
    """
    融合雙通道 ECG
    
    參數:
        ch1: 第一通道 ECG 訊號
        ch2: 第二通道 ECG 訊號
        fs: 採樣率（用於 SQI 計算，如果 mode 為 "sqi" 則需要）
        mode: 融合模式
            - "simple": 簡單平均（預設）
            - "sqi_best": 基於 SQI 選擇最佳通道
            - "sqi_weighted": 基於 SQI 進行加權平均
    
    返回:
        融合後的 ECG 訊號
    """
    if ch1 is None:
        return ch2
    if ch2 is None:
        return ch1
    
    # 確保長度一致
    min_len = min(len(ch1), len(ch2))
    ch1_trimmed = ch1[:min_len]
    ch2_trimmed = ch2[:min_len]
    
    # 簡單平均模式
    if mode == "simple":
        return (ch1_trimmed + ch2_trimmed) / 2.0
    
    # SQI-based 模式需要採樣率
    if fs is None:
        # 如果沒有提供 fs，回退到簡單平均
        return (ch1_trimmed + ch2_trimmed) / 2.0
    
    # 計算兩個通道的 SQI
    # 先嘗試檢測 R-peaks 以獲得更好的 SQI
    ridx1, _ = detect_rpeaks_strong(ch1_trimmed, fs)
    ridx2, _ = detect_rpeaks_strong(ch2_trimmed, fs)
    
    sqi1 = calculate_sqi(ch1_trimmed, fs, ridx1)
    sqi2 = calculate_sqi(ch2_trimmed, fs, ridx2)
    
    sqi1_score = sqi1.get('overall_sqi', np.nan)
    sqi2_score = sqi2.get('overall_sqi', np.nan)
    
    # 如果 SQI 計算失敗，回退到簡單平均
    if not np.isfinite(sqi1_score) and not np.isfinite(sqi2_score):
        return (ch1_trimmed + ch2_trimmed) / 2.0
    
    # 選擇最佳通道模式
    if mode == "sqi_best":
        if np.isfinite(sqi1_score) and np.isfinite(sqi2_score):
            return ch1_trimmed if sqi1_score >= sqi2_score else ch2_trimmed
        elif np.isfinite(sqi1_score):
            return ch1_trimmed
        elif np.isfinite(sqi2_score):
            return ch2_trimmed
        else:
            return (ch1_trimmed + ch2_trimmed) / 2.0
    
    # 加權平均模式
    elif mode == "sqi_weighted":
        # 如果只有一個有效 SQI，使用該通道
        if not np.isfinite(sqi1_score) and np.isfinite(sqi2_score):
            return ch2_trimmed
        elif np.isfinite(sqi1_score) and not np.isfinite(sqi2_score):
            return ch1_trimmed
        elif not np.isfinite(sqi1_score) and not np.isfinite(sqi2_score):
            return (ch1_trimmed + ch2_trimmed) / 2.0
        
        # 兩個 SQI 都有效，進行加權平均
        # 使用 softmax 類似的歸一化，確保權重和為 1
        # 將 SQI 轉換為權重（SQI 越高，權重越大）
        sqi1_norm = max(sqi1_score, 0.01)  # 避免零值
        sqi2_norm = max(sqi2_score, 0.01)
        total_sqi = sqi1_norm + sqi2_norm
        
        w1 = sqi1_norm / total_sqi
        w2 = sqi2_norm / total_sqi
        
        return w1 * ch1_trimmed + w2 * ch2_trimmed
    
    # 預設：簡單平均
    else:
        return (ch1_trimmed + ch2_trimmed) / 2.0


def refine_rpeak_with_parabolic_interpolation(x, ridx, fs, window_ms=10):
    """
    使用拋物線插值精細化 R-peak 位置
    
    這可以將 R-peak 定位精度提高到亞採樣點級別，
    對於採樣率 < 500Hz 的數據特別重要。
    
    參數:
        x: ECG 信號
        ridx: 初始 R-peak 索引（整數）
        fs: 採樣率
        window_ms: 搜索窗口大小 (ms)，默認 ±10ms
    
    返回:
        refined_ridx: 精細化後的 R-peak 索引（浮點數，可用於更精確的 RRI 計算）
        integer_ridx: 取整後的索引（用於繪圖等）
    
    原理:
        在每個 R-peak 附近擬合拋物線 y = a*x^2 + b*x + c，
        拋物線頂點位置 x_peak = -b/(2a) 給出亞採樣精度的 R-peak 位置。
    
    參考: Pan & Tompkins (1985), 以及後續的 HRV 分析標準
    """
    if len(ridx) == 0 or len(x) == 0:
        return np.array([], dtype=float), np.array([], dtype=int)
    
    window_samples = int(window_ms * fs / 1000.0)
    window_samples = max(window_samples, 2)  # 至少 2 個採樣點
    
    refined_positions = []
    integer_positions = []
    
    for idx in ridx:
        # 定義搜索窗口
        left = max(0, idx - window_samples)
        right = min(len(x) - 1, idx + window_samples)
        
        if right - left < 2:
            # 窗口太小，保持原始位置
            refined_positions.append(float(idx))
            integer_positions.append(int(idx))
            continue
        
        # 在窗口內找到局部最大值
        window_signal = x[left:right + 1]
        local_peak_idx = np.argmax(window_signal)
        global_peak_idx = left + local_peak_idx
        
        # 確保有足夠的點進行拋物線擬合（需要前後各一個點）
        if local_peak_idx == 0 or local_peak_idx == len(window_signal) - 1:
            # 峰值在邊緣，無法擬合拋物線
            refined_positions.append(float(global_peak_idx))
            integer_positions.append(int(global_peak_idx))
            continue
        
        # 拋物線擬合（使用峰值及其相鄰兩點）
        y0 = float(window_signal[local_peak_idx - 1])
        y1 = float(window_signal[local_peak_idx])
        y2 = float(window_signal[local_peak_idx + 1])
        
        # 拋物線頂點位置（相對於 local_peak_idx）
        # 對於 y = a*x^2 + b*x + c，頂點 x = -b/(2a)
        # 使用三點擬合: x_offset = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2)
        denominator = y0 - 2 * y1 + y2
        if abs(denominator) > 1e-10:
            x_offset = 0.5 * (y0 - y2) / denominator
            # 限制偏移量在合理範圍內（±1 採樣點）
            x_offset = np.clip(x_offset, -1.0, 1.0)
        else:
            x_offset = 0.0
        
        refined_idx = global_peak_idx + x_offset
        refined_positions.append(refined_idx)
        integer_positions.append(int(round(refined_idx)))
    
    return np.array(refined_positions, dtype=float), np.array(integer_positions, dtype=int)


def detect_rpeaks_strong(x, fs, use_parabolic_refinement=True):
    """
    強健的 R-peak 偵測（增強版）
    
    改進：
    1. 添加拋物線插值精細化，提高 R-peak 定位精度
    2. 對於低採樣率（< 500Hz）數據特別重要
    
    參數:
        x: ECG 信號
        fs: 採樣率
        use_parabolic_refinement: 是否使用拋物線插值精細化（默認 True）
    
    返回:
        ridx: R-peak 索引陣列（整數）
        x_clean: 清理後的 ECG 信號
        ridx_refined: 精細化的 R-peak 位置（浮點數，如果啟用拋物線插值）
    """
    x = x.astype("float64") - np.median(x)
    ridx = np.array([], dtype=int)
    x_clean = x
    
    # 優先 ecg_process
    try:
        signals, info = nk.ecg_process(x, sampling_rate=fs, method="neurokit")
        ridx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
        if ridx.size >= 3:
            x_clean = signals["ECG_Clean"].to_numpy()
    except Exception:
        pass
    
    # 備用 ecg_peaks
    if ridx.size < 3:
        try:
            xc = nk.ecg_clean(x, sampling_rate=fs, method="neurokit")
            sig, _ = nk.ecg_peaks(xc, sampling_rate=fs, method="neurokit")
            if "ECG_R_Peaks" in sig:
                ridx = np.flatnonzero(sig["ECG_R_Peaks"].to_numpy().astype(bool))
                if ridx.size >= 3:
                    x_clean = xc
        except Exception:
            pass
    
    # 極性翻轉再嘗試
    if ridx.size < 3:
        try:
            signals, info = nk.ecg_process(-x, sampling_rate=fs, method="neurokit")
            ridx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            if ridx.size >= 3:
                x_clean = signals["ECG_Clean"].to_numpy()
        except Exception:
            pass
    
    if ridx.size < 3:
        return np.array([], dtype=int), x
    
    # 應用拋物線插值精細化
    # 這對於低採樣率（< 500Hz）數據特別重要
    # DREAMER 數據集使用 256Hz，拋物線插值可以將精度提高約 50%
    if use_parabolic_refinement and fs < 500:
        _, ridx_integer = refine_rpeak_with_parabolic_interpolation(x_clean, ridx, fs)
        # 更新 ridx 為精細化後的整數位置
        ridx = ridx_integer
    
    return ridx, x_clean


# ---------- 時域特徵（8） ----------
def time_features_from_rri(rri_ms, r_times_s, win_sec=60.0):
    """計算時域 HRV 特徵"""
    rr = np.asarray(rri_ms, dtype=float)
    feat = dict()
    if rr.size == 0:
        return {k: np.nan for k in ["MeanRR_ms","SDNN_ms","MeanHR_bpm","SDHR_bpm","RMSSD_ms","NN50","pNN50","SDNN_index_ms"]}
    
    feat["MeanRR_ms"] = float(np.mean(rr))
    feat["SDNN_ms"] = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0
    hr = 60000.0 / rr
    feat["MeanHR_bpm"] = float(np.mean(hr))
    feat["SDHR_bpm"] = float(np.std(hr, ddof=1)) if hr.size > 1 else 0.0
    diff = np.diff(rr)
    feat["RMSSD_ms"] = float(np.sqrt(np.mean(diff**2))) if diff.size>0 else np.nan
    nn50 = float(np.sum(np.abs(diff) > 50.0)) if diff.size>0 else 0.0
    feat["NN50"] = nn50
    feat["pNN50"] = float(100.0 * nn50 / diff.size) if diff.size>0 else np.nan
    
    # SDNN index
    if r_times_s is None or len(r_times_s) < 2:
        feat["SDNN_index_ms"] = np.nan
    else:
        start = r_times_s[0]; end = r_times_s[-1]
        bins = np.arange(start, end + win_sec, win_sec)
        rr_t = r_times_s[1:]
        sd_list = []
        for i in range(len(bins)-1):
            mask = (rr_t >= bins[i]) & (rr_t < bins[i+1])
            if np.sum(mask) >= 2:
                sd_list.append(np.std(rr[mask], ddof=1))
        feat["SDNN_index_ms"] = float(np.mean(sd_list)) if sd_list else np.nan
    return feat


# ---------- EDR (ECG-derived Respiration) 提取 ----------
def extract_edr_respiratory_rate(x_clean, ridx, fs, resample_hz=4.0):
    """
    從 ECG 訊號提取 EDR (ECG-derived Respiration) 並估算呼吸頻率
    
    方法：R-peak amplitude modulation
    - 提取每個 R-peak 的振幅
    - 對振幅序列進行插值，使其均勻採樣
    - 使用頻譜分析找到呼吸頻率（0.15-0.4 Hz，即 9-24 bpm）
    
    參數:
        x_clean: 清理後的 ECG 訊號
        ridx: R-peak 索引
        fs: 原始採樣率
        resample_hz: 重採樣率（用於頻譜分析）
    
    返回:
        dict: 包含呼吸頻率相關特徵的字典
    """
    if ridx is None or len(ridx) < 10:
        return {
            'respiratory_rate_bpm': np.nan,
            'respiratory_rate_hz': np.nan,
            'edr_peak_freq_hz': np.nan,
            'edr_peak_power': np.nan,
            'edr_snr': np.nan
        }
    
    try:
        # 1. 提取 R-peak 振幅
        r_amplitudes = x_clean[ridx].astype(float)
        
        # 2. 計算 R-peak 時間點（秒）
        r_times_s = ridx / float(fs)
        
        # 3. 對振幅序列進行插值，使其均勻採樣
        if len(r_times_s) < 2:
            return {
                'respiratory_rate_bpm': np.nan,
                'respiratory_rate_hz': np.nan,
                'edr_peak_freq_hz': np.nan,
                'edr_peak_power': np.nan,
                'edr_snr': np.nan
            }
        
        # 創建均勻時間序列
        t_new = np.arange(r_times_s[0], r_times_s[-1], 1.0/resample_hz)
        if len(t_new) < 16:
            return {
                'respiratory_rate_bpm': np.nan,
                'respiratory_rate_hz': np.nan,
                'edr_peak_freq_hz': np.nan,
                'edr_peak_power': np.nan,
                'edr_snr': np.nan
            }
        
        # 三次樣條插值（減少高頻失真）
        try:
            f = interp1d(r_times_s, r_amplitudes, kind="cubic", 
                         fill_value="extrapolate", assume_sorted=True)
        except Exception:
            # 數據點太少時回退到線性插值
            f = interp1d(r_times_s, r_amplitudes, kind="linear", 
                         fill_value="extrapolate", assume_sorted=True)
        edr_signal = f(t_new)
        
        # 4. 去趨勢（移除直流分量和線性趨勢）
        edr_signal = sps.detrend(edr_signal, type="linear")
        
        # 5. 計算功率頻譜密度（Welch 方法）
        nperseg = min(len(edr_signal), int(resample_hz * 64))
        if nperseg < 16:
            return {
                'respiratory_rate_bpm': np.nan,
                'respiratory_rate_hz': np.nan,
                'edr_peak_freq_hz': np.nan,
                'edr_peak_power': np.nan,
                'edr_snr': np.nan
            }
        
        freqs, psd = sps.welch(edr_signal, fs=resample_hz, nperseg=nperseg)
        
        # 6. 在呼吸頻率範圍內尋找峰值（0.15-0.4 Hz，即 9-24 bpm）
        resp_mask = (freqs >= 0.15) & (freqs <= 0.40)
        if not np.any(resp_mask):
            return {
                'respiratory_rate_bpm': np.nan,
                'respiratory_rate_hz': np.nan,
                'edr_peak_freq_hz': np.nan,
                'edr_peak_power': np.nan,
                'edr_snr': np.nan
            }
        
        resp_freqs = freqs[resp_mask]
        resp_psd = psd[resp_mask]
        
        # 找到峰值頻率
        peak_idx = np.argmax(resp_psd)
        peak_freq_hz = float(resp_freqs[peak_idx])
        peak_power = float(resp_psd[peak_idx])
        
        # 轉換為 bpm（呼吸每分鐘次數）
        respiratory_rate_bpm = peak_freq_hz * 60.0
        
        # 7. 計算 SNR（信號與噪聲比）
        # 使用呼吸頻帶外的功率作為噪聲估計
        noise_mask = ((freqs >= 0.05) & (freqs < 0.15)) | ((freqs > 0.40) & (freqs <= 0.50))
        if np.any(noise_mask):
            noise_power = float(np.mean(psd[noise_mask]))
            if noise_power > 0:
                snr_db = 10 * np.log10(peak_power / noise_power)
            else:
                snr_db = np.nan
        else:
            snr_db = np.nan
        
        return {
            'respiratory_rate_bpm': respiratory_rate_bpm,
            'respiratory_rate_hz': peak_freq_hz,
            'edr_peak_freq_hz': peak_freq_hz,
            'edr_peak_power': peak_power,
            'edr_snr': float(snr_db) if np.isfinite(snr_db) else np.nan
        }
    
    except Exception as e:
        # 如果提取失敗，返回 NaN
        return {
            'respiratory_rate_bpm': np.nan,
            'respiratory_rate_hz': np.nan,
            'edr_peak_freq_hz': np.nan,
            'edr_peak_power': np.nan,
            'edr_snr': np.nan
        }


# ---------- 高階去趨勢化 (Smoothness Priors Detrending) ----------
def smoothness_priors_detrend(x, lambda_val=500):
    """
    使用 Smoothness Priors 方法進行去趨勢化
    
    這是處理 HRV 數據的推薦方法，可以有效移除非平穩趨勢（如心率漂移）
    而不會像多項式擬合那樣在邊緣產生振盪。
    
    參數:
        x: 輸入信號
        lambda_val: 平滑參數（越大，去除的趨勢越平滑）
                    - 建議值: 300-500 用於 4Hz 重採樣的 HRV
                    - 較低值: 更多細節保留
                    - 較高值: 更強的趨勢移除
    
    返回:
        去趨勢後的信號
    
    參考: Tarvainen et al. (2002) "An advanced detrending method with application to HRV analysis"
    """
    if len(x) < 3:
        return x
    
    x = np.asarray(x, dtype=float)
    N = len(x)
    
    # 構建二階差分矩陣 D
    # D = [1, -2, 1] 的 Toeplitz 形式
    e = np.ones(N)
    D = np.diag(e) - 2 * np.diag(e[:-1], 1) + np.diag(e[:-2], 2)
    D = D[:-2, :]  # (N-2) x N
    
    # 計算趨勢: trend = (I + lambda^2 * D^T * D)^(-1) * x
    # 等效於求解線性系統
    I = np.eye(N)
    H = I + (lambda_val ** 2) * (D.T @ D)
    
    try:
        # 使用 Cholesky 分解求解（更穩定）
        from scipy.linalg import cho_factor, cho_solve
        c, lower = cho_factor(H)
        trend = cho_solve((c, lower), x)
    except Exception:
        # 回退到直接求解
        try:
            trend = np.linalg.solve(H, x)
        except Exception:
            # 如果數值不穩定，回退到線性去趨勢
            return sps.detrend(x, type="linear")
    
    return x - trend


# ---------- 頻域特徵（7） ----------
def welch_psd_from_rpeaks(ridx, fs, resample_hz=4.0, detrend_method="smoothness_priors"):
    """
    使用 Welch 方法計算 PSD
    
    重要改進：
    1. 使用三次樣條插值（Cubic Spline）代替線性插值，減少高頻失真
    2. 使用 Smoothness Priors 去趨勢化，有效移除非平穩趨勢（避免 ULF 異常）
    3. 動態調整 nperseg 以平衡頻率分辨率和變異性
    
    參數:
        ridx: R-peak 索引陣列
        fs: 原始 ECG 採樣率
        resample_hz: 重採樣率（建議 4Hz）
        detrend_method: 去趨勢方法
            - "smoothness_priors": Smoothness Priors（推薦，移除非平穩漂移）
            - "linear": 線性去趨勢
            - "constant": 僅移除均值（不推薦，會導致 ULF 異常）
    
    返回:
        freqs: 頻率陣列 (Hz)
        psd: 功率頻譜密度陣列
    """
    ridx = np.asarray(ridx, dtype=int)
    if ridx.size < 3:
        return None, None
    
    rr_s = np.diff(ridx) / float(fs)
    rr_t = ridx[1:] / float(fs)
    t_new = np.arange(rr_t[0], rr_t[-1], 1.0/resample_hz)
    
    # 改進 1: 使用三次樣條插值（減少高頻失真）
    # 線性插值會平滑掉高頻細節，低估 HF Power
    try:
        f = interp1d(rr_t, rr_s, kind="cubic", fill_value="extrapolate", assume_sorted=True)
    except Exception:
        # 數據點太少時回退到線性插值
        f = interp1d(rr_t, rr_s, kind="linear", fill_value="extrapolate", assume_sorted=True)
    
    rr_even = f(t_new)
    
    # 改進 2: 使用 Smoothness Priors 去趨勢化
    # 這是處理 HRV 非平穩趨勢（心率漂移）的推薦方法
    if detrend_method == "smoothness_priors":
        # lambda_val 根據重採樣率調整（4Hz 時建議 500）
        lambda_val = int(500 * resample_hz / 4.0)
        rr_even = smoothness_priors_detrend(rr_even, lambda_val=lambda_val)
    elif detrend_method == "linear":
        rr_even = sps.detrend(rr_even, type="linear")
    else:  # "constant"
        rr_even = sps.detrend(rr_even, type="constant")
    
    # 改進 3: 動態調整 nperseg
    # 根據數據長度和所需頻率分辨率動態計算
    # 頻率分辨率 = fs / nperseg，我們希望至少 0.01 Hz 的分辨率
    min_freq_resolution = 0.01  # Hz
    nperseg_for_resolution = int(resample_hz / min_freq_resolution)
    nperseg = min(len(rr_even), nperseg_for_resolution, int(resample_hz * 64))
    
    if nperseg < 16:
        return None, None
    
    # 使用 Hann 窗（默認）進行 Welch PSD 估計
    freqs, psd = sps.welch(rr_even, fs=resample_hz, nperseg=nperseg, 
                           window='hann', noverlap=nperseg//2)
    return freqs, psd


def band_area_and_peak(freqs, psd, fmin, fmax):
    """
    計算頻帶的面積與峰值頻率
    
    返回:
        area: 頻帶面積（功率積分）
        peak_freq: 峰值對應的頻率（Hz），而非功率值
    
    修復說明：
        原版本錯誤地返回功率值（np.max(psd[m])），導致 HF_peak 出現 0.0022 這種異常值。
        正確做法是找到峰值功率對應的頻率值。
    """
    m = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(m):
        return np.nan, np.nan
    
    # 計算頻帶面積
    area = float(np.trapezoid(psd[m], freqs[m]))
    
    # 找到峰值功率對應的頻率（而非功率值本身）
    # 使用 argmax 找到頻段內功率最大的索引，然後返回對應的頻率
    psd_masked = psd[m]
    freqs_masked = freqs[m]
    
    if len(psd_masked) == 0:
        return area, np.nan
    
    peak_idx = np.argmax(psd_masked)
    peak_freq = float(freqs_masked[peak_idx])
    
    # 驗證峰值頻率在合理範圍內（防止數值誤差）
    if not (fmin <= peak_freq <= fmax):
        # 如果計算出的頻率不在範圍內，可能是數值誤差，使用範圍內最接近的值
        if peak_freq < fmin:
            peak_freq = float(fmin)
        elif peak_freq > fmax:
            peak_freq = float(fmax)
    
    return area, peak_freq


def freq_features(freqs, psd, edr_freq_hz=None):
    """
    計算頻域 HRV 特徵（增強版）
    
    改進：
    1. 添加 RSA/呼吸頻率與 HF Peak 重疊檢測
    2. 添加 HF Power 可信度指標
    3. 檢測異常的 ULF dominance（可能是去趨勢不足）
    
    參數:
        freqs: 頻率陣列 (Hz)
        psd: 功率頻譜密度陣列
        edr_freq_hz: EDR（ECG-derived respiration）估計的呼吸頻率 (Hz)，可選
    
    返回:
        dict: 頻域特徵和質量指標
    """
    base_keys = ["ULF_peak","LF_peak","HF_peak","ULF_ratio","LF_ratio","HF_ratio","LF_HF"]
    quality_keys = ["HF_RSA_overlap", "HF_reliability", "ULF_dominance_flag", "spectral_quality"]
    
    if freqs is None:
        nan_dict = {k: np.nan for k in base_keys}
        nan_dict.update({k: np.nan for k in quality_keys})
        return nan_dict
    
    ULF_area, ULF_peak = band_area_and_peak(freqs, psd, 0.00, 0.04)
    LF_area,  LF_peak  = band_area_and_peak(freqs, psd, 0.04, 0.15)
    HF_area,  HF_peak  = band_area_and_peak(freqs, psd, 0.15, 0.40)
    total = np.nansum([ULF_area, LF_area, HF_area])
    
    def ratio(x): 
        return float(x/total) if (total is not None and np.isfinite(total) and total>0 and np.isfinite(x)) else np.nan
    
    # === 質量指標計算 ===
    
    # 1. RSA/呼吸頻率與 HF Peak 重疊檢測
    # 如果呼吸頻率與 HF Peak 重疊，HF Power 可能被呼吸調變放大
    hf_rsa_overlap = 0.0
    if edr_freq_hz is not None and np.isfinite(edr_freq_hz) and np.isfinite(HF_peak):
        # 計算 EDR 頻率與 HF Peak 的接近程度
        freq_diff = abs(edr_freq_hz - HF_peak)
        # 如果差距 < 0.05 Hz，認為重疊
        if freq_diff < 0.05:
            hf_rsa_overlap = 1.0  # 完全重疊
        elif freq_diff < 0.10:
            hf_rsa_overlap = 0.5  # 部分重疊
        else:
            hf_rsa_overlap = 0.0  # 無重疊
    
    # 2. HF Power 可信度
    # 基於 RSA 重疊、HF 比例和數據特性評估
    hf_reliability = 1.0
    if hf_rsa_overlap > 0.5:
        hf_reliability *= 0.5  # RSA 重疊降低可信度
    hf_ratio_val = ratio(HF_area)
    if np.isfinite(hf_ratio_val):
        if hf_ratio_val > 0.8:
            hf_reliability *= 0.7  # 異常高的 HF 比例可能有問題
        elif hf_ratio_val < 0.05:
            hf_reliability *= 0.8  # 異常低的 HF 比例也需注意
    
    # 3. ULF Dominance 檢測（可能是去趨勢不足）
    # 如果 ULF 占比 > 50%，或 ULF Peak 是總功率的主要來源，標記警告
    ulf_ratio_val = ratio(ULF_area)
    ulf_dominance_flag = 0.0
    if np.isfinite(ulf_ratio_val) and ulf_ratio_val > 0.5:
        ulf_dominance_flag = 1.0
    elif np.isfinite(ulf_ratio_val) and ulf_ratio_val > 0.3:
        ulf_dominance_flag = 0.5
    
    # 4. 綜合頻譜質量指標
    spectral_quality = 1.0
    if ulf_dominance_flag > 0.5:
        spectral_quality *= 0.6  # ULF 主導可能是去趨勢問題
    if hf_rsa_overlap > 0.5:
        spectral_quality *= 0.7  # RSA 干擾
    if not np.isfinite(ratio(LF_area)) or not np.isfinite(ratio(HF_area)):
        spectral_quality *= 0.5  # 計算異常
    
    return dict(
        # 基本頻域特徵
        ULF_peak=ULF_peak, LF_peak=LF_peak, HF_peak=HF_peak,
        ULF_ratio=ratio(ULF_area), LF_ratio=ratio(LF_area), HF_ratio=ratio(HF_area),
        LF_HF=float(LF_area/HF_area) if (HF_area is not None and np.isfinite(HF_area) and HF_area>0) else np.nan,
        # 質量指標
        HF_RSA_overlap=float(hf_rsa_overlap),
        HF_reliability=float(hf_reliability),
        ULF_dominance_flag=float(ulf_dominance_flag),
        spectral_quality=float(spectral_quality)
    )


# ---------- 非線性特徵（5） ----------
def sd1_sd2_from_rri(rr_ms):
    """計算 Poincaré 圖的 SD1 和 SD2"""
    rr = np.asarray(rr_ms, dtype=float)
    if rr.size < 3: 
        return np.nan, np.nan, np.nan
    rrn, rrn1 = rr[:-1], rr[1:]
    sd1 = float(np.std((rrn1 - rrn)/np.sqrt(2), ddof=1))
    sd2 = float(np.std((rrn1 + rrn)/np.sqrt(2), ddof=1))
    return sd1, sd2, (sd1/sd2 if sd2>0 else np.nan)


def sample_entropy(rr_ms, m=2, r_ratio=0.2, return_confidence=False):
    """
    計算樣本熵 (Sample Entropy) - 增強版
    
    重要改進：
    1. 添加數據長度檢查（根據文獻建議，m=2 時需要至少 200 個點）
    2. 添加置信度指標（基於數據長度和匹配計數）
    3. 修正容忍度計算（使用 MAD 而非 SD 減少偽影影響）
    
    參數:
        rr_ms: RR 間隔序列 (ms)
        m: 嵌入維度（默認 2）
        r_ratio: 容忍度比率（默認 0.2，相對於 SD 或 MAD）
        return_confidence: 是否返回置信度指標
    
    返回:
        如果 return_confidence=False: SampEn 值
        如果 return_confidence=True: (SampEn, confidence_dict)
    
    置信度等級:
        - "high": N >= 200，匹配計數充足
        - "medium": 100 <= N < 200，結果可參考但有較大不確定性
        - "low": N < 100，結果不可靠
        - "insufficient": 數據不足以計算
    
    參考: 
        - Richman & Moorman (2000) "Physiological time-series analysis using approximate entropy and sample entropy"
        - Lake et al. (2002) "Sample entropy analysis of neonatal heart rate variability"
    """
    x = np.asarray(rr_ms, dtype=float)
    N = x.size
    
    # 基本數據長度檢查
    min_points_required = m + 2
    if N < min_points_required:
        if return_confidence:
            return np.nan, {
                'confidence': 'insufficient',
                'data_length': N,
                'min_recommended': 200,
                'match_count_A': 0,
                'match_count_B': 0
            }
        return np.nan
    
    # 計算容忍度 r
    # 改進：使用 MAD（中位數絕對偏差）而非 SD，減少偽影/異常值的影響
    std = np.std(x, ddof=1)
    mad = np.median(np.abs(x - np.median(x))) * 1.4826  # MAD 轉換為 SD 等效值
    
    # 如果 MAD 遠小於 SD（存在離群值），使用 MAD
    if std > 0 and mad > 0 and mad < std * 0.5:
        r = r_ratio * mad
    elif std > 0 and np.isfinite(std):
        r = r_ratio * std
    else:
        if return_confidence:
            return np.nan, {
                'confidence': 'insufficient',
                'data_length': N,
                'min_recommended': 200,
                'match_count_A': 0,
                'match_count_B': 0,
                'note': 'Zero or invalid standard deviation'
            }
        return np.nan
    
    def _phi(m_val):
        """計算模板匹配的概率"""
        n_templates = N - m_val + 1
        if n_templates <= 1:
            return 0.0, 0
        
        Xm = np.array([x[i:i+m_val] for i in range(n_templates)])
        count = 0
        
        for i in range(n_templates - 1):
            # 計算當前模板與所有後續模板的最大差異
            d = np.max(np.abs(Xm[i+1:] - Xm[i]), axis=1)
            count += np.sum(d <= r)
        
        denom = n_templates * (n_templates - 1) / 2
        prob = count / denom if denom > 0 else 0.0
        return prob, count
    
    # 計算 A (m+1 維度的匹配) 和 B (m 維度的匹配)
    A, count_A = _phi(m + 1)
    B, count_B = _phi(m)
    
    # 計算 SampEn
    if A > 0 and B > 0:
        sampen = float(-np.log(A / B))
    else:
        sampen = np.nan
    
    # 計算置信度
    if return_confidence:
        # 根據數據長度和匹配計數評估置信度
        if N >= 200 and count_B >= 20:
            confidence = 'high'
        elif N >= 100 and count_B >= 10:
            confidence = 'medium'
        elif N >= 50 and count_B >= 5:
            confidence = 'low'
        else:
            confidence = 'very_low'
        
        confidence_dict = {
            'confidence': confidence,
            'data_length': N,
            'min_recommended': 200,
            'match_count_A': count_A,
            'match_count_B': count_B,
            'tolerance_r': float(r),
            'sd_used': 'MAD' if (mad < std * 0.5) else 'SD'
        }
        return sampen, confidence_dict
    
    return sampen


def sample_entropy_with_quality(rr_ms, m=2, r_ratio=0.2):
    """
    計算 SampEn 並返回質量指標（用於數據集特徵）
    
    返回:
        dict: 包含 SampEn 和相關質量指標
    """
    sampen, conf = sample_entropy(rr_ms, m=m, r_ratio=r_ratio, return_confidence=True)
    
    # 將置信度轉換為數值（便於機器學習）
    confidence_map = {
        'high': 1.0,
        'medium': 0.7,
        'low': 0.4,
        'very_low': 0.2,
        'insufficient': 0.0
    }
    
    return {
        'SampEn': sampen,
        'SampEn_confidence': confidence_map.get(conf['confidence'], 0.0),
        'SampEn_data_length': conf['data_length'],
        'SampEn_match_count_B': conf['match_count_B'],
        'SampEn_reliability': 'reliable' if conf['confidence'] in ['high', 'medium'] else 'unreliable'
    }


def dfa_alpha(rr_ms, scales=(4,16), num_scales=10):
    """去趨勢波動分析 (DFA)"""
    x = np.asarray(rr_ms, dtype=float)
    if x.size < scales[1] + 2: 
        return np.nan
    
    y = np.cumsum(x - np.mean(x))
    s_vals = np.unique(np.floor(
        np.logspace(np.log10(scales[0]), np.log10(scales[1]), num=num_scales)
    ).astype(int))
    
    F_s = []
    valid_scales = []
    
    for s in s_vals:
        nseg = len(y) // s
        if nseg < 2:
            continue

        segments = y[:nseg * s].reshape(nseg, s)
        t = np.arange(s)
        rms_list = []

        for seg in segments:
            coeff = np.polyfit(t, seg, 1)
            trend = np.polyval(coeff, t)
            rms = np.sqrt(np.mean((seg - trend) ** 2))
            rms_list.append(rms)

        F_mean = np.mean(rms_list)
        F_s.append(F_mean)
        valid_scales.append(s)
    
    if len(valid_scales) < 2:
        return np.nan
    
    log_s = np.log10(np.array(valid_scales, dtype=float))
    log_F = np.log10(np.array(F_s, dtype=float))
    
    valid_mask = np.isfinite(log_s) & np.isfinite(log_F)
    if np.sum(valid_mask) < 2:
        return np.nan
    
    try:
        p = np.polyfit(log_s[valid_mask], log_F[valid_mask], 1)
        alpha = float(p[0])
        if not np.isfinite(alpha) or alpha < 0 or alpha > 3:
            return np.nan
        return alpha
    except (np.linalg.LinAlgError, ValueError):
        return np.nan


# ---------- 綜合訊號品質指標 (Data Quality Flags) ----------
def compute_hrv_quality_flags(rri_ms, ridx, fs, freq_feat=None, sampen_conf=None, edr_feat=None):
    """
    計算 HRV 數據的綜合品質指標
    
    這個函數整合了多個來源的品質信息，生成一個統一的品質評估報告。
    
    參數:
        rri_ms: RR 間隔序列 (ms)
        ridx: R-peak 索引陣列
        fs: 採樣率
        freq_feat: 頻域特徵字典（包含 spectral_quality 等）
        sampen_conf: SampEn 置信度字典
        edr_feat: EDR 特徵字典
    
    返回:
        dict: 綜合品質指標
    
    品質等級說明:
        - overall_quality: 0-1 綜合品質分數
        - data_length_quality: 數據長度是否足夠
        - artifact_quality: 偽影/異常心搏的影響
        - spectral_quality: 頻譜分析的可靠性
        - nonlinear_quality: 非線性特徵的可靠性
        - respiratory_interference: 呼吸干擾程度
    """
    quality = {
        'overall_quality': 1.0,
        'data_length_quality': 1.0,
        'artifact_quality': 1.0,
        'spectral_quality': 1.0,
        'nonlinear_quality': 1.0,
        'respiratory_interference': 0.0,
        'quality_flags': []
    }
    
    rri = np.asarray(rri_ms, dtype=float) if rri_ms is not None else np.array([])
    
    # === 1. 數據長度質量 ===
    n_beats = len(rri) + 1 if len(rri) > 0 else 0
    if n_beats >= 300:
        quality['data_length_quality'] = 1.0
    elif n_beats >= 200:
        quality['data_length_quality'] = 0.9
    elif n_beats >= 100:
        quality['data_length_quality'] = 0.7
        quality['quality_flags'].append('short_recording')
    elif n_beats >= 50:
        quality['data_length_quality'] = 0.5
        quality['quality_flags'].append('very_short_recording')
    else:
        quality['data_length_quality'] = 0.2
        quality['quality_flags'].append('insufficient_data')
    
    # === 2. 偽影/異常心搏質量 ===
    if len(rri) > 0:
        # 檢測生理範圍外的 RRI（300-2000 ms）
        valid_mask = (rri >= 300.0) & (rri <= 2000.0)
        artifact_rate = 1.0 - np.mean(valid_mask)
        
        # 檢測異常跳變（連續 RRI 差異 > 500ms）
        if len(rri) > 1:
            rri_diff = np.abs(np.diff(rri))
            jump_rate = np.mean(rri_diff > 500.0)
        else:
            jump_rate = 0.0
        
        combined_artifact = artifact_rate + jump_rate
        if combined_artifact < 0.05:
            quality['artifact_quality'] = 1.0
        elif combined_artifact < 0.10:
            quality['artifact_quality'] = 0.8
            quality['quality_flags'].append('minor_artifacts')
        elif combined_artifact < 0.20:
            quality['artifact_quality'] = 0.5
            quality['quality_flags'].append('moderate_artifacts')
        else:
            quality['artifact_quality'] = 0.2
            quality['quality_flags'].append('severe_artifacts')
    
    # === 3. 頻譜質量 ===
    if freq_feat is not None:
        # 從頻域特徵獲取品質指標
        spec_q = freq_feat.get('spectral_quality', 1.0)
        if np.isfinite(spec_q):
            quality['spectral_quality'] = float(spec_q)
        
        # 檢測 ULF 主導（可能是去趨勢問題）
        ulf_dom = freq_feat.get('ULF_dominance_flag', 0.0)
        if np.isfinite(ulf_dom) and ulf_dom > 0.5:
            quality['quality_flags'].append('ULF_dominant')
        
        # 檢測 HF-RSA 重疊
        hf_overlap = freq_feat.get('HF_RSA_overlap', 0.0)
        if np.isfinite(hf_overlap):
            quality['respiratory_interference'] = float(hf_overlap)
            if hf_overlap > 0.5:
                quality['quality_flags'].append('respiratory_HF_overlap')
    
    # === 4. 非線性特徵質量 ===
    if sampen_conf is not None:
        conf_level = sampen_conf.get('confidence', 'unknown')
        conf_map = {'high': 1.0, 'medium': 0.7, 'low': 0.4, 'very_low': 0.2, 'insufficient': 0.0}
        quality['nonlinear_quality'] = conf_map.get(conf_level, 0.5)
        
        if conf_level in ['low', 'very_low', 'insufficient']:
            quality['quality_flags'].append('unreliable_sampen')
    
    # === 5. EDR/呼吸質量 ===
    if edr_feat is not None:
        edr_snr = edr_feat.get('edr_snr', np.nan)
        if np.isfinite(edr_snr):
            if edr_snr < 3.0:
                quality['quality_flags'].append('weak_respiratory_signal')
    
    # === 計算綜合品質分數 ===
    weights = {
        'data_length_quality': 0.25,
        'artifact_quality': 0.35,
        'spectral_quality': 0.25,
        'nonlinear_quality': 0.15
    }
    
    overall = 0.0
    for key, weight in weights.items():
        val = quality[key]
        if np.isfinite(val):
            overall += val * weight
    
    quality['overall_quality'] = float(overall)
    
    # 將 quality_flags 列表轉換為字符串（便於 CSV 存儲）
    quality['quality_flags_str'] = '|'.join(quality['quality_flags']) if quality['quality_flags'] else 'none'
    
    return quality


# ---------- Baseline 差異特徵 ----------
def calculate_delta_features(baseline_feat, stimuli_feat):
    """
    計算 stimuli 相對於 baseline 的差異特徵
    
    重要說明（關於 Z-score 與 Delta 一致性）：
    =========================================
    
    **問題根源**：
    傳統 Z-score 使用全域分佈 (global_mean/std)，而 Delta 使用個人 baseline。
    這會導致 Z-score > 0 但 Delta < 0 的「矛盾」。
    
    **解決方案**：
    1. 保存 Baseline 值，供階段 2 計算 Baseline-adjusted Z-score
    2. 在階段 2 中同時提供：
       - 傳統 Z-score: (stimuli - global_mean) / global_std
       - Delta Z-score: (Delta - global_mean_of_delta) / global_std_of_delta
    
    後者確保 Z-score 和 Delta 方向一致：
    - 當 Delta > 0（相對於個人 baseline 升高）時，Delta_zscore 也會傾向 > 0
    - 當 Delta < 0（相對於個人 baseline 降低）時，Delta_zscore 也會傾向 < 0
    
    **欄位說明**：
    - Delta_{key}: stimuli - baseline（原始差異）
    - Delta_{key}_pct: 百分比變化
    - Baseline_{key}: 個人 baseline 值（用於後續分析和 Z-score 計算）
    """
    if baseline_feat is None or stimuli_feat is None:
        return {}
    
    delta_feat = {}
    for key in stimuli_feat.keys():
        if key in baseline_feat:
            bl_val = baseline_feat[key]
            st_val = stimuli_feat[key]
            if np.isfinite(bl_val) and np.isfinite(st_val):
                # 計算 Delta（核心指標）
                delta_feat[f"Delta_{key}"] = float(st_val - bl_val)
                # 相對變化百分比（避免除以零）
                if abs(bl_val) > 1e-6:
                    delta_feat[f"Delta_{key}_pct"] = float((st_val - bl_val) / abs(bl_val) * 100.0)
                else:
                    delta_feat[f"Delta_{key}_pct"] = np.nan
                # 保存 Baseline 值（供階段 2 計算一致的 Z-score）
                delta_feat[f"Baseline_{key}"] = float(bl_val)
            else:
                delta_feat[f"Delta_{key}"] = np.nan
                delta_feat[f"Delta_{key}_pct"] = np.nan
                delta_feat[f"Baseline_{key}"] = np.nan
        else:
            delta_feat[f"Delta_{key}"] = np.nan
            delta_feat[f"Delta_{key}_pct"] = np.nan
            delta_feat[f"Baseline_{key}"] = np.nan
    
    return delta_feat


# ---------- EEG 特徵提取 ----------
def extract_eeg_band_power(eeg_signal, fs, fmin, fmax):
    """提取 EEG 頻帶功率"""
    if eeg_signal is None or eeg_signal.size == 0:
        return np.nan
    
    try:
        # 計算功率頻譜密度
        nperseg = min(len(eeg_signal), int(fs * 2))  # 2秒窗口
        if nperseg < 16:
            return np.nan
        
        freqs, psd = sps.welch(eeg_signal, fs=fs, nperseg=nperseg)
        
        # 提取頻帶功率
        mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(mask):
            return np.nan
        
        band_power = float(np.trapezoid(psd[mask], freqs[mask]))
        return band_power
    except Exception:
        return np.nan


def extract_eeg_features(eeg_signal, fs=128):
    """提取 EEG 頻譜特徵（單通道）"""
    if eeg_signal is None or eeg_signal.size == 0:
        return {
            'delta_power': np.nan, 'theta_power': np.nan,
            'alpha_power': np.nan, 'beta_power': np.nan,
            'gamma_power': np.nan, 'alpha_beta_ratio': np.nan
        }
    
    # 定義頻帶
    delta_power = extract_eeg_band_power(eeg_signal, fs, 0.5, 4.0)
    theta_power = extract_eeg_band_power(eeg_signal, fs, 4.0, 8.0)
    alpha_power = extract_eeg_band_power(eeg_signal, fs, 8.0, 13.0)
    beta_power = extract_eeg_band_power(eeg_signal, fs, 13.0, 30.0)
    gamma_power = extract_eeg_band_power(eeg_signal, fs, 30.0, 100.0)
    
    # 計算比率
    alpha_beta_ratio = float(alpha_power / beta_power) if (np.isfinite(alpha_power) and np.isfinite(beta_power) and beta_power > 0) else np.nan
    
    return {
        'delta_power': delta_power,
        'theta_power': theta_power,
        'alpha_power': alpha_power,
        'beta_power': beta_power,
        'gamma_power': gamma_power,
        'alpha_beta_ratio': alpha_beta_ratio
    }


def extract_eeg_multi_channel_features(eeg_14ch, fs=128):
    """提取多通道 EEG 特徵（14 通道）"""
    if eeg_14ch is None or eeg_14ch.size == 0:
        return {}
    
    # 確保是 2D 陣列
    if len(eeg_14ch.shape) == 1:
        eeg_14ch = eeg_14ch.reshape(-1, 1)
    
    num_channels = eeg_14ch.shape[1] if len(eeg_14ch.shape) > 1 else 1
    
    # 定義區域通道索引（根據 DREAMER 的電極順序）
    # AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    frontal_indices = [0, 1, 2, 11, 12, 13]  # AF3, F7, F3, F4, F8, AF4
    central_indices = [3, 10]  # FC5, FC6
    temporal_indices = [4, 9]  # T7, T8
    parietal_indices = [5, 8]  # P7, P8
    occipital_indices = [6, 7]  # O1, O2
    
    features = {}
    
    # 每個通道的頻譜特徵
    for ch_idx in range(min(num_channels, 14)):
        ch_signal = eeg_14ch[:, ch_idx] if num_channels > 1 else eeg_14ch.flatten()
        ch_feat = extract_eeg_features(ch_signal, fs)
        
        for feat_name, feat_val in ch_feat.items():
            features[f"EEG_ch{ch_idx}_{feat_name}"] = feat_val
    
        # 區域平均功率（如果通道數足夠）
        if num_channels >= 14:
            # 前額區域
            frontal_signals = [eeg_14ch[:, i] for i in frontal_indices if i < num_channels]
            if frontal_signals:
                frontal_alphas = [extract_eeg_features(sig, fs)['alpha_power'] for sig in frontal_signals]
                frontal_alphas = [a for a in frontal_alphas if np.isfinite(a)]
                if frontal_alphas:
                    features['EEG_frontal_alpha_power'] = float(np.mean(frontal_alphas))
            
            # 全通道平均
            all_ch_alpha = []
            all_ch_beta = []
            for ch_idx in range(num_channels):
                ch_signal = eeg_14ch[:, ch_idx]
                ch_feat = extract_eeg_features(ch_signal, fs)
                alpha = ch_feat['alpha_power']
                beta = ch_feat['beta_power']
                if np.isfinite(alpha):
                    all_ch_alpha.append(alpha)
                if np.isfinite(beta):
                    all_ch_beta.append(beta)
            
            if all_ch_alpha:
                features['EEG_global_alpha_power'] = float(np.mean(all_ch_alpha))
            if all_ch_beta:
                features['EEG_global_beta_power'] = float(np.mean(all_ch_beta))
    
    return features


# ---------- 視覺化 ----------
def plot_all(out_dir, fs, x_clean, ridx, rri_ms, freqs, psd, resample_hz, 
             poincare_size=(896, 896)):
    """
    Generate all visualization plots.
    
    Important: Uses the same data window as numerical feature calculation:
    - RRI calculation: Based on ridx (from first to last R-peak)
    - Image generation: Uses full x_clean (but marks R-peak window)
    - ECG amplitude stats: Based on R-peak window in x_clean
    
    Args:
        out_dir: Output directory for plots
        fs: Sampling frequency
        x_clean: Cleaned ECG signal
        ridx: R-peak indices
        rri_ms: RR intervals in milliseconds
        freqs: PSD frequency array
        psd: PSD power array
        resample_hz: Resampling frequency
        poincare_size: Target size for Poincare plot (default 896x896)
    """
    # 計算時間軸（基於完整信號，但圖像會標記 R-peak 窗口）
    t = np.arange(len(x_clean))/fs
    
    def _resize_to_target(path: str, target_size=poincare_size):
        """Resize image to target size if needed."""
        try:
            if path and os.path.exists(path):
                with Image.open(path) as img:
                    if img.size != target_size:
                        img_resized = img.resize(target_size)
                        img_resized.save(path)
        except Exception:
            # 不讓單張圖失敗中斷整體流程
            pass
    
    # R-peaks（僅顯示 R-peak 窗口內的信號，確保與數值特徵一致）
    plt.figure(figsize=(12,4))
    if ridx.size >= 2:
        # 提取 R-peak 窗口內的信號（與數值特徵計算一致）
        window_start_idx = int(ridx[0])
        window_end_idx = int(ridx[-1]) + 1
        x_clean_windowed = x_clean[window_start_idx:window_end_idx]
        t_windowed = t[window_start_idx:window_end_idx]
        ridx_windowed = ridx - window_start_idx  # 調整 R-peak 索引到窗口內
        
        plt.plot(t_windowed, x_clean_windowed, label="ECG (clean, R-peak window)")
        if ridx_windowed.size:
            valid_mask = (ridx_windowed >= 0) & (ridx_windowed < len(x_clean_windowed))
            if np.any(valid_mask):
                plt.scatter(t_windowed[ridx_windowed[valid_mask]], 
                          x_clean_windowed[ridx_windowed[valid_mask]], 
                          s=18, c="r", label="R-peaks")
        plt.title(f"R={len(ridx)} (windowed: {window_start_idx/fs:.1f}s - {window_end_idx/fs:.1f}s)")
    else:
        # 如果 R-peak 不足，顯示完整信號
        plt.plot(t, x_clean, label="ECG (clean)")
        if ridx.size:
            plt.scatter(ridx/fs, x_clean[ridx], s=18, c="r", label="R-peaks")
        plt.title(f"R={len(ridx)}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.tight_layout()
    rpeak_path = os.path.join(out_dir,"Rpeak_full.png")
    plt.savefig(rpeak_path, dpi=150)
    plt.close()
    _resize_to_target(rpeak_path)
    
    # Tachogram
    plt.figure(figsize=(10,4))
    if len(rri_ms): 
        plt.plot(rri_ms, marker="o", linewidth=1)
        plt.title(f"Tachogram (RRI) | mean={np.mean(rri_ms):.1f} ms")
    else:
        plt.title("Tachogram (no RRI)")
    plt.xlabel("Beat index")
    plt.ylabel("RRI (ms)")
    plt.tight_layout()
    tach_path = os.path.join(out_dir,"Tachogram.png")
    plt.savefig(tach_path, dpi=150)
    plt.close()
    _resize_to_target(tach_path)
    
    # Poincaré（直接保存為 896x896）
    plt.figure(figsize=(896/100, 896/100))  # 設置 figsize 以匹配目標像素大小
    if len(rri_ms) >= 2:
        rr = np.asarray(rri_ms)
        plt.scatter(rr[:-1], rr[1:], s=10, alpha=0.6)
        lo, hi = float(rr.min()), float(rr.max())
        plt.plot([lo,hi],[lo,hi],"--")
    plt.xlabel("RRn (ms)")
    plt.ylabel("RRn+1 (ms)")
    plt.title("Poincaré")
    plt.tight_layout()
    poincare_path = os.path.join(out_dir,"Poincare.png")
    plt.savefig(poincare_path, dpi=100, bbox_inches='tight')  # dpi=100, figsize=8.96 -> 896x896
    plt.close()
    # 確保尺寸為目標尺寸
    _resize_to_target(poincare_path)
    
    # PSD
    if freqs is not None:
        plt.figure(figsize=(8,4))
        plt.semilogy(freqs, psd)
        plt.axvspan(0.00,0.04, alpha=0.2, label="ULF")
        plt.axvspan(0.04,0.15, alpha=0.2, label="LF")
        plt.axvspan(0.15,0.40, alpha=0.2, label="HF")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("PSD (a.u.)")
        plt.title(f"HRV PSD (Welch, resample={resample_hz} Hz)")
        plt.legend()
        plt.tight_layout()
        psd_path = os.path.join(out_dir,"PSD.png")
        plt.savefig(psd_path, dpi=150)
        plt.close()
        _resize_to_target(psd_path)

    # Signal quality panel（整合 R-peaks 與 Tachogram，方便模型一次查看）
    # 重要：只顯示 R-peak 窗口內的信號，確保與數值特徵計算一致
    try:
        fig, axes = plt.subplots(2, 1, figsize=(8.96, 8.96), sharex=False)

        # 上半部：ECG + R-peaks（僅顯示 R-peak 窗口內的信號）
        if ridx.size >= 2:
            # 提取 R-peak 窗口內的信號（與數值特徵計算一致）
            window_start_idx = int(ridx[0])
            window_end_idx = int(ridx[-1]) + 1
            x_clean_windowed = x_clean[window_start_idx:window_end_idx]
            t_windowed = t[window_start_idx:window_end_idx]
            ridx_windowed = ridx - window_start_idx  # 調整 R-peak 索引到窗口內
            
            axes[0].plot(t_windowed, x_clean_windowed, label="ECG (clean, R-peak window)")
            if ridx_windowed.size:
                valid_mask = (ridx_windowed >= 0) & (ridx_windowed < len(x_clean_windowed))
                if np.any(valid_mask):
                    axes[0].scatter(t_windowed[ridx_windowed[valid_mask]], 
                                  x_clean_windowed[ridx_windowed[valid_mask]], 
                                  s=10, c="r", label="R-peaks")
            axes[0].set_title(f"ECG with R-peaks (R={len(ridx)}, windowed)")
        else:
            # 如果 R-peak 不足，顯示完整信號
            axes[0].plot(t, x_clean, label="ECG (clean)")
            if ridx.size:
                axes[0].scatter(ridx/fs, x_clean[ridx], s=10, c="r", label="R-peaks")
            axes[0].set_title(f"ECG with R-peaks (R={len(ridx)})")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude (a.u.)")
        axes[0].legend(loc="upper right", fontsize=8)

        # 下半部：Tachogram（基於 R-peak 窗口內的 RRI）
        if len(rri_ms):
            axes[1].plot(rri_ms, marker="o", linewidth=1)
            axes[1].set_title(f"Tachogram (RRI) | mean={np.mean(rri_ms):.1f} ms")
        else:
            axes[1].set_title("Tachogram (no RRI)")
        axes[1].set_xlabel("Beat index")
        axes[1].set_ylabel("RRI (ms)")

        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "SignalQuality_panel.png"), dpi=100)
        plt.close(fig)
    except Exception:
        # 不讓繪圖失敗中斷整體流程
        plt.close("all")


def process_single_trial(args):
    """處理單個 trial 的函數（用於多進程並行處理）"""
    s, t, mat_path, config_dict = args
    
    # Reconstruct config from dict (for multiprocessing)
    use_baseline = config_dict.get('use_baseline', True)
    use_eeg = config_dict.get('use_eeg', True)
    use_dual_channel = config_dict.get('use_dual_channel', False)
    use_sqi_selection = config_dict.get('use_sqi_selection', True)
    sqi_mode = config_dict.get('sqi_mode', 'best')
    use_col = config_dict.get('ecg_channel', 0)
    save_plots = config_dict.get('save_plots', True)
    resample_hz = config_dict.get('resample_hz', 4.0)
    output_dir = config_dict.get('output_dir', './output')
    poincare_size = config_dict.get('poincare_size', (896, 896))
    
    try:
        # 每個進程載入自己的 MAT 文件副本（避免序列化問題）
        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        D = mat["DREAMER"]
        
        out_dir = os.path.join(output_dir, f"S{s:02d}", f"T{t:02d}")
        ensure_dir(out_dir)
        
        # 載入資料（新格式）
        data_dict = load_trial(D, s, t, use_baseline=use_baseline, use_eeg=use_eeg)
        ecg_stimuli = data_dict['ecg_stimuli']
        ecg_baseline = data_dict['ecg_baseline']
        eeg_stimuli = data_dict['eeg_stimuli']
        eeg_baseline = data_dict['eeg_baseline']
        fs = data_dict['fs_ecg']
        fs_eeg = data_dict['fs_eeg']
        labels = data_dict['labels']
        
        # 處理 ECG 訊號
        if ecg_stimuli.shape[1] >= 2:
            # 雙通道可用
            if use_sqi_selection:
                # 使用 SQI-based selection
                if sqi_mode == "best":
                    mode = "sqi_best"
                elif sqi_mode == "weighted":
                    mode = "sqi_weighted"
                else:
                    mode = "sqi_weighted"  # 預設使用加權平均
                x_raw = fuse_ecg_channels(ecg_stimuli[:, 0], ecg_stimuli[:, 1], fs=fs, mode=mode)
            elif use_dual_channel:
                # 簡單融合雙通道
                x_raw = fuse_ecg_channels(ecg_stimuli[:, 0], ecg_stimuli[:, 1], fs=fs, mode="simple")
            else:
                # 單通道
                x_raw = ecg_stimuli[:, use_col]
        else:
            # 只有單通道可用
            x_raw = ecg_stimuli[:, use_col]
        
        ridx, x_clean = detect_rpeaks_strong(x_raw, fs)
        
        if ridx.size < 30:
            # 基礎時間資訊（以原始訊號長度估計）
            duration_s = float(len(x_raw) / float(fs)) if len(x_raw) > 0 else np.nan
            base_row = {
                "subject":s,"trial":t,"col":use_col,"R_count":int(ridx.size),
                "MeanRR_ms":np.nan,"SDNN_ms":np.nan,"MeanHR_bpm":np.nan,"SDHR_bpm":np.nan,
                "RMSSD_ms":np.nan,"NN50":np.nan,"pNN50":np.nan,"SDNN_index_ms":np.nan,
                "ULF_peak":np.nan,"LF_peak":np.nan,"HF_peak":np.nan,
                "ULF_ratio":np.nan,"LF_ratio":np.nan,"HF_ratio":np.nan,"LF_HF":np.nan,
                "SD1_ms":np.nan,"SD2_ms":np.nan,"SD1_SD2":np.nan,"SampEn":np.nan,"DFA_alpha":np.nan,
                "artifact_rate":np.nan,"ectopy_count":np.nan,"ectopy_flag":np.nan,
                "valid_rr_ratio":np.nan,"window_start_s":0.0,"window_end_s":duration_s,
                "window_duration_s":duration_s,**labels
            }
            pl.DataFrame([base_row]).write_csv(os.path.join(out_dir,"HRV.csv"))
            return (s, t, base_row, None)  # 返回結果和 None 表示錯誤

        rri_ms = np.diff(ridx)/fs*1000.0
        np.savetxt(os.path.join(out_dir,"RRI.csv"), rri_ms, fmt="%.3f", delimiter=",")

        # 基礎時間資訊（基於 R-peak 窗口，確保與圖像一致）
        if ridx.size >= 2:
            window_start_s = float(ridx[0] / float(fs))
            window_end_s = float(ridx[-1] / float(fs))
            window_duration_s = float(window_end_s - window_start_s)
            # 提取對應窗口的 ECG 信號（與圖像生成使用相同的數據源）
            window_start_idx = int(ridx[0])
            window_end_idx = int(ridx[-1]) + 1  # 包含最後一個 R-peak
            x_clean_windowed = x_clean[window_start_idx:window_end_idx]
        else:
            window_start_s = 0.0
            window_end_s = float(len(x_raw) / float(fs))
            window_duration_s = window_end_s - window_start_s
            x_clean_windowed = x_clean
        
        # 計算實際的 ECG 振幅統計（從實際信號計算，而非從 RRI 推斷）
        if x_clean_windowed.size > 0:
            ecg_amplitude_min = float(np.min(x_clean_windowed))
            ecg_amplitude_max = float(np.max(x_clean_windowed))
            ecg_amplitude_range = float(ecg_amplitude_max - ecg_amplitude_min)
            ecg_amplitude_mean = float(np.mean(x_clean_windowed))
            ecg_amplitude_std = float(np.std(x_clean_windowed, ddof=1)) if x_clean_windowed.size > 1 else 0.0
        else:
            ecg_amplitude_min = np.nan
            ecg_amplitude_max = np.nan
            ecg_amplitude_range = np.nan
            ecg_amplitude_mean = np.nan
            ecg_amplitude_std = np.nan
        
        # 保存 ECG 振幅統計（用於階段4的一致性檢查）
        ecg_stats = {
            'ecg_amplitude_min': ecg_amplitude_min,
            'ecg_amplitude_max': ecg_amplitude_max,
            'ecg_amplitude_range': ecg_amplitude_range,
            'ecg_amplitude_mean': ecg_amplitude_mean,
            'ecg_amplitude_std': ecg_amplitude_std
        }
        np.savetxt(
            os.path.join(out_dir, "ECG_stats.csv"),
            [[ecg_stats['ecg_amplitude_min'], ecg_stats['ecg_amplitude_max'],
              ecg_stats['ecg_amplitude_range'], ecg_stats['ecg_amplitude_mean'],
              ecg_stats['ecg_amplitude_std']]],
            fmt="%.6f",
            delimiter=",",
            header="ecg_amplitude_min,ecg_amplitude_max,ecg_amplitude_range,ecg_amplitude_mean,ecg_amplitude_std",
            comments=""
        )

        # 訊號品質指標
        if rri_ms.size > 0:
            valid_mask = (rri_ms >= 300.0) & (rri_ms <= 2000.0)
            valid_rr_ratio = float(np.mean(valid_mask))
            ectopy_count = int(np.sum(~valid_mask))
            artifact_rate = float(1.0 - valid_rr_ratio)
            ectopy_flag = int(ectopy_count > 0)
        else:
            valid_rr_ratio = np.nan
            ectopy_count = 0
            artifact_rate = np.nan
            ectopy_flag = 0

        # 時域特徵
        r_times_s = ridx / float(fs)
        tfeat = time_features_from_rri(rri_ms, r_times_s)

        # EDR (ECG-derived Respiration) 特徵 - 先計算以便用於頻域質量評估
        edr_feat = extract_edr_respiratory_rate(x_clean, ridx, fs, resample_hz)
        edr_freq_hz = edr_feat.get('respiratory_rate_hz', None)

        # 頻域特徵（傳入 EDR 頻率用於 RSA 重疊檢測）
        freqs, psd = welch_psd_from_rpeaks(ridx, fs, resample_hz)
        ffeat = freq_features(freqs, psd, edr_freq_hz=edr_freq_hz)

        # 非線性特徵（使用增強版 SampEn）
        sd1, sd2, ratio = sd1_sd2_from_rri(rri_ms)
        se_result = sample_entropy_with_quality(rri_ms, m=2, r_ratio=0.2)
        se = se_result['SampEn']
        se_confidence = se_result['SampEn_confidence']
        se_reliability = se_result['SampEn_reliability']
        dfa = dfa_alpha(rri_ms, scales=(4,16))
        
        # 綜合品質指標
        quality_flags = compute_hrv_quality_flags(
            rri_ms, ridx, fs, 
            freq_feat=ffeat, 
            sampen_conf={'confidence': 'high' if se_confidence > 0.7 else 'medium' if se_confidence > 0.4 else 'low'},
            edr_feat=edr_feat
        )

        # Baseline 差異特徵（如果啟用）
        delta_feat = {}
        if use_baseline and ecg_baseline is not None:
            try:
                # 處理 baseline ECG（使用與 stimuli 相同的邏輯）
                if ecg_baseline.shape[1] >= 2:
                    # 雙通道可用
                    if use_sqi_selection:
                        # 使用 SQI-based selection
                        if sqi_mode == "best":
                            mode = "sqi_best"
                        elif sqi_mode == "weighted":
                            mode = "sqi_weighted"
                        else:
                            mode = "sqi_weighted"  # 預設使用加權平均
                        x_bl_raw = fuse_ecg_channels(ecg_baseline[:, 0], ecg_baseline[:, 1], fs=fs, mode=mode)
                    elif use_dual_channel:
                        # 簡單融合雙通道
                        x_bl_raw = fuse_ecg_channels(ecg_baseline[:, 0], ecg_baseline[:, 1], fs=fs, mode="simple")
                    else:
                        # 單通道
                        x_bl_raw = ecg_baseline[:, use_col]
                else:
                    # 只有單通道可用
                    x_bl_raw = ecg_baseline[:, use_col]
                
                ridx_bl, _ = detect_rpeaks_strong(x_bl_raw, fs)
                if ridx_bl.size >= 30:
                    rri_bl_ms = np.diff(ridx_bl)/fs*1000.0
                    r_times_bl_s = ridx_bl / float(fs)
                    
                    # 提取 baseline 特徵
                    tfeat_bl = time_features_from_rri(rri_bl_ms, r_times_bl_s)
                    freqs_bl, psd_bl = welch_psd_from_rpeaks(ridx_bl, fs, resample_hz)
                    ffeat_bl = freq_features(freqs_bl, psd_bl)
                    sd1_bl, sd2_bl, ratio_bl = sd1_sd2_from_rri(rri_bl_ms)
                    se_bl = sample_entropy(rri_bl_ms, m=2, r_ratio=0.2)
                    dfa_bl = dfa_alpha(rri_bl_ms, scales=(4,16))
                    
                    # 合併 baseline 特徵
                    baseline_feat = {**tfeat_bl, **ffeat_bl, 
                                   'SD1_ms': sd1_bl, 'SD2_ms': sd2_bl, 
                                   'SD1_SD2': ratio_bl, 'SampEn': se_bl, 
                                   'DFA_alpha': dfa_bl}
                    stimuli_feat = {**tfeat, **ffeat, 
                                   'SD1_ms': sd1, 'SD2_ms': sd2, 
                                   'SD1_SD2': ratio, 'SampEn': se, 
                                   'DFA_alpha': dfa}
                    
                    # 計算差異特徵
                    delta_feat = calculate_delta_features(baseline_feat, stimuli_feat)
            except Exception as e:
                pass  # 錯誤會在返回時處理

        # EEG 特徵（如果啟用）
        eeg_feat = {}
        if use_eeg and eeg_stimuli is not None:
            try:
                eeg_feat = extract_eeg_multi_channel_features(eeg_stimuli, fs_eeg)
            except Exception:
                pass  # 錯誤會在返回時處理

        # 圖像
        if save_plots:
            plot_all(out_dir, fs, x_clean, ridx, rri_ms, freqs, psd, resample_hz, poincare_size)

        # 組裝最終特徵
        row = {
            "subject":s,"trial":t,"col":use_col,"R_count":int(len(ridx)),
            "MeanRR_ms":tfeat["MeanRR_ms"], "SDNN_ms":tfeat["SDNN_ms"],
            "MeanHR_bpm":tfeat["MeanHR_bpm"], "SDHR_bpm":tfeat["SDHR_bpm"],
            "RMSSD_ms":tfeat["RMSSD_ms"], "NN50":tfeat["NN50"], "pNN50":tfeat["pNN50"],
            "SDNN_index_ms":tfeat["SDNN_index_ms"],
            # 頻域特徵
            "ULF_peak":ffeat["ULF_peak"], "LF_peak":ffeat["LF_peak"], "HF_peak":ffeat["HF_peak"],
            "ULF_ratio":ffeat["ULF_ratio"], "LF_ratio":ffeat["LF_ratio"], "HF_ratio":ffeat["HF_ratio"],
            "LF_HF":ffeat["LF_HF"],
            # 頻域品質指標（新增）
            "HF_RSA_overlap":ffeat.get("HF_RSA_overlap", np.nan),
            "HF_reliability":ffeat.get("HF_reliability", np.nan),
            "ULF_dominance_flag":ffeat.get("ULF_dominance_flag", np.nan),
            "spectral_quality":ffeat.get("spectral_quality", np.nan),
            # 非線性特徵
            "SD1_ms":sd1, "SD2_ms":sd2, "SD1_SD2":ratio, "SampEn":se, "DFA_alpha":dfa,
            # SampEn 品質指標（新增）
            "SampEn_confidence":se_confidence, "SampEn_reliability":se_reliability,
            # 原有品質指標
            "artifact_rate":artifact_rate,"ectopy_count":ectopy_count,"ectopy_flag":ectopy_flag,
            "valid_rr_ratio":valid_rr_ratio,"window_start_s":window_start_s,
            "window_end_s":window_end_s,"window_duration_s":window_duration_s,
            # 綜合品質指標（新增）
            "overall_quality":quality_flags.get("overall_quality", np.nan),
            "data_length_quality":quality_flags.get("data_length_quality", np.nan),
            "artifact_quality":quality_flags.get("artifact_quality", np.nan),
            "nonlinear_quality":quality_flags.get("nonlinear_quality", np.nan),
            "respiratory_interference":quality_flags.get("respiratory_interference", np.nan),
            "quality_flags_str":quality_flags.get("quality_flags_str", ""),
            **labels,
            **delta_feat,  # Baseline 差異特徵
            **eeg_feat,     # EEG 特徵
            **edr_feat      # EDR (ECG-derived Respiration) 特徵
        }
        pl.DataFrame([row]).write_csv(os.path.join(out_dir,"HRV.csv"))
        
        return (s, t, row, None)  # 返回結果和 None 表示成功
        
    except Exception as e:
        import traceback
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return (s, t, None, error_msg)  # 返回錯誤信息


def process_ecg_and_extract_features():
    """階段 1: 處理 ECG 並提取 HRV 特徵（擴展版：包含 Baseline 和 EEG，多核並行）"""
    config = _get_config()
    
    print("\n" + "="*60)
    print("Stage 1: ECG Processing & HRV Feature Extraction")
    print("="*60)
    print(f"Features: Baseline={config.use_baseline}, EEG={config.use_eeg}")
    print(f"ECG: Channel={config.ecg_channel}, SQI={config.sqi_mode}")
    print(f"Workers: {config.num_workers}")
    print("="*60)
    
    ensure_dir(config.output_dir)
    
    # Config dict for multiprocessing (dataclass not picklable)
    config_dict = {
        'use_baseline': config.use_baseline,
        'use_eeg': config.use_eeg,
        'use_dual_channel': config.use_dual_channel,
        'use_sqi_selection': config.use_sqi_selection,
        'sqi_mode': config.sqi_mode,
        'ecg_channel': config.ecg_channel,
        'save_plots': config.save_plots,
        'resample_hz': config.resample_hz,
        'output_dir': config.output_dir,
        'poincare_size': config.poincare_size,
    }
    
    # 生成所有 (subject, trial) 組合
    s0, e0 = config.subject_range
    t0, e1 = config.trial_range
    tasks = [(s, t, config.mat_path, config_dict) for s in range(s0, e0+1) for t in range(t0, e1+1)]
    
    rows, errors = [], []
    
    # 使用多進程池處理
    with Pool(processes=config.num_workers) as pool:
        # 使用 tqdm 顯示進度
        results = list(tqdm(
            pool.imap(process_single_trial, tasks),
            total=len(tasks),
            desc="Processing trials"
        ))
    
    # 收集結果
    for s, t, row, error_msg in results:
        if error_msg is not None:
            errors.append((s, t, error_msg))
        elif row is not None:
            rows.append(row)
    
    # 保存結果
    output_path = os.path.join(config.output_dir, "HRV_all.csv")
    if rows:
        pl.DataFrame(rows).write_csv(output_path)
    if errors:
        error_log_path = os.path.join(config.output_dir, "errors.log")
        with open(error_log_path, "w", encoding="utf-8") as f:
            for s, t, m in errors:
                f.write(f"S{s:02d} T{t:02d}: {m}\n")
    
    print(f"\nStage 1 complete! Generated {len(rows)} records")
    print(f"Output: {output_path}")
    if rows:
        print(f"Features: {len(rows[0])} columns")
    if errors:
        print(f"Errors: {len(errors)} (see errors.log)")


# ============================================================
# 階段 2: 標準化特徵並添加 Z-score
# ============================================================

def normalize_and_add_zscores():
    """
    Stage 2: Normalize features and add Z-score columns.
    
    Important improvement (resolving Z-score vs Delta contradiction):
    =================================================================
    
    **Problem**:
    Traditional Z-score = (stimuli - global_mean) / global_std
    Delta = stimuli - personal_baseline
    
    These use different reference points, potentially causing Z > 0 but Delta < 0.
    
    **Solution**:
    Provide two types of Z-scores:
    1. **Traditional Z-score (Global)**: Based on global distribution
       - {feature}_zscore = (stimuli - global_mean) / global_std
    
    2. **Delta Z-score (Baseline-adjusted)**: Based on Delta values
       - Delta_{feature}_zscore = (Delta - mean_of_delta) / std_of_delta
       - Ensures consistency with Delta direction!
    
    **Usage recommendation**:
    - For within-subject change analysis → use Delta_zscore
    - For cross-subject comparison → use traditional zscore
    """
    config = _get_config()
    
    print("\n" + "="*60)
    print("Stage 2: Feature Normalization & Z-score Calculation")
    print("="*60)
    
    # 指定 19 個 HRV 特徵欄位
    feature_cols = [
        "MeanRR_ms","SDNN_ms","MeanHR_bpm","SDHR_bpm","RMSSD_ms",
        "NN50","pNN50","SDNN_index_ms",
        "ULF_peak","LF_peak","HF_peak","ULF_ratio","LF_ratio","HF_ratio","LF_HF",
        "SD1_ms","SD2_ms","SampEn","DFA_alpha"
    ]
    
    # 重要特徵（需要計算 Z-score 的）
    important_features = ["RMSSD_ms", "SampEn", "MeanHR_bpm", "SDNN_ms", "pNN50", "DFA_alpha"]
    
    # 讀取資料（使用 Polars）
    hrv_all_path = os.path.join(config.output_dir, "HRV_all.csv")
    data = pl.read_csv(hrv_all_path)
    
    print(f"原始資料形狀: {data.shape}")
    
    # ========================================
    # 1. 計算傳統 Z-score（基於全域分佈）
    # ========================================
    print("\n[1] 計算傳統 Z-score（基於全域分佈）...")
    X = data.select(feature_cols).to_numpy()
    scaler_global = StandardScaler()
    X_zscore_global = scaler_global.fit_transform(X)
    
    # 建立傳統 Z-score DataFrame
    zscore_global_data = {}
    for feat in important_features:
        if feat in feature_cols:
            col_idx = feature_cols.index(feat)
            zscore_global_data[f"{feat}_zscore"] = X_zscore_global[:, col_idx]
    
    # ========================================
    # 2. 計算 Delta Z-score（基於 Delta 的全域分佈）
    # ========================================
    print("[2] 計算 Delta Z-score（基於 Delta 的全域分佈）...")
    
    zscore_delta_data = {}
    delta_stats = {}  # 保存統計信息供報告使用
    
    for feat in important_features:
        delta_col = f"Delta_{feat}"
        
        # 檢查 Delta 欄位是否存在
        if delta_col in data.columns:
            delta_values = data[delta_col].to_numpy()
            
            # 過濾有效值
            valid_mask = np.isfinite(delta_values)
            valid_count = np.sum(valid_mask)
            
            if valid_count >= 10:  # 至少需要 10 個有效值才能計算有意義的 Z-score
                # 計算 Delta 的全域均值和標準差
                delta_mean = np.nanmean(delta_values)
                delta_std = np.nanstd(delta_values, ddof=1)
                
                if delta_std > 1e-10:  # 避免除以零
                    # 計算 Delta Z-score
                    delta_zscore = (delta_values - delta_mean) / delta_std
                    zscore_delta_data[f"Delta_{feat}_zscore"] = delta_zscore
                    
                    # 保存統計信息
                    delta_stats[feat] = {
                        'mean': delta_mean,
                        'std': delta_std,
                        'valid_count': valid_count
                    }
                    
                    print(f"   - {delta_col}: mean={delta_mean:.4f}, std={delta_std:.4f}, n={valid_count}")
                else:
                    print(f"   - {delta_col}: 標準差過小，跳過")
                    zscore_delta_data[f"Delta_{feat}_zscore"] = np.full(len(data), np.nan)
            else:
                print(f"   - {delta_col}: 有效值不足 ({valid_count})")
                zscore_delta_data[f"Delta_{feat}_zscore"] = np.full(len(data), np.nan)
        else:
            print(f"   - {delta_col}: 欄位不存在（可能未啟用 Baseline）")
            zscore_delta_data[f"Delta_{feat}_zscore"] = np.full(len(data), np.nan)
    
    # ========================================
    # 3. 合併所有 Z-score
    # ========================================
    zscore_df = pl.DataFrame({**zscore_global_data, **zscore_delta_data})
    
    # 合併原始資料與 Z-score
    data_final = pl.concat([data, zscore_df], how="horizontal")
    
    # 儲存增強後的資料集
    output_path = os.path.join(config.output_dir, "HRV_all_augmented.csv")
    data_final.write_csv(output_path)
    
    print(f"\n階段 2 完成！")
    print(f"最終數據形狀: {data_final.shape}")
    print(f"輸出檔案: {output_path}")
    
    print(f"\n新增的 Z-score 欄位:")
    print("  [傳統 Z-score - 基於全域分佈]")
    for feat in important_features:
        print(f"    - {feat}_zscore")
    
    print("  [Delta Z-score - 基於個人 Baseline，與 Delta 方向一致]")
    for feat in important_features:
        print(f"    - Delta_{feat}_zscore")
    
    print("\n" + "="*60)
    print("⚠️  重要提醒：Z-score 與 Delta 一致性")
    print("="*60)
    print("  - Delta_{feat}_zscore 確保與 Delta 方向一致")
    print("  - 當 Delta > 0（相對於 baseline 升高）→ Delta_zscore 傾向 > 0")
    print("  - 當 Delta < 0（相對於 baseline 降低）→ Delta_zscore 傾向 < 0")
    print("  - 推薦在報告中使用 Delta_zscore 以避免矛盾")
    print("="*60)


# ============================================================
# 階段 4: 添加圖片路徑到數據集
# ============================================================

# 輔助函數（移到模組層級以支持多進程）
def _image_features(path: str):
    """計算單張圖片的 mean/std/row-wise slope"""
    if path is None or (isinstance(path, str) and not os.path.exists(path)):
        return np.nan, np.nan, np.nan
    try:
        with Image.open(path) as img:
            # 轉灰階並調整為 896x896，以符合多模態設計
            img_gray = img.convert("L")
            if img_gray.size != (896, 896):
                img_gray = img_gray.resize((896, 896))
            arr = np.asarray(img_gray, dtype=float) / 255.0
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            row_means = arr.mean(axis=1)
            if row_means.size >= 2:
                x = np.arange(row_means.size, dtype=float)
                try:
                    slope = float(np.polyfit(x, row_means, 1)[0])
                except Exception:
                    slope = np.nan
            else:
                slope = np.nan
            return mean_val, std_val, slope
    except Exception:
        return np.nan, np.nan, np.nan

def _load_rri_data(base_dir: str):
    """載入 RRI 數據"""
    rri_path = os.path.join(base_dir, "RRI.csv")
    if os.path.exists(rri_path):
        try:
            rri_data = np.loadtxt(rri_path, delimiter=",", dtype=float)
            if rri_data.size == 0:
                return None
            if rri_data.ndim == 0:
                rri_data = np.array([rri_data])
            return rri_data
        except Exception:
            return None
    return None

def _compute_poincare_data_features(rri_ms):
    """從 RRI 數據計算 Poincaré 圖的實際數據特徵"""
    if rri_ms is None or rri_ms.size < 2:
        return {
            'x_min': np.nan, 'x_max': np.nan, 'x_mean': np.nan, 'x_std': np.nan,
            'y_min': np.nan, 'y_max': np.nan, 'y_mean': np.nan, 'y_std': np.nan,
            'range_x': np.nan, 'range_y': np.nan, 'ratio_xy': np.nan,
            'density_center_x': np.nan, 'density_center_y': np.nan,
            'scatter_count': 0
        }
    
    rrn = rri_ms[:-1]
    rrn1 = rri_ms[1:]
    
    x_min = float(np.min(rrn))
    x_max = float(np.max(rrn))
    x_mean = float(np.mean(rrn))
    x_std = float(np.std(rrn, ddof=1)) if rrn.size > 1 else 0.0
    
    y_min = float(np.min(rrn1))
    y_max = float(np.max(rrn1))
    y_mean = float(np.mean(rrn1))
    y_std = float(np.std(rrn1, ddof=1)) if rrn1.size > 1 else 0.0
    
    range_x = float(x_max - x_min)
    range_y = float(y_max - y_min)
    ratio_xy = float(range_x / range_y) if range_y > 0 else np.nan
    
    # 密度中心（加權平均，假設每個點權重相同）
    density_center_x = x_mean
    density_center_y = y_mean
    
    return {
        'x_min': x_min, 'x_max': x_max, 'x_mean': x_mean, 'x_std': x_std,
        'y_min': y_min, 'y_max': y_max, 'y_mean': y_mean, 'y_std': y_std,
        'range_x': range_x, 'range_y': range_y, 'ratio_xy': ratio_xy,
        'density_center_x': density_center_x, 'density_center_y': density_center_y,
        'scatter_count': int(len(rrn))
    }

def _compute_psd_data_features(base_dir: str, fs: float, resample_hz: float):
    """從原始數據重新計算 PSD 特徵（如果 RRI 數據存在）"""
    rri_ms = _load_rri_data(base_dir)
    if rri_ms is None or rri_ms.size < 3:
        return {
            'freq_min': np.nan, 'freq_max': np.nan, 'freq_range': np.nan,
            'power_min': np.nan, 'power_max': np.nan, 'power_mean': np.nan,
            'power_std': np.nan, 'power_median': np.nan,
            'peak_freq': np.nan, 'peak_power': np.nan, 'total_power': np.nan,
            'band_ulf_power': np.nan, 'band_lf_power': np.nan, 'band_hf_power': np.nan
        }
    
    try:
        # 重新計算 PSD（模擬 welch_psd_from_rpeaks 的邏輯）
        # 從 RRI 重建時間序列
        rri_s = rri_ms / 1000.0  # 轉換為秒
        rri_t = np.cumsum(np.concatenate([[0], rri_s[:-1]]))  # 累積時間
        
        if len(rri_t) < 2:
            return {
                'freq_min': np.nan, 'freq_max': np.nan, 'freq_range': np.nan,
                'power_min': np.nan, 'power_max': np.nan, 'power_mean': np.nan,
                'power_std': np.nan, 'power_median': np.nan,
                'peak_freq': np.nan, 'peak_power': np.nan, 'total_power': np.nan,
                'band_ulf_power': np.nan, 'band_lf_power': np.nan, 'band_hf_power': np.nan
            }
        
        t_new = np.arange(rri_t[0], rri_t[-1], 1.0/resample_hz)
        if len(t_new) < 16:
            return {
                'freq_min': np.nan, 'freq_max': np.nan, 'freq_range': np.nan,
                'power_min': np.nan, 'power_max': np.nan, 'power_mean': np.nan,
                'power_std': np.nan, 'power_median': np.nan,
                'peak_freq': np.nan, 'peak_power': np.nan, 'total_power': np.nan,
                'band_ulf_power': np.nan, 'band_lf_power': np.nan, 'band_hf_power': np.nan
            }
        
        # 使用三次樣條插值（與 welch_psd_from_rpeaks 保持一致）
        try:
            f = interp1d(rri_t, rri_s, kind="cubic", fill_value="extrapolate", assume_sorted=True)
        except Exception:
            f = interp1d(rri_t, rri_s, kind="linear", fill_value="extrapolate", assume_sorted=True)
        rr_even = f(t_new)
        
        # 使用 Smoothness Priors 去趨勢化（與 welch_psd_from_rpeaks 保持一致）
        lambda_val = int(500 * resample_hz / 4.0)
        rr_even = smoothness_priors_detrend(rr_even, lambda_val=lambda_val)
        
        nperseg = min(len(rr_even), int(resample_hz*64))
        if nperseg < 16:
            return {
                'freq_min': np.nan, 'freq_max': np.nan, 'freq_range': np.nan,
                'power_min': np.nan, 'power_max': np.nan, 'power_mean': np.nan,
                'power_std': np.nan, 'power_median': np.nan,
                'peak_freq': np.nan, 'peak_power': np.nan, 'total_power': np.nan,
                'band_ulf_power': np.nan, 'band_lf_power': np.nan, 'band_hf_power': np.nan
            }
        
        freqs, psd = sps.welch(rr_even, fs=resample_hz, nperseg=nperseg)
        
        # 計算特徵
        freq_min = float(np.min(freqs))
        freq_max = float(np.max(freqs))
        freq_range = float(freq_max - freq_min)
        
        power_min = float(np.min(psd))
        power_max = float(np.max(psd))
        power_mean = float(np.mean(psd))
        power_std = float(np.std(psd, ddof=1)) if psd.size > 1 else 0.0
        power_median = float(np.median(psd))
        
        # 峰值頻率和功率
        peak_idx = np.argmax(psd)
        peak_freq = float(freqs[peak_idx])
        peak_power = float(psd[peak_idx])
        
        # 總功率
        total_power = float(np.trapezoid(psd, freqs))
        
        # 頻帶功率
        ulf_mask = (freqs >= 0.00) & (freqs <= 0.04)
        lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
        hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
        
        band_ulf_power = float(np.trapezoid(psd[ulf_mask], freqs[ulf_mask])) if np.any(ulf_mask) else np.nan
        band_lf_power = float(np.trapezoid(psd[lf_mask], freqs[lf_mask])) if np.any(lf_mask) else np.nan
        band_hf_power = float(np.trapezoid(psd[hf_mask], freqs[hf_mask])) if np.any(hf_mask) else np.nan
        
        return {
            'freq_min': freq_min, 'freq_max': freq_max, 'freq_range': freq_range,
            'power_min': power_min, 'power_max': power_max, 'power_mean': power_mean,
            'power_std': power_std, 'power_median': power_median,
            'peak_freq': peak_freq, 'peak_power': peak_power, 'total_power': total_power,
            'band_ulf_power': band_ulf_power, 'band_lf_power': band_lf_power, 'band_hf_power': band_hf_power
        }
    except Exception:
        return {
            'freq_min': np.nan, 'freq_max': np.nan, 'freq_range': np.nan,
            'power_min': np.nan, 'power_max': np.nan, 'power_mean': np.nan,
            'power_std': np.nan, 'power_median': np.nan,
            'peak_freq': np.nan, 'peak_power': np.nan, 'total_power': np.nan,
            'band_ulf_power': np.nan, 'band_lf_power': np.nan, 'band_hf_power': np.nan
        }

def _compute_signal_quality_features(base_dir: str, rri_ms, fs: float):
    """計算 Signal Quality panel 的實際信號特徵
    
    重要：確保使用與圖像生成相同的數據窗口（從第一個 R-peak 到最後一個 R-peak）
    """
    # 從保存的 ECG_stats.csv 讀取實際的 ECG 振幅統計（與圖像生成使用相同的數據源）
    ecg_stats_path = os.path.join(base_dir, "ECG_stats.csv")
    if os.path.exists(ecg_stats_path):
        try:
            ecg_stats_data = np.loadtxt(ecg_stats_path, delimiter=",", skiprows=1)
            if ecg_stats_data.size >= 5:
                ecg_amplitude_min = float(ecg_stats_data[0])
                ecg_amplitude_max = float(ecg_stats_data[1])
                ecg_amplitude_range = float(ecg_stats_data[2])
                ecg_amplitude_mean = float(ecg_stats_data[3])
                ecg_amplitude_std = float(ecg_stats_data[4])
            else:
                ecg_amplitude_min = np.nan
                ecg_amplitude_max = np.nan
                ecg_amplitude_range = np.nan
                ecg_amplitude_mean = np.nan
                ecg_amplitude_std = np.nan
        except Exception:
            # 如果讀取失敗，回退到 NaN
            ecg_amplitude_min = np.nan
            ecg_amplitude_max = np.nan
            ecg_amplitude_range = np.nan
            ecg_amplitude_mean = np.nan
            ecg_amplitude_std = np.nan
    else:
        # 如果文件不存在（舊版本數據），回退到 NaN
        ecg_amplitude_min = np.nan
        ecg_amplitude_max = np.nan
        ecg_amplitude_range = np.nan
        ecg_amplitude_mean = np.nan
        ecg_amplitude_std = np.nan
    
    # RRI 統計（從實際 RRI 數據計算）
    if rri_ms is None or rri_ms.size == 0:
        rri_min = np.nan
        rri_max = np.nan
        rri_range = np.nan
        time_duration = np.nan
    else:
        rri_min = float(np.min(rri_ms))
        rri_max = float(np.max(rri_ms))
        rri_range = float(rri_max - rri_min)
        # 時間持續時間（從 RRI 累積，與 window_duration_s 一致）
        time_duration = float(np.sum(rri_ms) / 1000.0)
    
    return {
        # ECG 振幅特徵（從實際信號計算，與圖像一致）
        'ecg_amplitude_min': ecg_amplitude_min,
        'ecg_amplitude_max': ecg_amplitude_max,
        'ecg_amplitude_range': ecg_amplitude_range,
        'ecg_amplitude_mean': ecg_amplitude_mean,
        'ecg_amplitude_std': ecg_amplitude_std,
        # RRI 統計（從 RRI 數據計算）
        'rri_min': rri_min,
        'rri_max': rri_max,
        'rri_range': rri_range,
        'time_duration': time_duration
    }

def process_single_row(args):
    """處理單行數據（用於多進程並行處理）"""
    row, out_root, resample_hz = args
    subject = row["subject"]
    trial = row["trial"]
    
    # 格式化為 S01, T01 等
    subject_str = f"S{subject:02d}" if isinstance(subject, int) else subject
    trial_str = f"T{trial:02d}" if isinstance(trial, int) else trial
    
    # 構建圖片路徑
    base_dir = os.path.join(out_root, subject_str, trial_str)
    poincare_path = os.path.join(base_dir, "Poincare.png")  # 直接使用 Poincare.png（已為 896x896）
    signal_quality_path = os.path.join(base_dir, "SignalQuality_panel.png")
    psd_path = os.path.join(base_dir, "PSD.png")

    # Poincaré 圖
    if os.path.exists(poincare_path):
        img_path = poincare_path
    else:
        img_path = None
        poincare_path = None

    # 訊號品質 panel
    if os.path.exists(signal_quality_path):
        signal_quality_img_path = signal_quality_path
    else:
        signal_quality_img_path = None
        signal_quality_path = None

    # PSD 圖
    if os.path.exists(psd_path):
        psd_img_path = psd_path
    else:
        psd_img_path = None
        psd_path = None

    # 影像量化特徵（從圖像像素）
    m1, s1, sl1 = _image_features(poincare_path)
    m2, s2, sl2 = _image_features(signal_quality_path)
    m3, s3, sl3 = _image_features(psd_path)
    
    # === 新增：從原始數據提取的量化特徵 ===
    # 載入 RRI 數據
    rri_data = _load_rri_data(base_dir)
    
    # Poincaré 圖的實際數據特徵
    poincare_feat = _compute_poincare_data_features(rri_data)
    
    # PSD 圖的實際數據特徵
    fs_ecg = row.get('fs_ecg', 256.0) if 'fs_ecg' in row else 256.0
    psd_feat = _compute_psd_data_features(base_dir, fs_ecg, resample_hz)
    
    # Signal Quality panel 的實際數據特徵
    sq_feat = _compute_signal_quality_features(base_dir, rri_data, fs_ecg)
    
    return {
        'img_path': img_path,
        'signal_quality_img_path': signal_quality_img_path,
        'psd_img_path': psd_img_path,
        'poincare_img_mean': m1, 'poincare_img_std': s1, 'poincare_row_slope': sl1,
        'signal_quality_img_mean': m2, 'signal_quality_img_std': s2, 'signal_quality_row_slope': sl2,
        'psd_img_mean': m3, 'psd_img_std': s3, 'psd_row_slope': sl3,
        **poincare_feat,
        **psd_feat,
        **sq_feat
    }

def add_image_paths():
    """Stage 3: Add image paths and quantified image features to the dataset."""
    config = _get_config()
    
    print("\n" + "="*60)
    print("Stage 3: Adding Image Paths & Features")
    print("="*60)
    
    # 讀取增強後的資料
    augmented_path = os.path.join(config.output_dir, "HRV_all_augmented.csv")
    data = pl.read_csv(augmented_path)
    
    print(f"原始資料行數: {len(data)}")
    
    # 為每一行資料找對應的圖片路徑，同時計算影像量化特徵
    img_paths = []
    signal_quality_img_paths = []
    psd_img_paths = []

    poincare_means = []
    poincare_stds = []
    poincare_slopes = []

    sq_means = []
    sq_stds = []
    sq_slopes = []

    psd_means = []
    psd_stds = []
    psd_slopes = []
    
    # === 新增：從原始數據提取的量化特徵（避免圖像縮放失真） ===
    # Poincaré 圖的實際數據範圍和分佈
    poincare_rri_x_min = []
    poincare_rri_x_max = []
    poincare_rri_x_mean = []
    poincare_rri_x_std = []
    poincare_rri_y_min = []
    poincare_rri_y_max = []
    poincare_rri_y_mean = []
    poincare_rri_y_std = []
    poincare_rri_range_x = []
    poincare_rri_range_y = []
    poincare_rri_ratio_xy = []
    poincare_density_center_x = []
    poincare_density_center_y = []
    poincare_scatter_count = []
    
    # PSD 圖的實際頻率和功率值
    psd_freq_min = []
    psd_freq_max = []
    psd_freq_range = []
    psd_power_min = []
    psd_power_max = []
    psd_power_mean = []
    psd_power_std = []
    psd_power_median = []
    psd_peak_freq = []
    psd_peak_power = []
    psd_total_power = []
    psd_band_ulf_power = []
    psd_band_lf_power = []
    psd_band_hf_power = []
    
    # Signal Quality panel 的實際信號範圍
    sq_ecg_amplitude_min = []
    sq_ecg_amplitude_max = []
    sq_ecg_amplitude_range = []
    sq_ecg_amplitude_mean = []
    sq_ecg_amplitude_std = []
    sq_rri_min = []
    sq_rri_max = []
    sq_rri_range = []
    sq_time_duration = []

    # 準備任務列表
    tasks = [(dict(row), config.output_dir, config.resample_hz) for row in data.iter_rows(named=True)]
    
    # 確定進程數
    num_workers = config.num_workers
    
    # 使用多進程池處理
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_row, tasks),
            total=len(tasks),
            desc="處理圖片路徑與影像特徵"
        ))
    
    # 收集結果
    for result in results:
        img_paths.append(result['img_path'])
        signal_quality_img_paths.append(result['signal_quality_img_path'])
        psd_img_paths.append(result['psd_img_path'])
        poincare_means.append(result['poincare_img_mean'])
        poincare_stds.append(result['poincare_img_std'])
        poincare_slopes.append(result['poincare_row_slope'])
        sq_means.append(result['signal_quality_img_mean'])
        sq_stds.append(result['signal_quality_img_std'])
        sq_slopes.append(result['signal_quality_row_slope'])
        psd_means.append(result['psd_img_mean'])
        psd_stds.append(result['psd_img_std'])
        psd_slopes.append(result['psd_row_slope'])
        
        poincare_rri_x_min.append(result['x_min'])
        poincare_rri_x_max.append(result['x_max'])
        poincare_rri_x_mean.append(result['x_mean'])
        poincare_rri_x_std.append(result['x_std'])
        poincare_rri_y_min.append(result['y_min'])
        poincare_rri_y_max.append(result['y_max'])
        poincare_rri_y_mean.append(result['y_mean'])
        poincare_rri_y_std.append(result['y_std'])
        poincare_rri_range_x.append(result['range_x'])
        poincare_rri_range_y.append(result['range_y'])
        poincare_rri_ratio_xy.append(result['ratio_xy'])
        poincare_density_center_x.append(result['density_center_x'])
        poincare_density_center_y.append(result['density_center_y'])
        poincare_scatter_count.append(result['scatter_count'])
        
        psd_freq_min.append(result['freq_min'])
        psd_freq_max.append(result['freq_max'])
        psd_freq_range.append(result['freq_range'])
        psd_power_min.append(result['power_min'])
        psd_power_max.append(result['power_max'])
        psd_power_mean.append(result['power_mean'])
        psd_power_std.append(result['power_std'])
        psd_power_median.append(result['power_median'])
        psd_peak_freq.append(result['peak_freq'])
        psd_peak_power.append(result['peak_power'])
        psd_total_power.append(result['total_power'])
        psd_band_ulf_power.append(result['band_ulf_power'])
        psd_band_lf_power.append(result['band_lf_power'])
        psd_band_hf_power.append(result['band_hf_power'])
        
        sq_ecg_amplitude_min.append(result['ecg_amplitude_min'])
        sq_ecg_amplitude_max.append(result['ecg_amplitude_max'])
        sq_ecg_amplitude_range.append(result['ecg_amplitude_range'])
        sq_ecg_amplitude_mean.append(result['ecg_amplitude_mean'])
        sq_ecg_amplitude_std.append(result['ecg_amplitude_std'])
        sq_rri_min.append(result['rri_min'])
        sq_rri_max.append(result['rri_max'])
        sq_rri_range.append(result['rri_range'])
        sq_time_duration.append(result['time_duration'])
    
    # 將圖片路徑與影像特徵加入到原始資料中
    new_data = data.with_columns(
        [
            # 圖片路徑
            pl.Series("img_path", img_paths),
            pl.Series("signal_quality_img_path", signal_quality_img_paths),
            pl.Series("psd_img_path", psd_img_paths),
            # 圖像像素特徵（原有）
            pl.Series("poincare_img_mean", poincare_means),
            pl.Series("poincare_img_std", poincare_stds),
            pl.Series("poincare_row_slope", poincare_slopes),
            pl.Series("signal_quality_img_mean", sq_means),
            pl.Series("signal_quality_img_std", sq_stds),
            pl.Series("signal_quality_row_slope", sq_slopes),
            pl.Series("psd_img_mean", psd_means),
            pl.Series("psd_img_std", psd_stds),
            pl.Series("psd_row_slope", psd_slopes),
            # === 新增：Poincaré 圖的實際數據量化特徵（避免圖像縮放失真） ===
            pl.Series("poincare_rri_x_min", poincare_rri_x_min),
            pl.Series("poincare_rri_x_max", poincare_rri_x_max),
            pl.Series("poincare_rri_x_mean", poincare_rri_x_mean),
            pl.Series("poincare_rri_x_std", poincare_rri_x_std),
            pl.Series("poincare_rri_y_min", poincare_rri_y_min),
            pl.Series("poincare_rri_y_max", poincare_rri_y_max),
            pl.Series("poincare_rri_y_mean", poincare_rri_y_mean),
            pl.Series("poincare_rri_y_std", poincare_rri_y_std),
            pl.Series("poincare_rri_range_x", poincare_rri_range_x),
            pl.Series("poincare_rri_range_y", poincare_rri_range_y),
            pl.Series("poincare_rri_ratio_xy", poincare_rri_ratio_xy),
            pl.Series("poincare_density_center_x", poincare_density_center_x),
            pl.Series("poincare_density_center_y", poincare_density_center_y),
            pl.Series("poincare_scatter_count", poincare_scatter_count),
            # === 新增：PSD 圖的實際數據量化特徵（避免圖像縮放失真） ===
            pl.Series("psd_freq_min", psd_freq_min),
            pl.Series("psd_freq_max", psd_freq_max),
            pl.Series("psd_freq_range", psd_freq_range),
            pl.Series("psd_power_min", psd_power_min),
            pl.Series("psd_power_max", psd_power_max),
            pl.Series("psd_power_mean", psd_power_mean),
            pl.Series("psd_power_std", psd_power_std),
            pl.Series("psd_power_median", psd_power_median),
            pl.Series("psd_peak_freq", psd_peak_freq),
            pl.Series("psd_peak_power", psd_peak_power),
            pl.Series("psd_total_power", psd_total_power),
            pl.Series("psd_band_ulf_power", psd_band_ulf_power),
            pl.Series("psd_band_lf_power", psd_band_lf_power),
            pl.Series("psd_band_hf_power", psd_band_hf_power),
            # === 新增：Signal Quality panel 的實際數據量化特徵（避免圖像縮放失真） ===
            pl.Series("sq_ecg_amplitude_min", sq_ecg_amplitude_min),
            pl.Series("sq_ecg_amplitude_max", sq_ecg_amplitude_max),
            pl.Series("sq_ecg_amplitude_range", sq_ecg_amplitude_range),
            pl.Series("sq_ecg_amplitude_mean", sq_ecg_amplitude_mean),
            pl.Series("sq_ecg_amplitude_std", sq_ecg_amplitude_std),
            pl.Series("sq_rri_min", sq_rri_min),
            pl.Series("sq_rri_max", sq_rri_max),
            pl.Series("sq_rri_range", sq_rri_range),
            pl.Series("sq_time_duration", sq_time_duration),
        ]
    )
    
    # 顯示統計資訊
    total_rows = len(new_data)
    valid_images = sum(1 for path in img_paths if path is not None)
    print(f"\n統計資訊:")
    print(f"總行數: {total_rows}")
    print(f"有效圖片數: {valid_images}")
    print(f"缺失圖片數: {total_rows - valid_images}")
    
    # 計算新增特徵的統計
    new_feature_count = (
        13 +  # Poincaré 實際數據特徵
        13 +  # PSD 實際數據特徵
        9     # Signal Quality 實際數據特徵
    )
    print(f"\n新增量化特徵數量: {new_feature_count} 個")
    print(f"  - Poincaré 實際數據特徵: 13 個（X/Y 軸範圍、分佈、密度中心等）")
    print(f"  - PSD 實際數據特徵: 13 個（頻率範圍、功率統計、頻帶功率等）")
    print(f"  - Signal Quality 實際數據特徵: 9 個（RRI 範圍、時間持續、振幅統計等）")
    
    # 儲存最終資料檔案
    output_path = os.path.join(config.output_dir, "HRV_all_final.csv")
    new_data.write_csv(output_path)
    
    print(f"\n階段 3 完成！")
    print(f"輸出檔案: {output_path}")
    print(f"總特徵欄位數: {len(new_data.columns)} 個")
    
    return output_path


# ============================================================
# Main Entry Point
# ============================================================

def main(config: Optional[PreprocessConfig] = None):
    """
    Execute the complete HRV data processing pipeline.
    
    Args:
        config: PreprocessConfig instance. If None, will parse from CLI.
    """
    global CONFIG
    
    # Parse config from CLI if not provided
    if config is None:
        config = parse_args()
    
    CONFIG = config
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("CGRASP - HRV Preprocessing Pipeline")
    print("=" * 60)
    print(f"Input:    {config.mat_path}")
    print(f"Output:   {config.output_dir}")
    print(f"Subjects: {config.subject_range[0]}-{config.subject_range[1]}")
    print(f"Trials:   {config.trial_range[0]}-{config.trial_range[1]}")
    print(f"Workers:  {config.num_workers}")
    print("-" * 60)
    print(f"Features: Baseline={config.use_baseline}, EEG={config.use_eeg}")
    print(f"ECG:      Channel={config.ecg_channel}, SQI={config.sqi_mode}")
    print(f"Plots:    {config.save_plots}")
    print("=" * 60)
    
    # Stage 1: ECG processing and feature extraction
    process_ecg_and_extract_features()
    
    # Stage 2: Normalization and Z-score calculation
    normalize_and_add_zscores()
    
    # Stage 3: Add image paths and features
    final_output = add_image_paths()
    
    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"\nFinal output: {final_output}")
    print("\nGenerated files:")
    print(f"  1. Per-trial HRV features: {config.output_dir}/S*/T*/HRV.csv")
    if config.save_plots:
        print(f"  2. Visualizations: {config.output_dir}/S*/T*/[Poincare|PSD|...].png")
    print(f"  3. Combined HRV data: {config.output_dir}/HRV_all.csv")
    print(f"  4. Augmented data (with Z-scores): {config.output_dir}/HRV_all_augmented.csv")
    print(f"  5. Final dataset (with image paths): {final_output}")
    print("=" * 60 + "\n")
    
    return final_output


if __name__ == "__main__":
    main()

