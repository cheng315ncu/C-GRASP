#!/usr/bin/env python3
"""
CGRASP - Clinical Grade RAG System for Psychophysiology
Unified Configuration Module

All paths can be configured via environment variables.
See .env.example for reference.
"""

import os
from pathlib import Path
from transformers import BitsAndBytesConfig

# ============================================================================
# Base Paths (auto-detect project root)
# ============================================================================

# Project root directory (where this config.py is located)
PROJECT_ROOT = Path(__file__).parent.resolve()

def _get_path(env_var: str, default: str) -> str:
    """Get path from environment variable or use default (relative to PROJECT_ROOT)."""
    value = os.environ.get(env_var)
    if value:
        return str(Path(value).resolve())
    # If default is relative, make it relative to PROJECT_ROOT
    default_path = Path(default)
    if not default_path.is_absolute():
        return str(PROJECT_ROOT / default_path)
    return default

# ============================================================================
# Data & Output Paths (configurable via environment variables)
# ============================================================================

# Input CSV path - REQUIRED: set via CGRASP_CSV_PATH or place in data/
CSV_PATH = _get_path("CGRASP_CSV_PATH", "data/HRV_all_final.csv")

# Output directory for generated reports
OUTPUT_DIR = _get_path("CGRASP_OUTPUT_DIR", "outputs/reports")

# ============================================================================
# PIKE-RAG Configuration
# ============================================================================

# Clinical PDF directory for RAG knowledge base
CLINICAL_PDF_DIR = _get_path("CGRASP_PDF_DIR", "clinical_pdfs")

# ChromaDB vector store directory
CHROMA_DB_DIR = _get_path("CGRASP_CHROMA_DIR", "chroma_db")
ENABLE_EEG_ANALYSIS = False                    # 是否執行 EEG 多模態分析
ENABLE_RAG = True                              # 是否啟用 RAG
ENABLE_GUARDRAILS = True                       # 是否啟用 Guardrails（呼吸干擾/非線性可信度/頻域限制）
ENABLE_DELTA_ZSCORE = True                     # 是否啟用 Delta z-score（個體化 baseline 優先）
TEMPERATURE = 0.3                              # 降低溫度以
TOP_P = 0.85                                   # 略微降低 top_p
INFERENCE_OUTPUT_LENGTH = 1024                 # 推理輸出長度（Step 1-7）
SUMMARY_OUTPUT_LENGTH = 4096                   # 摘要輸出長度（Step 8，必須足夠長以完成 7 步推理 + 6 項答案）
RAG_TOP_K = 5                                  # 檢索前 K 個文件
RAG_SCORE_THRESHOLD = 0.25                     # 相關性閾值（0.3-0.5 推薦，0.5 過於嚴格）
RAG_QUERY_VERBOSITY = "balanced"         # 查詢詳細程度 ("minimal" | "balanced" | "comprehensive")
RAG_FORCE_REBUILD = True                        # 向量庫已建立，設為 False 避免重建
RAG_COLLECTION_NAME = "clinical_knowledge_base"  # 使用已建立的 Qwen3-Embedding 向量庫
# 嵌入模型選擇（按記憶體需求排序，由小到大）
# RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~90MB，最輕量
RAG_EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"               # ~420MB，生物醫學專用
# RAG_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"                   # ~8GB，需要大量 GPU 記憶體

# ============================================================================
# Model Configuration
# ============================================================================

# Model selection (can be overridden via environment variable or CLI)
# Supported models:
#   - "Qwen/Qwen3-VL-8B-Instruct" (recommended, good balance)
#   - "Qwen/Qwen3-VL-4B-Instruct" (lighter, faster)
#   - "google/medgemma-4b-it" (medical-specialized)
#   - "Cannae-AI/MedicalQwen3-Reasoning-14B-IT" (larger, slower)
MODEL_ID = os.environ.get("CGRASP_MODEL_ID", "Qwen/Qwen3-VL-8B-Instruct")

# BitsAndBytesConfig 用於 4-bit 量化，以節省記憶體
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="fp4",
    bnb_4bit_compute_dtype="bfloat16",  # 使用字符串而不是 torch.bfloat16
)

USE_4BIT = False


if USE_4BIT:
    quantization_config = bnb_config
else:
    quantization_config = None

# ============================================================================
# 測試參數
# ============================================================================

TEST = False  # True for testing, False for full run
TRAILS = 18  # Total 18 Trails
SUBJECTS = 23 # Total 23 Subjects

"""
# config.py 切換（Ablation 實驗用）

# Full system
ENABLE_RAG = True
ENABLE_GUARDRAILS = True
ENABLE_DELTA_ZSCORE = True

# w/o RAG
ENABLE_RAG = False
ENABLE_GUARDRAILS = True
ENABLE_DELTA_ZSCORE = True

# w/o guardrails
ENABLE_RAG = True
ENABLE_GUARDRAILS = False
ENABLE_DELTA_ZSCORE = True

# w/o Δz (Delta z-score)
ENABLE_RAG = True
ENABLE_GUARDRAILS = True
ENABLE_DELTA_ZSCORE = False

# Minimal baseline
ENABLE_RAG = False
ENABLE_GUARDRAILS = False
ENABLE_DELTA_ZSCORE = False
"""
