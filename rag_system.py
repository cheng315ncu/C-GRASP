#!/usr/bin/env python3
"""
MedGemma + PIKE-RAG 整合版本 - RAG 系統模組
管理 PIKE-RAG 臨床知識檢索系統的初始化和配置
"""

import os
from typing import Optional
from pikerag_medical_integration import (
    ClinicalKnowledgeRetriever,
    create_hrv_analysis_query,
    METADATA_LIST_DELIMITER,
)


def initialize_rag_system(enable: Optional[bool] = None, config=None):
    """
    初始化 PIKE-RAG 臨床知識檢索系統

    參數:
        enable: 如為 None 則使用全域 ENABLE_RAG；否則以此參數為準。
        config: 配置模組，包含路徑和參數設定

    返回:
        ClinicalKnowledgeRetriever 實例，或 None（如果初始化失敗或停用）
    """
    # 使用配置模組中的參數
    if config is None:
        from config import ENABLE_RAG, CLINICAL_PDF_DIR, CHROMA_DB_DIR, RAG_FORCE_REBUILD, RAG_EMBEDDING_MODEL, RAG_COLLECTION_NAME
    else:
        ENABLE_RAG = config.ENABLE_RAG
        CLINICAL_PDF_DIR = config.CLINICAL_PDF_DIR
        CHROMA_DB_DIR = config.CHROMA_DB_DIR
        RAG_FORCE_REBUILD = config.RAG_FORCE_REBUILD
        RAG_EMBEDDING_MODEL = config.RAG_EMBEDDING_MODEL
        RAG_COLLECTION_NAME = getattr(config, 'RAG_COLLECTION_NAME', 'clinical_hrv_knowledge')

    # 允許由呼叫端覆寫是否啟用 RAG
    rag_enabled = ENABLE_RAG if enable is None else bool(enable)

    if not rag_enabled:
        print("\n" + "="*80)
        print("RAG 檢索增強已停用")
        print("將使用原始 MedGemma 模型，不注入外部臨床知識")
        print("="*80 + "\n")
        return None

    print("\n" + "="*80)
    print("正在初始化 PIKE-RAG 臨床知識檢索系統...")
    print("="*80)

    try:
        retriever = ClinicalKnowledgeRetriever(
            pdf_directory=CLINICAL_PDF_DIR,
            collection_name=RAG_COLLECTION_NAME,
            persist_directory=CHROMA_DB_DIR,
            embedding_model=RAG_EMBEDDING_MODEL,
            device="cuda" if __import__('torch').cuda.is_available() else "cpu",
            chunk_size=1024,
            chunk_overlap=256
        )

        # 建立或載入向量存儲
        retriever.build_vector_store(force_rebuild=RAG_FORCE_REBUILD)

        print("✓ PIKE-RAG 系統初始化完成")
        print("="*80 + "\n")

        return retriever

    except FileNotFoundError as e:
        print(f"✗ 錯誤: {e}")
        print("\n解決方案:")
        print("  1. 創建 PDF 目錄: mkdir -p clinical_pdfs")
        print("  2. 將臨床診斷 PDF 文件放入 clinical_pdfs/ 目錄")
        print("  3. 重新運行程式")
        print("="*80 + "\n")
        return None

    except Exception as e:
        print(f"✗ RAG 系統初始化失敗: {e}")
        print("將繼續運行但不使用 RAG 增強功能")
        print("="*80 + "\n")
        return None
