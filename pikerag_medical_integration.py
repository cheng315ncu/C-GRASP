#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PIKE-RAG 醫療知識檢索模組
用於從臨床診斷 PDF 中檢索相關知識，增強 HRV 情緒分析能力
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Dict
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


class ClinicalKnowledgeRetriever:
    """
    臨床知識檢索器
    負責載入臨床診斷 PDF，建立向量存儲，並根據查詢檢索相關知識
    """
    
    def __init__(
        self,
        pdf_directory: str,
        collection_name: str = "clinical_knowledge_base",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        初始化臨床知識檢索器
        
        參數:
            pdf_directory: PDF 文件所在目錄
            collection_name: 向量存儲集合名稱
            persist_directory: 向量存儲持久化目錄
            embedding_model: 嵌入模型名稱
            device: 運算設備 (cuda/cpu)
            chunk_size: 文本分塊大小
            chunk_overlap: 分塊重疊大小
        """
        self.pdf_directory = pdf_directory
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model  # 保存嵌入模型名稱
        
        print(f"正在初始化嵌入模型: {embedding_model} (設備: {device})")
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 32}
        )
        
        self.vector_store: Optional[Chroma] = None
        self.documents: List[Document] = []
        
    def load_clinical_documents(self) -> List[Document]:
        """
        載入臨床 PDF 文件並分塊
        
        返回:
            文件列表
        """
        print(f"\n[步驟 1] 載入臨床 PDF 文件從: {self.pdf_directory}")
        
        if not os.path.exists(self.pdf_directory):
            raise FileNotFoundError(
                f"PDF 目錄不存在: {self.pdf_directory}\n"
                f"請創建目錄並放入臨床診斷 PDF 文件:\n"
                f"  mkdir -p {self.pdf_directory}"
            )
        
        # 載入所有 PDF 文件
        pdf_files = list(Path(self.pdf_directory).glob("**/*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(
                f"在 {self.pdf_directory} 中沒有找到 PDF 文件\n"
                f"請放入臨床診斷相關的 PDF 文件"
            )
        
        print(f"找到 {len(pdf_files)} 個 PDF 文件")
        
        documents = []
        for pdf_file in pdf_files:
            print(f"  正在載入: {pdf_file.name}")
            try:
                loader = PyPDFLoader(str(pdf_file))
                docs = loader.load()
                
                # 添加來源元數據
                for doc in docs:
                    doc.metadata['source_file'] = pdf_file.name
                    doc.metadata['file_path'] = str(pdf_file)
                    _merge_prior_metadata(doc.metadata, pdf_file.name)
                
                documents.extend(docs)
                print(f"    ✓ 載入 {len(docs)} 頁")
            except Exception as e:
                print(f"    ✗ 載入失敗: {e}")
                continue
        
        print(f"\n總共載入 {len(documents)} 個文件頁面")
        
        # 文本分塊
        print(f"\n[步驟 2] 文本分塊 (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )
        
        split_documents = text_splitter.split_documents(documents)
        print(f"  ✓ 分塊完成，共 {len(split_documents)} 個文本塊")
        for chunk in split_documents:
            metrics = _extract_hrv_metrics(chunk.page_content)
            if metrics:
                chunk.metadata['hrv_metrics'] = metrics
            # 確保優先資訊在分塊後仍存在
            source_file = chunk.metadata.get('source_file')
            if source_file:
                _merge_prior_metadata(chunk.metadata, source_file)
        
        self.documents = split_documents
        return split_documents
    
    def build_vector_store(self, force_rebuild: bool = False) -> Chroma:
        """
        建立或載入向量存儲
        
        參數:
            force_rebuild: 是否強制重建向量存儲
            
        返回:
            Chroma 向量存儲實例
        """
        import shutil
        
        # 如果強制重建，先徹底刪除整個持久化目錄
        if force_rebuild and os.path.exists(self.persist_directory):
            print(f"\n[步驟 3] 強制重建模式：刪除舊向量存儲目錄")
            print(f"  正在刪除: {self.persist_directory}")
            try:
                shutil.rmtree(self.persist_directory)
                print(f"  ✓ 舊向量存儲已刪除")
            except Exception as e:
                print(f"  ⚠ 刪除時出現警告: {e}")
                # 嘗試刪除集合（如果 ChromaDB 已初始化）
                try:
                    import chromadb
                    client = chromadb.PersistentClient(path=self.persist_directory)
                    try:
                        client.delete_collection(name=self.collection_name)
                        print(f"  ✓ 已刪除集合: {self.collection_name}")
                    except Exception:
                        pass  # 集合可能不存在
                except Exception:
                    pass  # ChromaDB 可能未初始化
        
        # 檢查是否已存在向量存儲（在非強制重建模式下）
        if not force_rebuild and os.path.exists(self.persist_directory):
            print(f"\n[步驟 3] 嘗試載入現有向量存儲: {self.persist_directory}")
            
            # 先檢查嵌入模型是否匹配
            try:
                import chromadb
                import json
                client = chromadb.PersistentClient(path=self.persist_directory)
                
                # 檢查集合是否存在
                try:
                    existing_collection = client.get_collection(name=self.collection_name)
                    
                    # 檢查嵌入模型名稱（從 metadata 中讀取）
                    stored_model = existing_collection.metadata.get('embedding_model')
                    if stored_model and stored_model != self.embedding_model_name:
                        print(f"  ⚠ 嵌入模型已改變: 現有={stored_model}, 當前={self.embedding_model_name}")
                        print(f"  ⚠ 需要重建向量存儲以匹配新模型")
                        raise ValueError(f"嵌入模型不匹配: {stored_model} vs {self.embedding_model_name}")
                    
                    # 檢查嵌入維度是否匹配
                    try:
                        # 獲取當前嵌入模型的維度
                        test_embedding = self.embedding_function.embed_query("test")
                        current_embedding_dim = len(test_embedding)
                        
                        # 從現有集合獲取樣本向量來檢查維度
                        sample_result = existing_collection.peek(limit=1)
                        if sample_result['embeddings'] and len(sample_result['embeddings']) > 0:
                            existing_dim = len(sample_result['embeddings'][0])
                            if existing_dim != current_embedding_dim:
                                print(f"  ⚠ 嵌入維度不匹配: 現有={existing_dim}, 當前={current_embedding_dim}")
                                print(f"  ⚠ 需要重建向量存儲")
                                raise ValueError(f"嵌入維度不匹配: {existing_dim} vs {current_embedding_dim}")
                    except Exception as dim_e:
                        # 如果維度檢查失敗，但模型名稱匹配，繼續嘗試載入
                        if "嵌入維度不匹配" in str(dim_e):
                            raise
                        pass
                        
                except ValueError:
                    # 模型不匹配，需要重建
                    raise
                except Exception as e:
                    # 集合不存在或其他錯誤，繼續嘗試載入
                    pass
                    
            except ValueError:
                # 嵌入模型或維度不匹配，需要重建
                if os.path.exists(self.persist_directory):
                    try:
                        shutil.rmtree(self.persist_directory)
                        print(f"  ✓ 已刪除舊向量存儲目錄（嵌入模型已改變）")
                    except Exception as e2:
                        print(f"  ⚠ 刪除目錄時出現警告: {e2}")
            except Exception:
                # ChromaDB 檢查失敗，繼續嘗試載入
                pass
            
            # 嘗試載入現有向量存儲
            try:
                self.vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_function,
                    persist_directory=self.persist_directory
                )
                count = self.vector_store._collection.count()
                print(f"  ✓ 向量存儲載入成功，包含 {count} 個向量")
                return self.vector_store
            except Exception as e:
                error_msg = str(e)
                # 檢查是否是嵌入維度不匹配的錯誤
                if "dimension" in error_msg.lower() or "embedding" in error_msg.lower():
                    print(f"  ⚠ 載入失敗（嵌入維度不匹配）: {error_msg}")
                    print(f"  ℹ 這通常是因為嵌入模型已改變")
                    print(f"  ℹ 將刪除舊向量存儲並重建...")
                else:
                    print(f"  ✗ 載入失敗: {error_msg}")
                    print("  將重新建立向量存儲...")
                
                # 刪除有問題的目錄並重建
                if os.path.exists(self.persist_directory):
                    try:
                        shutil.rmtree(self.persist_directory)
                        print(f"  ✓ 已刪除舊向量存儲目錄")
                    except Exception as e2:
                        print(f"  ⚠ 刪除目錄時出現警告: {e2}")
        
        # 建立新的向量存儲
        if not self.documents:
            self.load_clinical_documents()
        
        print(f"\n[步驟 3] 建立向量存儲: {self.persist_directory}")
        
        print(f"  正在生成 {len(self.documents)} 個文本塊的嵌入向量...")
        print(f"  (這可能需要幾分鐘，請耐心等待...)")
        
        # 為每個文檔生成唯一 ID，確保去重
        import hashlib
        doc_ids = []
        for i, doc in enumerate(self.documents):
            # 使用內容哈希 + 索引確保唯一性
            content_hash = hashlib.md5(doc.page_content.encode('utf-8')).hexdigest()[:8]
            source = doc.metadata.get('source_file', 'unknown')
            doc_id = f"{source}_{i}_{content_hash}"
            doc_ids.append(doc_id)
        
        print(f"  生成了 {len(doc_ids)} 個唯一文檔 ID")
        
        self.vector_store = Chroma.from_documents(
            documents=[
                Document(
                    page_content=doc.page_content,
                    metadata=_serialize_metadata(doc.metadata or {})
                )
                for doc in self.documents
            ],
            embedding=self.embedding_function,
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            ids=doc_ids
        )
        
        # 在集合的 metadata 中保存嵌入模型名稱，以便後續檢查
        try:
            self.vector_store._collection.modify(
                metadata={"embedding_model": self.embedding_model_name}
            )
        except Exception:
            # 如果無法設置 metadata，不影響主要功能
            pass
        
        print(f"  ✓ 向量存儲建立完成，包含 {self.vector_store._collection.count()} 個向量")
        return self.vector_store
    
    def retrieve_relevant_knowledge(
        self,
        query: str,
        k: int = 3,
        score_threshold: float = 0.3,
        verbose: bool = False
    ) -> List[Dict[str, any]]:
        """
        根據查詢檢索相關的臨床知識
        
        參數:
            query: 查詢字串
            k: 返回前 k 個最相關的文件
            score_threshold: 相關性分數閾值
            verbose: 是否輸出詳細調試信息
            
        返回:
            包含文件內容和元數據的字典列表
        """
        if self.vector_store is None:
            raise RuntimeError("向量存儲未初始化，請先調用 build_vector_store()")
        
        if verbose:
            print(f"\n[RAG 檢索] 查詢: {query[:100]}...")
            print(f"[RAG 檢索] 參數: k={k}, score_threshold={score_threshold}")
        
        # 使用相似度搜索
        results = self.vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=k
        )
        
        if verbose:
            print(f"[RAG 檢索] 原始結果數: {len(results)}")
            if results:
                print(f"[RAG 檢索] 分數範圍: {min(s for _, s in results):.3f} - {max(s for _, s in results):.3f}")

        query_topics = _detect_query_topics(query)
        if verbose and query_topics:
            print(f"[RAG 檢索] 偵測查詢主題: {', '.join(query_topics)}")

        enriched_results = []
        for doc, raw_score in results:
            metadata = doc.metadata or {}
            domain_weight = _compute_domain_weight(metadata, query_topics)
            adjusted_score = raw_score * domain_weight
            enriched_results.append((doc, raw_score, adjusted_score, domain_weight))

        if verbose and enriched_results:
            adjusted_scores = [adj for _, _, adj, _ in enriched_results]
            print(f"[RAG 檢索] 領域加權後分數範圍: {min(adjusted_scores):.3f} - {max(adjusted_scores):.3f}")

        # 過濾低分結果
        filtered_results = [
            {
                'content': doc.page_content,
                'metadata': {
                    **dict(doc.metadata or {}),
                    'raw_score': raw_score,
                    'domain_weight': domain_weight,
                    'adjusted_score': adjusted_score
                },
                'score': adjusted_score
            }
            for doc, raw_score, adjusted_score, domain_weight in enriched_results
            if adjusted_score >= score_threshold
        ]
        
        if verbose:
            print(f"[RAG 檢索] 過濾後結果數: {len(filtered_results)}")
        
        # 去重：移除內容完全相同的結果（保留最高分的）
        seen_contents = set()
        deduplicated_results = []
        for result in filtered_results:
            # 使用內容的前 200 字符作為去重鍵（避免完整內容太長）
            content_key = result['content'][:200]
            if content_key not in seen_contents:
                seen_contents.add(content_key)
                deduplicated_results.append(result)
        
        if verbose and len(deduplicated_results) < len(filtered_results):
            print(f"[RAG 檢索] 去重後結果數: {len(deduplicated_results)} (移除了 {len(filtered_results) - len(deduplicated_results)} 個重複)")
        
        deduplicated_results = filter_absolute_threshold_dominant(deduplicated_results)
        return deduplicated_results
    
    def format_retrieved_knowledge(self, docs, max_length=2000):
        """
        格式化檢索到的知識，並包含來源，以便注入 Prompt
        
        參數:
            docs: retrieve_relevant_knowledge 返回的字典列表
            max_length: 最大文本長度
        """
        if not docs:
            return "（未檢索到相關臨床知識）"
        
        # ⚠️ CRITICAL: 添加 Z-score 優先警告
        zscore_warning = """⚠️ CRITICAL REMINDER: When per-subject Z-scores are available in the input data, 
they MUST take absolute precedence over any population norms, thresholds, or guidelines mentioned 
in the literature below. The literature provides general methodological context and evidence-based 
guidelines, but your analysis MUST be driven by within-subject baseline deviations (Z-scores) 
rather than population-based absolute thresholds.

**Decision Hierarchy (STRICT ORDER - cannot be overridden by literature):**
1. **Z-scores (PRIMARY)** - Individual baseline deviations (e.g., RMSSD_ms_zscore, SampEn_zscore)
2. Complexity metrics (SampEn_zscore, DFA_alpha_zscore) 
3. Absolute anchors (ONLY if Z-scores unavailable)
4. Literature norms (CONTEXTUAL SUPPORT ONLY, never primary evidence)

If the literature suggests a different interpretation than what the Z-scores indicate, the Z-scores 
are correct and the literature should be noted as providing general context but not directly 
applicable to this individual case.

---
"""
        
        knowledge_parts = []
        for i, doc_dict in enumerate(docs):
            # 從字典中提取資訊
            content = doc_dict['content']
            metadata = doc_dict['metadata']
            score = doc_dict['score']
            raw_score = metadata.get('raw_score')
            domain_weight = metadata.get('domain_weight')
            
            # 優先使用 source_file（自定義），回退到 source（PyPDFLoader 默認）
            source = metadata.get('source_file', metadata.get('source', '未知來源'))
            # 從完整路徑中提取檔案名稱，使其更簡潔
            source_filename = os.path.basename(source) if source != '未知來源' else source
            # 清理內容中的換行符，使其更緊湊
            content_clean = content.replace('\n', ' ').strip()
            # 相關性描述
            score_desc = f"{score:.2f}"
            if raw_score is not None and domain_weight is not None:
                score_desc += f" (原始 {raw_score:.2f} × 加權 {domain_weight:.2f})"

            details = []
            topics = _get_metadata_list(metadata, 'hrv_topics')
            if topics:
                details.append(f"主題：{', '.join(topics)}")
            metrics = _get_metadata_list(metadata, 'hrv_metrics')
            if metrics:
                details.append(f"指標：{', '.join(metrics)}")
            key_points = _get_metadata_list(metadata, 'key_points')
            if key_points:
                details.append("關鍵發現：" + "； ".join(key_points[:2]))

            segment = [f"[來源 {i+1} ({source_filename}), 相關性: {score_desc}]"]
            if details:
                segment.extend(f"  - {detail}" for detail in details)
            segment.append(f"  - 摘要：{content_clean}")
            knowledge_parts.append("\n".join(segment))

        knowledge = "\n\n".join(knowledge_parts)
        
        if len(knowledge) > max_length:
            knowledge = knowledge[:max_length] + "\n... (內容已截斷)"
            
        return zscore_warning + f"以下是從臨床知識庫中檢索到的相關資訊：\n\n{knowledge}"


# ============================================================================
# HRV 測量證據權重 (基於方法學文獻的可靠性)
# ============================================================================
HRV_EVIDENCE_WEIGHTS = {
    "LF_HF": 0.3,      # 降權：Billman 2013 證明不可靠
    "RMSSD": 0.9,      # 高權：公認可靠的副交感指標
    "SDNN": 0.7,       # 中權：總體變異性，需 HR 校正
    "SampEn": 0.6,     # 中權：高度依賴參數選擇
    "DFA_alpha": 0.5,  # 中低權：非自主神經直接指標
    "SD1_SD2": 0.4,    # 低權：幾何測量，非複雜度
}

# ============================================================================
# 針對性文獻檢索關鍵詞 (依缺陷類型分層)
# ============================================================================
CRITICAL_LITERATURE_QUERIES = {
    "lf_hf_critique": [
        "Billman LF/HF does not measure sympathovagal balance",
        "Heathers Everything Hertz methodological issues frequency domain",
        "respiratory artifacts LF HF power spectral analysis"
    ],
    
    "dfa_interpretation": [
        "DFA alpha heart rate variability long-range correlation NOT autonomic",
        "detrended fluctuation analysis CHF patients paradox sympathetic",
        "fractal scaling exponent physiological meaning exercise intensity"
    ],
    
    "poincare_complexity": [
        "SD1 SD2 ratio geometric variability NOT sample entropy complexity",
        "Poincaré plot short-term long-term HRV nonlinear dynamics",
        "complex correlation measure CCM lag-response curvilinearity"
    ],
    
    "threshold_calibration": [
        "HRV normative values age sex heart rate correction Gasior",
        "RMSSD SDNN threshold individual baseline task-dependent",
        "heart rate variability reference ranges population-specific"
    ],
    
    "entropy_parameters": [
        "sample entropy parameter selection m r tolerance Mayer",
        "ApEn SampEn DistEn comparison short data length",
        "entropy measure sensitivity data length embedding dimension"
    ],
    
    "coactivation_states": [
        "sympathetic parasympathetic coactivation Eickholt atrial fibrillation",
        "autonomic nervous system simultaneous activation Berg",
        "mixed autonomic states continuous distribution probabilistic"
    ],
    
    "inference_limitations": [
        "HRV cognitive function systematic review Forte",
        "emotion recognition physiological signals overinterpretation",
        "autonomic arousal learning engagement behavioral validation multimodal"
    ]
}


def _safe_float(value):
    """將輸入轉為浮點數，失敗時返回 None。"""
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def classify_relative_change(
    z_value,
    raw_value,
    high_cutoff: Optional[float] = None,
    low_cutoff: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """
    以 Z-score 為優先，評估指標相對個體基準的變化方向與幅度。
    若無 Z-score，才回退至絕對閾值比較。
    """
    z = _safe_float(z_value)
    raw = _safe_float(raw_value)
    info = {
        "direction": "baseline",
        "magnitude": "baseline",
        "source": None,
        "z_value": z,
        "raw_value": raw,
    }

    if z is not None:
        info["source"] = "zscore"
        abs_z = abs(z)
        if abs_z < 0.5:
            return info
        info["direction"] = "high" if z > 0 else "low"
        if abs_z >= 2.0:
            info["magnitude"] = "marked"
        elif abs_z >= 1.0:
            info["magnitude"] = "moderate"
        else:
            info["magnitude"] = "mild"
        return info

    info["source"] = "absolute"
    if raw is None:
        return info
    if high_cutoff is not None and raw >= high_cutoff:
        info["direction"] = "high"
        info["magnitude"] = "contextual"
    elif low_cutoff is not None and raw <= low_cutoff:
        info["direction"] = "low"
        info["magnitude"] = "contextual"
    return info


def describe_relative_state(
    info: Dict[str, Optional[float]],
    elevated_template: str,
    reduced_template: str,
    neutral_template: str = "near baseline",
    fallback_template: str = "context-dependent",
) -> str:
    """依據 classify_relative_change 的結果生成文字描述。"""
    direction = info.get("direction", "baseline")
    magnitude = info.get("magnitude", "baseline")
    prefix_map = {
        "marked": "顯著",
        "moderate": "中等程度",
        "mild": "輕度",
        "contextual": "相對",
    }
    prefix = prefix_map.get(magnitude, "")

    if direction == "high":
        return f"{prefix}{elevated_template}".strip()
    if direction == "low":
        return f"{prefix}{reduced_template}".strip()
    if direction == "baseline":
        return neutral_template
    return fallback_template


def relative_keyword(
    info: Dict[str, Optional[float]],
    elevated: str = "elevated",
    neutral: str = "baseline",
    reduced: str = "reduced",
) -> str:
    """提供用於查詢的關鍵字描述。"""
    direction = info.get("direction", "baseline")
    if direction == "high":
        return elevated
    if direction == "low":
        return reduced
    return neutral


def get_conditional_warning_queries(
    raw_metrics: Dict,
    zscore_metrics: Optional[Dict] = None,
) -> List[str]:
    """
    根據 HRV 指標矛盾情況生成條件性警示查詢
    
    參數:
        raw_metrics: 包含 HRV 原始指標的字典
        zscore_metrics: 包含 HRV Z-score 指標的字典
        
    返回:
        警示查詢列表
    """
    warnings = []
    
    zscores = zscore_metrics or {}

    rmssd_info = classify_relative_change(
        zscores.get('RMSSD_ms_zscore'),
        raw_metrics.get('RMSSD_ms'),
        high_cutoff=40,
        low_cutoff=25,
    )
    lfhf_info = classify_relative_change(
        zscores.get('LF_HF_zscore'),
        raw_metrics.get('LF_HF'),
        high_cutoff=2.0,
        low_cutoff=0.7,
    )
    hr_info = classify_relative_change(
        zscores.get('MeanHR_bpm_zscore'),
        raw_metrics.get('MeanHR_bpm'),
        high_cutoff=85,
        low_cutoff=60,
    )
    sampen_info = classify_relative_change(
        zscores.get('SampEn_zscore'),
        raw_metrics.get('SampEn'),
        high_cutoff=1.5,
        low_cutoff=1.0,
    )
    dfa_key = 'DFA_alpha1' if 'DFA_alpha1' in raw_metrics else 'DFA_alpha'
    dfa_info = classify_relative_change(
        zscores.get(f'{dfa_key}_zscore'),
        raw_metrics.get(dfa_key),
        high_cutoff=1.1,
        low_cutoff=0.7,
    )
    sd1 = _safe_float(raw_metrics.get('SD1_ms'))
    sd2 = _safe_float(raw_metrics.get('SD2_ms'))
    sd1_sd2_raw = _safe_float(raw_metrics.get('SD1_SD2'))
    if sd1_sd2_raw is None and sd1 is not None and sd2 not in (None, 0):
        sd1_sd2_raw = sd1 / sd2
    sd1_sd2_info = classify_relative_change(
        zscores.get('SD1_SD2_zscore'),
        sd1_sd2_raw,
        high_cutoff=1.0,
        low_cutoff=0.6,
    )
    
    # 檢測 1: 迷走張力相對提升但 LF/HF 亦偏高 → 可能雙重激活
    if rmssd_info['direction'] == 'high' and lfhf_info['direction'] == 'high':
        warnings.append(
            "sympathetic parasympathetic coactivation mechanism respiratory confound artifact"
        )
    
    # 檢測 2: 心率升高、複雜度下降，但 LF/HF 顯示副交感主導 → 可能呼吸或方法學誤差
    if (
        hr_info['direction'] == 'high'
        and sampen_info['direction'] == 'low'
        and lfhf_info['direction'] == 'low'
    ):
        warnings.append(
            "high heart rate low complexity parasympathetic dominance contradiction LF/HF unreliable"
        )
    
    # 檢測 3: 長期相關性升高但短期變異低 → 可能代表不同時間尺度矛盾
    if dfa_info['direction'] == 'high' and rmssd_info['direction'] == 'low':
        warnings.append(
            "DFA alpha interpretation long-range correlation NOT parasympathetic direct measure"
        )
    
    # 檢測 4: 幾何比與熵同時降低 → 可能資料品質或參數設定問題
    if sd1_sd2_info['direction'] == 'low' and sampen_info['direction'] == 'low':
        warnings.append(
            "SD1/SD2 sample entropy contradiction geometric vs complexity measures different constructs"
        )
    
    # 檢測 5: LF/HF 極端偏移 → 可能受呼吸或訊號處理影響
    if lfhf_info['source'] == 'absolute':
        raw_lfhf = _safe_float(raw_metrics.get('LF_HF'))
        if raw_lfhf is not None and (raw_lfhf > 3.0 or raw_lfhf < 0.3):
            warnings.append(
                "extreme LF/HF ratio respiratory rate breathing artifacts frequency domain HRV methodological"
            )
    elif lfhf_info['magnitude'] in {"marked", "moderate"}:
        warnings.append(
            "extreme LF/HF ratio respiratory rate breathing artifacts frequency domain HRV methodological"
        )
    
    return warnings


def _normalize_filename(path: str) -> str:
    """統一路徑名稱為小寫檔名。"""
    if not path:
        return ""
    return os.path.basename(path).lower()


def _merge_prior_metadata(metadata: Dict, pdf_name: str) -> None:
    """依照預先整理的臨床 PDF 資訊補齊元資料。"""
    prior = CLINICAL_PDF_PRIORS.get(pdf_name.lower())
    if not prior:
        return
    for key, value in prior.items():
        if isinstance(value, list):
            existing = metadata.get(key, [])
            # 保持原有順序但去除重複
            combined = list(dict.fromkeys([*existing, *value]))
            metadata[key] = combined
        else:
            metadata.setdefault(key, value)


def _serialize_metadata(metadata: Dict) -> Dict:
    """將 Document metadata 轉換為 Chroma 可接受的基本型態。"""
    serialized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            serialized[key] = METADATA_LIST_DELIMITER.join(str(v) for v in value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            serialized[key] = value
        else:
            serialized[key] = str(value)
    return serialized


def _get_metadata_list(metadata: Dict, key: str) -> List[str]:
    """從元資料中擷取列表型欄位，支援字串分隔還原。"""
    value = metadata.get(key)
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [item.strip() for item in value.split(METADATA_LIST_DELIMITER) if item.strip()]
    return []


def _extract_hrv_metrics(text: str) -> List[str]:
    """從文本內容偵測常見 HRV 指標名稱。"""
    if not text:
        return []
    detected = []
    for metric, pattern in HRV_METRIC_PATTERNS.items():
        if re.search(pattern, text, flags=re.IGNORECASE):
            detected.append(metric)
    return sorted(set(detected))


def _detect_query_topics(query: str) -> List[str]:
    """將查詢字串映射到預定義的 HRV 主題標籤。"""
    if not query:
        return []
    query_lower = query.lower()
    topics = []
    for topic, keywords in TOPIC_KEYWORD_MAP.items():
        if any(keyword in query_lower for keyword in keywords):
            topics.append(topic)
    return topics


def _compute_domain_weight(metadata: Dict, query_topics: List[str]) -> float:
    """根據文獻優先級與查詢主題計算領域加權係數。"""
    weight = float(metadata.get("evidence_weight", 1.0) or 1.0)

    doc_topics = set(_get_metadata_list(metadata, "hrv_topics"))
    if doc_topics and query_topics:
        overlap = doc_topics.intersection(query_topics)
        if overlap:
            weight *= 1.05 + 0.05 * min(len(overlap), 3)

    metrics = _get_metadata_list(metadata, "hrv_metrics")
    metric_weights = [
        HRV_EVIDENCE_WEIGHTS[m]
        for m in metrics
        if m in HRV_EVIDENCE_WEIGHTS
    ]
    if metric_weights:
        max_metric_weight = max(metric_weights)
        weight *= 1 + (max_metric_weight - 0.5) * 0.4

    study_design = metadata.get("study_design", "")
    if study_design in {"clinical_case_control", "controlled_trial", "randomized_intervention"}:
        weight *= 1.08
    elif study_design in {"population_reference", "clinical_observational"}:
        weight *= 1.05
    elif study_design.startswith("opinion"):
        weight *= 0.97

    return max(weight, 0.1)


def create_hrv_analysis_query(
    sample_data: Dict, 
    use_critical_approach: bool = True,
    verbosity: str = "balanced"  # "minimal", "balanced", "comprehensive"
) -> str:
    """
    根據 HRV 樣本數據創建檢索查詢
    
    參數:
        sample_data: HRV 樣本數據字典
        use_critical_approach: 是否使用批判性方法學視角 (預設 True)
        verbosity: 查詢詳細程度
            - "minimal": 精簡查詢 (適合 threshold > 0.5)
            - "balanced": 平衡查詢 (適合 threshold 0.3-0.5) **推薦**
            - "comprehensive": 完整查詢 (適合 threshold < 0.3)
        
    返回:
        檢索查詢字串
    
    策略 (批判性方法學視角):
        - 優先檢索方法學限制與爭議文獻
        - 使用校正後的指標解讀方式
        - 納入測量不確定性與混合狀態
        - 避免過度推論生理到情緒的因果關係
    """
    raw_features = sample_data.get('raw_features', {}) or {}
    zscore_features = sample_data.get('zscore_features', {}) or {}
    has_zscore = any(_safe_float(v) is not None for v in zscore_features.values())

    rmssd_info = classify_relative_change(
        zscore_features.get('RMSSD_ms_zscore'),
        raw_features.get('RMSSD_ms'),
        high_cutoff=40,
        low_cutoff=25,
    )
    sampen_info = classify_relative_change(
        zscore_features.get('SampEn_zscore'),
        raw_features.get('SampEn'),
        high_cutoff=1.5,
        low_cutoff=1.0,
    )
    sdnn_info = classify_relative_change(
        zscore_features.get('SDNN_ms_zscore'),
        raw_features.get('SDNN_ms'),
        high_cutoff=60,
        low_cutoff=40,
    )
    hr_info = classify_relative_change(
        zscore_features.get('MeanHR_bpm_zscore'),
        raw_features.get('MeanHR_bpm'),
        high_cutoff=85,
        low_cutoff=60,
    )
    lfhf_info = classify_relative_change(
        zscore_features.get('LF_HF_zscore'),
        raw_features.get('LF_HF'),
        high_cutoff=2.0,
        low_cutoff=0.7,
    )
    dfa_key = 'DFA_alpha1' if 'DFA_alpha1' in raw_features else 'DFA_alpha'
    dfa_info = classify_relative_change(
        zscore_features.get(f'{dfa_key}_zscore'),
        raw_features.get(dfa_key),
        high_cutoff=1.1,
        low_cutoff=0.7,
    )

    vagal_phrase = describe_relative_state(
        rmssd_info,
        "迷走張力相對基準偏高",
        "迷走張力相對基準偏低",
        "迷走張力接近個體基準",
    )
    sdnn_phrase = describe_relative_state(
        sdnn_info,
        "自主神經總體變異度高於基準",
        "自主神經總體變異度低於基準",
        "自主神經變異度接近基準",
    )
    complexity_phrase = describe_relative_state(
        sampen_info,
        "HRV 非線性複雜度上升",
        "HRV 非線性複雜度下降",
        "HRV 複雜度接近基準",
    )
    arousal_phrase = describe_relative_state(
        hr_info,
        "心搏速率／喚醒度高於個體基準",
        "心搏速率／喚醒度低於個體基準",
        "心搏速率接近基準",
    )
    ans_balance_phrase = describe_relative_state(
        lfhf_info,
        "交感調節相對偏高",
        "副交感調節相對偏高",
        "自律神經平衡接近基準",
        "自律神經樣貌需視情境判讀",
    )

    vagal_keyword = relative_keyword(rmssd_info, "elevated", "baseline", "reduced")
    complexity_keyword = relative_keyword(sampen_info, "increased", "baseline", "reduced")
    arousal_keyword = relative_keyword(hr_info, "heightened", "baseline", "lowered")
    ans_keyword = relative_keyword(lfhf_info, "sympathetic-leaning", "balanced", "parasympathetic-leaning")
    
    if use_critical_approach:
        # === 基於方法學批判文獻的查詢構建 ===
        
        if verbosity == "minimal":
            # 精簡版：只保留最核心的 3-4 個查詢點
            query_parts = [
                f"heart rate variability RMSSD parasympathetic {vagal_keyword} individualized baseline adjustment",
                f"SDNN autonomic flexibility {relative_keyword(sdnn_info, 'elevated', 'baseline', 'reduced')} participant-specific",
                f"sample entropy complexity {complexity_keyword} window length sensitivity",
                "LF/HF ratio limitations respiratory artifacts"
            ]
            
        elif verbosity == "balanced":
            # 平衡版：涵蓋核心批判點，但移除作者名和過於專業的術語
            query_parts = [
                # 核心指標解讀
                f"RMSSD parasympathetic vagal tone {vagal_phrase} heart rate corrected interpretation",
                f"SDNN autonomic regulation {sdnn_phrase} individualized baseline calibration",
                f"sample entropy parameters {complexity_phrase} 資料長度敏感度",
                
                # 方法學限制
                "LF/HF ratio reliability limitations sympathovagal balance respiratory artifacts",
                "HRV normative values individualized baseline differences age sex heart rate",
                
                # 混合狀態與推論限制
                f"sympathetic parasympathetic coactivation {ans_balance_phrase}",
                "HRV emotion recognition limitations behavioral validation"
            ]
            
            # 添加條件性警示（僅高優先級）
            warnings = get_conditional_warning_queries(raw_features, zscore_features)
            if warnings:
                query_parts.append(warnings[0])  # 只加第一個最重要的警示
                
        else:  # comprehensive
            # 完整版：保留所有層次的查詢（原始版本，但移除作者名）
            query_parts = [
                # 第一層：核心方法學批判
                "LF/HF ratio reliability limitations sympathovagal balance critique respiratory artifacts",
                "sample entropy parameters m r tolerance data length sensitivity methodological",
                
                # 第二層：替代性指標與校正方法
                f"heart rate corrected HRV normative values individualized {arousal_keyword} arousal",
                f"RMSSD parasympathetic vagal tone HR-corrected {vagal_keyword} short-term",
                f"SDNN autonomic regulation individualized baseline total variability {relative_keyword(sdnn_info, 'elevated', 'baseline', 'reduced')}",
                
                # 第三層：DFA 與非線性測度的正確解讀
                f"DFA alpha long-range correlation scaling exponent {relative_keyword(dfa_info, 'elevated', 'baseline', 'reduced')} paradox",
                "SD1 SD2 ratio Poincaré short-term long-term variability geometric measure",
                f"sample entropy ApEn comparison conflicting results parameter sensitivity {complexity_keyword}",
                
                # 第四層：自主神經混合狀態
                f"sympathetic parasympathetic coactivation simultaneous activation {ans_keyword}",
                "autonomic nervous system mixed states transition uncertainty quantification",
                
                # 第五層：生理到情緒推論的限制
                "HRV emotion recognition overinterpretation causal inference limitations multimodal validation",
                f"physiological arousal cognitive state engagement attention {arousal_keyword}",
                
                # 第六層：任務與條件依賴性
                "HRV task-dependent baseline recording conditions posture respiration standardization"
            ]
            
            # 添加所有條件性警示
            conditional_warnings = get_conditional_warning_queries(raw_features, zscore_features)
            query_parts.extend(conditional_warnings)
        
    else:
        # === 傳統簡單查詢（向後兼容） ===
        query_parts = [
            "heart rate variability HRV clinical interpretation",
            f"RMSSD parasympathetic activity {vagal_phrase}",
            f"sample entropy complexity {complexity_phrase}",
            f"SDNN autonomic regulation {sdnn_phrase}",
            f"heart rate {arousal_phrase}",
            f"LF HF ratio {ans_balance_phrase}",
            "emotion affective state psychophysiology",
            "vagal tone cardiac autonomic nervous system"
        ]
    
    if has_zscore:
        baseline_prefix = (
            "individualized baseline within-subject Z-score interpretation priority over population norms "
            "personalized HRV assessment individual calibration "
        )
    else:
        baseline_prefix = ""
    query = baseline_prefix + " ".join(query_parts)
    return query


def build_layered_rag_queries(sample_data: Dict) -> Dict[str, str]:
    """
    構建分層的 RAG 查詢，用於多輪檢索或加權整合
    
    參數:
        sample_data: HRV 樣本數據字典
        
    返回:
        包含不同層次查詢的字典
    """
    raw_features = sample_data.get('raw_features', {}) or {}
    zscore_features = sample_data.get('zscore_features', {}) or {}

    rmssd_info = classify_relative_change(
        zscore_features.get('RMSSD_ms_zscore'),
        raw_features.get('RMSSD_ms'),
        high_cutoff=40,
        low_cutoff=25,
    )
    sampen_info = classify_relative_change(
        zscore_features.get('SampEn_zscore'),
        raw_features.get('SampEn'),
        high_cutoff=1.5,
        low_cutoff=1.0,
    )
    hr_info = classify_relative_change(
        zscore_features.get('MeanHR_bpm_zscore'),
        raw_features.get('MeanHR_bpm'),
        high_cutoff=85,
        low_cutoff=60,
    )
    lfhf_info = classify_relative_change(
        zscore_features.get('LF_HF_zscore'),
        raw_features.get('LF_HF'),
        high_cutoff=2.0,
        low_cutoff=0.7,
    )
    
    return {
        "methodological_critique": (
            "LF/HF sympathovagal balance unreliable Billman Heathers "
            "respiratory artifacts frequency domain limitations"
        ),
        
        "corrected_interpretation": (
            f"heart rate corrected RMSSD {relative_keyword(rmssd_info, 'elevated', 'baseline', 'reduced')} vagal tone normative baseline "
            f"SDNN total variability HR correction task-dependent"
        ),
        
        "nonlinear_measures": (
            f"DFA alpha long-range correlation NOT autonomic paradox "
            f"sample entropy m r parameters {relative_keyword(sampen_info, 'elevated', 'baseline', 'reduced')} data length sensitivity"
        ),
        
        "mixed_states": (
            f"sympathetic parasympathetic coactivation simultaneous {relative_keyword(lfhf_info, 'sympathetic', 'balanced', 'parasympathetic')} "
            "autonomic mixed states probabilistic uncertainty Bayesian"
        ),
        
        "inference_limits": (
            f"HRV emotion cognitive inference limitations behavioral validation "
            f"physiological arousal {relative_keyword(hr_info, 'heightened', 'baseline', 'lowered')} overinterpretation multimodal"
        ),
        
        "conditional_warnings": " ".join(get_conditional_warning_queries(raw_features, zscore_features))
    }


# ============================================================================
# 過度依賴絕對閾值的文獻片語與過濾
# ============================================================================
ABS_THRESHOLD_PHRASES = [
    "rmssd > 40 ms",
    "rmssd>=40",
    "high if greater than",
    "population norm is",
    "lf/hf ratio >",
    "lf/hf ratio <",
    "sdnn > 60 ms indicates"
]


def filter_absolute_threshold_dominant(docs, penalty=0.85):
    """降權過度依賴人口閾值的片段。"""
    filtered = []
    for entry in docs:
        doc_dict = dict(entry)
        metadata = dict(doc_dict.get('metadata', {}))
        content_lower = doc_dict.get('content', '').lower()
        has_threshold = any(phrase in content_lower for phrase in ABS_THRESHOLD_PHRASES)
        metadata['threshold_warning'] = has_threshold
        if has_threshold:
            doc_dict['score'] = (doc_dict.get('score') or 0.0) * penalty
            notes = metadata.get('notes', '')
            metadata['notes'] = (notes + " | threshold-heavy").strip(" |")
        doc_dict['metadata'] = metadata
        filtered.append(doc_dict)
    return filtered


# ============================================================================
# 已標註的臨床 PDF 優先級資訊 (依據文獻重點與研究設計)
# ============================================================================
CLINICAL_PDF_PRIORS = {
    "fphys-04-00026.pdf": {
        "hrv_topics": ["lfhf_methodology", "frequency_domain_limitations", "methodology_critique"],
        "key_points": [
            "LF/HF 比值無法精準衡量交感/副交感平衡，動態情境下失真顯著。",
            "呼吸速率與非線性耦合會嚴重干擾頻域指標的生理解讀。"
        ],
        "study_design": "opinion_methodology",
        "evidence_weight": 1.22
    },
    "fphys-05-00177.pdf": {
        "hrv_topics": ["frequency_domain_limitations", "signal_quality", "methodology_critique"],
        "key_points": [
            "短期頻域 HRV 受窗長、呼吸與輸入配置影響，需嚴格控制測量協議。",
            "建議在報告 LF/HF 前同步提供呼吸資訊與訊號品質檢查。"
        ],
        "study_design": "review_methodology",
        "evidence_weight": 1.18
    },
    "fphys-07-00356.pdf": {
        "hrv_topics": ["hr_correction", "respiratory_influence", "repeatability"],
        "key_points": [
            "平均心率與呼吸速率可大幅影響 HRV 指標重現性，需進行 HR 校正。",
            "HR 校正後 RMSSD、SDNN 等指標的組內一致性明顯提升。"
        ],
        "population": "adult_clinical",
        "study_design": "controlled_trial",
        "evidence_weight": 1.20
    },
    "fphys-09-01495.pdf": {
        "hrv_topics": ["normative_values", "hr_correction", "pediatric_reference"],
        "key_points": [
            "提供校正平均心率後的兒童 HRV 常模，避免誤判副交感不足。",
            "建議任何跨個案比較均納入 Mean HR 差異調整。"
        ],
        "population": "pediatric",
        "study_design": "population_reference",
        "evidence_weight": 1.17
    },
    "medscimonit-24-2164.pdf": {
        "hrv_topics": ["coactivation_clinical", "atrial_fibrillation", "mixed_states"],
        "key_points": [
            "房顫患者可呈現交感/副交感同時活化，導致 HRV 動態混亂。",
            "CoA 狀態需結合臨床節律與血流資料共同解讀。"
        ],
        "population": "atrial_fibrillation",
        "study_design": "clinical_observational",
        "evidence_weight": 1.12
    },
    "fneur-02-00071.pdf": {
        "hrv_topics": ["coactivation_experimental", "mixed_states", "hypertension_model"],
        "key_points": [
            "高血壓動物模型顯示交感/副交感同時釋放時，心血管反應非線性。",
            "解釋 HRV 變化時需考慮雙重激活導致的指標反轉。"
        ],
        "population": "animal_model",
        "study_design": "experimental",
        "evidence_weight": 1.08
    },
    "0000726.pdf": {
        "hrv_topics": ["dfa_local_scaling", "nonlinear_dynamics", "fractal_analysis"],
        "key_points": [
            "局部尺度指數剖面（Local Scaling Exponent Profile）可辨識無明顯尺度區間的 DFA 特性。",
            "針對窄窗口 HRV，局部尺度指數較全域 DFA 更具敏感度。"
        ],
        "study_design": "methodology_research",
        "evidence_weight": 1.10
    },
    "1471-2105-15-S6-S2.pdf": {
        "hrv_topics": ["entropy_parameters", "nonlinear_dynamics", "feature_selection"],
        "key_points": [
            "針對 SampEn、FuzzyEn 等指標提供 r、m、資料長度參數選擇建議。",
            "不同臨床情境下需依資料長度調整容許度與嵌入維度。"
        ],
        "study_design": "methodology_research",
        "evidence_weight": 1.16
    },
    "entropy-17-06270.pdf": {
        "hrv_topics": ["entropy_parameters", "clinical_population_chf", "nonlinear_dynamics"],
        "key_points": [
            "針對心衰患者辨識最佳 SampEn 與 FuzzyEn 參數組合，提高分類性能。",
            "同時評估資料長度與容許度參數，建議於短片段 HRV 謹慎設定 r。"
        ],
        "population": "heart_failure",
        "study_design": "clinical_case_control",
        "evidence_weight": 1.14
    },
    "medicina-55-00532.pdf": {
        "hrv_topics": ["geometric_indices", "exercise_intervention", "metabolic_syndrome"],
        "key_points": [
            "16 週間期化間歇訓練可顯著提升 SD1、SD2 與 RRtri 等幾何指標。",
            "SD1/SD2 比值未顯著改變，提示幾何衡量需搭配其他複雜度指標。"
        ],
        "population": "metabolic_syndrome",
        "study_design": "randomized_intervention",
        "evidence_weight": 1.11
    }
}


# 用於序列化列表型元資料的分隔符
METADATA_LIST_DELIMITER = "|||"


# ============================================================================
# HRV 量測指標偵測與查詢主題映射
# ============================================================================
HRV_METRIC_PATTERNS = {
    "RMSSD": r"RMSSD",
    "SDNN": r"SDNN",
    "SampEn": r"Samp\\s*En|Sample\\s+Entropy",
    "LF_HF": r"LF/?HF",
    "DFA_alpha": r"DFA(?:\\s*alpha)?|detrended fluctuation",
    "SD1_SD2": r"SD1\\s*/\\s*SD2|SD1\\s+and\\s+SD2|Poincar[eé]",
    "MeanHR": r"Mean\\s*HR|heart rate",
    "RRtri": r"RRtri|triangular index",
    "TINN": r"TINN",
    "FuzzyEn": r"Fuzzy\\s+Entropy",
    "ApEn": r"Approximate\\s+Entropy|ApEn"
}

TOPIC_KEYWORD_MAP = {
    "lfhf_methodology": ["lf/hf", "low frequency", "high frequency", "sympathovagal"],
    "frequency_domain_limitations": ["frequency-domain", "power spectral", "spectral"],
    "entropy_parameters": ["entropy", "sampen", "apen", "fuzzyen", "fuzzy measure"],
    "nonlinear_dynamics": ["fractal", "dfa", "scaling exponent", "nonlinear"],
    "coactivation_clinical": ["coactivation", "co-activation", "simultaneous activation"],
    "hr_correction": ["heart rate correction", "mean hr", "prevailing heart rate"],
    "normative_values": ["normative", "reference", "percentile", "school-aged", "children"],
    "geometric_indices": ["poincar", "sd1", "sd2", "rrtri", "geometric"],
    "exercise_intervention": ["training", "exercise", "aerobic", "intervention"],
    "mixed_states": ["mixed states", "dual activation"],
    "clinical_population_chf": ["congestive heart failure", "chf"],
    "metabolic_syndrome": ["metabolic syndrome", "mets"]
}

# 使用範例
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # 導入配置模組
    try:
        # 嘗試從 PIKE-RAG 目錄導入 config
        sys.path.insert(0, str(Path(__file__).parent))
        import config
    except ImportError:
        print("警告: 無法導入 config.py，使用預設配置")
        config = None
    
    print("="*80)
    print("PIKE-RAG 醫療知識檢索模組 - 建立向量存儲")
    print("="*80)
    
    # 從 config.py 讀取配置，如果不存在則使用預設值
    if config:
        pdf_directory = getattr(config, 'CLINICAL_PDF_DIR', './clinical_pdfs')
        persist_directory = getattr(config, 'CHROMA_DB_DIR', './chroma_db')
        embedding_model = getattr(config, 'RAG_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
        force_rebuild = getattr(config, 'RAG_FORCE_REBUILD', False)
        collection_name = getattr(config, 'RAG_COLLECTION_NAME', 'clinical_knowledge_base')
    else:
        pdf_directory = './clinical_pdfs'
        persist_directory = './chroma_db'
        embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
        force_rebuild = False
        collection_name = "clinical_knowledge_base"
    
    print(f"\n配置資訊:")
    print(f"  PDF 目錄: {pdf_directory}")
    print(f"  向量存儲目錄: {persist_directory}")
    print(f"  嵌入模型: {embedding_model}")
    print(f"  強制重建: {force_rebuild}")
    print(f"  集合名稱: {collection_name}")
    print()
    
    # 初始化檢索器（使用 config.py 中的 embedding model）
    retriever = ClinicalKnowledgeRetriever(
        pdf_directory=pdf_directory,
        collection_name=collection_name,
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # 載入文件並建立向量存儲
    print("\n開始建立向量存儲...")
    retriever.load_clinical_documents()
    retriever.build_vector_store(force_rebuild=force_rebuild)
    
    print("\n" + "="*80)
    print("向量存儲建立完成！")
    print("="*80)

