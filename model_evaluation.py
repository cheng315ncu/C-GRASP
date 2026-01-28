#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CGRASP - Model Evaluation Pipeline
===================================

Integrated evaluation pipeline for comparing multiple LLM model outputs.
Combines report parsing (from test2.py) and analysis/visualization (from ana.py).

Features:
- Parse model reports and extract predictions
- Compare against ground truth (Valence/Arousal)
- Compute Clinical Reasoning Consistency (CRC)
- Calculate F1 scores, WAD (Weighted Affective Distance)
- Generate publication-quality visualizations

Usage:
    python model_evaluation.py --help
    python model_evaluation.py --config models.yml
    python model_evaluation.py --model-dir ./reports --output ./analysis
"""

import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

# Optional imports
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns
    HAS_MATPLOTLIB = True
    HAS_SEABORN = True
except ImportError:
    HAS_MATPLOTLIB = False
    HAS_SEABORN = False
    plt = None
    matplotlib = None
    sns = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    path: str
    is_baseline: bool = False


@dataclass
class EvalConfig:
    """Configuration for the evaluation pipeline."""
    models: List[ModelConfig] = field(default_factory=list)
    output_dir: str = "./analysis"
    baseline_model: str = ""
    generate_plots: bool = True
    dpi: int = 300
    
    @classmethod
    def from_yaml(cls, path: str) -> "EvalConfig":
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required for YAML config files")
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        
        models = []
        baseline = data.get("baseline_model", "")
        for m in data.get("models", []):
            models.append(ModelConfig(
                name=m["name"],
                path=m["path"],
                is_baseline=(m["name"] == baseline)
            ))
        
        return cls(
            models=models,
            output_dir=data.get("output_dir", "./analysis"),
            baseline_model=baseline,
            generate_plots=data.get("generate_plots", True),
            dpi=data.get("dpi", 300),
        )
    
    @classmethod
    def from_args(cls, args) -> "EvalConfig":
        """Create configuration from CLI arguments."""
        if args.config and os.path.exists(args.config):
            return cls.from_yaml(args.config)
        
        # Build from individual arguments
        models = []
        if args.model_dirs:
            for i, path in enumerate(args.model_dirs):
                name = args.model_names[i] if args.model_names and i < len(args.model_names) else f"Model_{i+1}"
                models.append(ModelConfig(name=name, path=path))
        
        baseline = args.baseline if args.baseline else (models[0].name if models else "")
        for m in models:
            m.is_baseline = (m.name == baseline)
        
        return cls(
            models=models,
            output_dir=args.output,
            baseline_model=baseline,
            generate_plots=not args.no_plots,
            dpi=args.dpi,
        )


# =============================================================================
# Utility Classes
# =============================================================================

class Tee:
    """Write output to multiple streams simultaneously."""
    def __init__(self, *files):
        self.files = files
    
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    
    def flush(self):
        for f in self.files:
            f.flush()


# =============================================================================
# Constants
# =============================================================================

# Circumplex Model quadrant labels
CONFUSION_LABELS = ["HVHA", "HVLA", "LVHA", "LVLA"]
PRED_OTHER = "Other/Unknown"
GT_OTHER = "Other"
EMPTY_VALUES = {"", None}

# Quadrant coordinates for WAD calculation
QUADRANT_COORDS: Dict[str, Tuple[int, int]] = {
    "HVHA": (+1, +1),  # High Valence, High Arousal
    "HVLA": (+1, -1),  # High Valence, Low Arousal
    "LVHA": (-1, +1),  # Low Valence, High Arousal
    "LVLA": (-1, -1),  # Low Valence, Low Arousal
}

# CRC (Clinical Reasoning Consistency) metric definitions
CRC_METRICS = {
    "RMSSD": {
        "negative_keywords": [
            "reduced vagal", "decreased vagal", "low vagal",
            "reduced parasympathetic", "decreased parasympathetic",
            "sympathetic dominance", "reduced hrv", "decreased hrv",
            "reduced variability", "stress", "strain",
        ],
        "positive_keywords": [
            "increased vagal", "elevated vagal", "high vagal",
            "increased parasympathetic", "parasympathetic dominance",
            "increased hrv", "elevated hrv", "relaxed", "calm",
        ],
    },
    "SDNN": {
        "negative_keywords": [
            "reduced hrv", "decreased hrv", "low hrv",
            "reduced overall variability", "autonomic dysfunction",
            "stress", "reduced adaptability",
        ],
        "positive_keywords": [
            "increased hrv", "elevated hrv", "high hrv",
            "good variability", "healthy variability", "adaptable",
        ],
    },
    "pNN50": {
        "negative_keywords": [
            "reduced vagal", "decreased vagal", "low vagal",
            "reduced parasympathetic", "sympathetic dominance",
        ],
        "positive_keywords": [
            "increased vagal", "elevated vagal", "high vagal",
            "good vagal modulation", "parasympathetic",
        ],
    },
    "MeanHR": {
        "negative_keywords": [
            "decreased heart rate", "low heart rate", "bradycardia",
            "reduced hr", "parasympathetic dominance",
        ],
        "positive_keywords": [
            "increased heart rate", "elevated heart rate", "tachycardia",
            "sympathetic activation", "stress", "arousal",
        ],
    },
    "LF_HF_ratio": {
        "negative_keywords": [
            "parasympathetic dominance", "vagal dominance",
            "reduced sympathetic",
        ],
        "positive_keywords": [
            "sympathetic dominance", "sympathetic activation",
            "stress", "arousal", "increased sympathetic",
        ],
    },
    "SampEn": {
        "negative_keywords": [
            "reduced complexity", "decreased complexity",
            "more regular", "rigid", "reduced entropy",
        ],
        "positive_keywords": [
            "increased complexity", "elevated complexity",
            "healthy complexity", "adaptable", "good entropy",
        ],
    },
    "DFA_alpha": {
        "negative_keywords": [
            "reduced complexity", "decreased fractal",
            "random", "uncorrelated",
        ],
        "positive_keywords": [
            "increased correlation", "long-range correlation",
            "fractal", "trending",
        ],
    },
}


# =============================================================================
# Stage 1: Report Parsing
# =============================================================================

def _parse_score(text: str, key: str) -> Optional[float]:
    """Parse Valence=3/5 or Valence=3.0 format."""
    m = re.search(rf"{key}=([0-9.]+)(?:/5)?", text, re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_ground_truth(text: str) -> Dict[str, Any]:
    """Extract ground truth (Valence and Arousal) from second line."""
    gt = {"valence": None, "arousal": None, "valence_label": "", "arousal_label": "", "gt_state": ""}
    
    lines = text.split('\n')
    if len(lines) < 2:
        return gt
    
    second_line = lines[1]
    val = _parse_score(second_line, "Valence")
    aro = _parse_score(second_line, "Arousal")

    if val is not None:
        gt["valence"] = val
        if gt["valence"] < 3:
            gt["valence_label"] = "L"
        elif gt["valence"] > 3:
            gt["valence_label"] = "H"
        else:
            gt["valence_label"] = "N"
    
    if aro is not None:
        gt["arousal"] = aro
        if gt["arousal"] < 3:
            gt["arousal_label"] = "L"
        elif gt["arousal"] > 3:
            gt["arousal_label"] = "H"
        else:
            gt["arousal_label"] = "N"
    
    if gt["valence_label"] and gt["arousal_label"]:
        if gt["valence_label"] == "N" or gt["arousal_label"] == "N":
            gt["gt_state"] = "Neutral"
        else:
            gt["gt_state"] = f"{gt['valence_label']}V{gt['arousal_label']}A"
    
    return gt


def _last_match(pattern: str, text: str, flags=re.MULTILINE | re.IGNORECASE):
    """Find last match of pattern in text."""
    last = None
    for m in re.finditer(pattern, text, flags):
        last = m
    return last


def _normalize_confidence(val: str) -> str:
    """Normalize confidence value."""
    v = (val or "").strip().lower()
    if v in {"high", "medium", "low"}:
        return v.capitalize()
    return (val or "").strip()


def parse_answer_fields(text: str) -> Dict[str, str]:
    """Extract State, Learning, Confidence from report."""
    ans = {"state": "", "learning": "", "confidence": ""}
    
    # Try simple format first (most stable)
    state_match = _last_match(r"^State:\s*(HVHA|HVLA|LVHA|LVLA)\s*$", text)
    if state_match:
        ans["state"] = state_match.group(1).upper()
    
    learning_match = _last_match(
        r"^Learning:\s*(Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused)\s*$",
        text
    )
    if learning_match:
        ans["learning"] = learning_match.group(1)
    
    confidence_match = _last_match(r"^Confidence:\s*(High|Medium|Low)\s*$", text)
    if confidence_match:
        ans["confidence"] = _normalize_confidence(confidence_match.group(1))
    
    # Fallback: extract from <answer> block
    if not ans["state"] or not ans["learning"] or not ans["confidence"]:
        m = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
        block = m.group(1) if m else text

        if not ans["state"]:
            patterns_state = [
                r"Inferred Psychophysiological State:\s*\*\*([^*]+)\*\*",
                r"Inferred Psychophysiological State:\s*([^\n]+)",
            ]
            for pattern in patterns_state:
                m1 = re.search(pattern, block, re.IGNORECASE | re.MULTILINE)
                if m1:
                    ans["state"] = m1.group(1).strip().rstrip(".")
                    break
        
        if not ans["learning"]:
            patterns_learning = [
                r"Learning\s+State\s+Correlate:\s*\*\*([^*]+)\*\*",
                r"Learning\s+State\s+Correlate:\s*([^\n]+)",
            ]
            for pattern in patterns_learning:
                m2 = re.search(pattern, block, re.IGNORECASE | re.MULTILINE)
                if m2:
                    ans["learning"] = m2.group(1).strip().rstrip(".")
                    break
        
        if not ans["confidence"]:
            m3 = _last_match(r"Confidence\s+Level:\s*(\w+)", block, re.IGNORECASE)
            if m3:
                ans["confidence"] = _normalize_confidence(m3.group(1))
    
    return ans


def has_rag_header(text: str) -> bool:
    """Check if report contains RAG header."""
    return "RAG 增強" in text or "PIKE-RAG" in text or "RAG 檢索結果" in text


def has_z_scores(text: str) -> bool:
    """Check if report contains Z-scores."""
    return "(z =" in text or "z =" in text


def parse_z_scores(text: str) -> Dict[str, float]:
    """Parse Z-score values from report."""
    z_scores = {}
    patterns = [
        r'(\w+)(?:_ms|_bpm)?(?::\s*[\d.]+)?\s*\(z\s*=\s*([+-]?[\d.]+)\)',
        r'\*\*(\w+)(?:_ms|_bpm)?\s*\(z\s*=\s*([+-]?[\d.]+)\)\*\*',
        r'\b(\w+)\s*\(z\s*=\s*([+-]?[\d.]+)\)',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            metric_name = match.group(1).upper().replace("_MS", "").replace("_BPM", "")
            try:
                z_value = float(match.group(2))
                for key in CRC_METRICS.keys():
                    if key.upper() in metric_name or metric_name in key.upper():
                        z_scores[key] = z_value
                        break
            except ValueError:
                continue
    
    return z_scores


def check_text_consistency(text: str, metric: str, z_value: float, threshold: float = 0.5) -> str:
    """Check if text description is consistent with Z-score direction."""
    if metric not in CRC_METRICS:
        return "unknown"
    
    text_lower = text.lower()
    config = CRC_METRICS[metric]
    
    if abs(z_value) <= threshold:
        return "neutral"
    
    neg_found = sum(1 for kw in config["negative_keywords"] if kw in text_lower)
    pos_found = sum(1 for kw in config["positive_keywords"] if kw in text_lower)
    
    if neg_found == 0 and pos_found == 0:
        return "no_keywords"
    
    if z_value < -threshold:
        if neg_found > pos_found:
            return "consistent"
        elif pos_found > neg_found:
            return "inconsistent"
        else:
            return "ambiguous"
    else:
        if pos_found > neg_found:
            return "consistent"
        elif neg_found > pos_found:
            return "inconsistent"
        else:
            return "ambiguous"


def compute_crc_score(text: str) -> Dict[str, Any]:
    """Compute Clinical Reasoning Consistency score."""
    z_scores = parse_z_scores(text)
    
    result = {
        "crc_score": 0.0,
        "total_checked": 0,
        "consistent": 0,
        "inconsistent": 0,
        "neutral": 0,
        "no_keywords": 0,
        "details": {},
    }
    
    if not z_scores:
        return result
    
    for metric, z_value in z_scores.items():
        consistency = check_text_consistency(text, metric, z_value)
        result["details"][metric] = {"z": z_value, "result": consistency}
        
        if consistency == "consistent":
            result["consistent"] += 1
            result["total_checked"] += 1
        elif consistency == "inconsistent":
            result["inconsistent"] += 1
            result["total_checked"] += 1
        elif consistency == "neutral":
            result["neutral"] += 1
        elif consistency == "no_keywords":
            result["no_keywords"] += 1
    
    if result["total_checked"] > 0:
        result["crc_score"] = result["consistent"] / result["total_checked"]
    
    return result


def clean_field(text: str) -> str:
    """Clean field by removing ** markers."""
    if not text or text == "":
        return ""
    return text.replace("**", "").strip()


def extract_main_state(text: str) -> str:
    """Extract main state label (LVHA/HVHA/HVLA/LVLA)."""
    if not text:
        return "Unknown"
    text = clean_field(text)
    patterns = [
        r'\b(LVHA|HVHA|HVLA|LVLA)\b',
        r'(Low Vagal.*?High.*?Arousal)',
        r'(High Vagal.*?High.*?Arousal)',
        r'(High Vagal.*?Low.*?Arousal)',
        r'(Low Vagal.*?Low.*?Arousal)',
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            label = m.group(1).upper()
            if 'LOW VAGAL' in label and 'HIGH' in label:
                return 'LVHA'
            elif 'HIGH VAGAL' in label and 'HIGH' in label:
                return 'HVHA'
            elif 'HIGH VAGAL' in label and 'LOW' in label:
                return 'HVLA'
            elif 'LOW VAGAL' in label and 'LOW' in label:
                return 'LVLA'
            else:
                return label[:4] if len(label) >= 4 else label
    return "Other"


def extract_learning_category(text: str) -> str:
    """Extract learning state category."""
    if not text:
        return "Unknown"
    text = clean_field(text)
    text_lower = text.lower()
    
    if any(k in text_lower for k in ['engaged', 'curious', 'problem-solving']):
        return 'Engaged/Curious'
    elif any(k in text_lower for k in ['focused', 'confident', 'flow', 'zone']):
        return 'Focused/Flow'
    elif any(k in text_lower for k in ['anxious', 'stressed', 'agitated', 'overwhelmed']):
        return 'Anxious/Stressed'
    elif any(k in text_lower for k in ['confused', 'disengaged', 'zoned out', 'fatigued']):
        return 'Disengaged/Confused'
    else:
        return 'Other'


def list_tasks(root: Path) -> List[Tuple[str, str]]:
    """List all subject/trial pairs in a directory."""
    tasks = []
    for sdir in sorted([p for p in root.iterdir() if p.is_dir()]):
        for f in sorted(sdir.glob("T*.txt")):
            sid = sdir.name
            tid = f.stem
            tasks.append((sid, tid))
    return tasks


def parse_all_reports(config: EvalConfig) -> List[Dict[str, Any]]:
    """Parse all reports and generate comparison data."""
    model_dirs = {m.name: Path(m.path) for m in config.models}
    baseline_name = config.baseline_model
    
    if baseline_name not in model_dirs:
        raise ValueError(f"Baseline model '{baseline_name}' not found in model list")
    
    baseline_root = model_dirs[baseline_name]
    tasks = list_tasks(baseline_root)
    
    rows = []
    missing = 0
    
    for sid, tid in tasks:
        texts = {}
        missing_models = []
        
        for name, root in model_dirs.items():
            f_task = root / sid / f"{tid}.txt"
            if not f_task.exists():
                missing_models.append(name)
            else:
                texts[name] = f_task.read_text(encoding="utf-8", errors="ignore")

        if missing_models:
            missing += 1
            print(f"[{sid} {tid}] Missing reports: {', '.join(missing_models)}")
            continue

        tv_base = texts[baseline_name]
        gt = parse_ground_truth(tv_base)

        model_info = {}
        for name, text in texts.items():
            a = parse_answer_fields(text)
            state_label = extract_main_state(a["state"])
            learning_cat = extract_learning_category(a["learning"])
            crc = compute_crc_score(text)

            info = {
                "state": a["state"],
                "state_label": state_label,
                "learning": a["learning"],
                "learning_cat": learning_cat,
                "conf": a["confidence"],
                "header": "Y" if has_rag_header(text) else "N",
                "has_zscore": "Y" if has_z_scores(text) else "N",
                "crc_score": crc["crc_score"],
                "crc_consistent": crc["consistent"],
                "crc_inconsistent": crc["inconsistent"],
                "crc_total": crc["total_checked"],
            }

            # GT matching
            if gt["gt_state"] == "Neutral":
                info["gt_match"] = "Neutral"
            else:
                if state_label not in ("Unknown", "Other") and state_label == gt["gt_state"]:
                    info["gt_match"] = "Y"
                else:
                    info["gt_match"] = "N"

            # Arousal matching
            if gt["arousal_label"] == "N":
                info["arousal_match"] = "Neutral"
            else:
                pred_arousal = state_label[2:4] if len(state_label) == 4 else ""
                gt_arousal = gt["arousal_label"] + "A" if gt["arousal_label"] else ""
                info["arousal_match"] = "Y" if (pred_arousal and gt_arousal and pred_arousal == gt_arousal) else "N"

            # Vagal matching
            if gt["valence_label"] == "N":
                info["vagal_match"] = "Neutral"
            else:
                pred_vagal = state_label[0:2] if len(state_label) == 4 else ""
                gt_vagal = gt["valence_label"] + "V" if gt["valence_label"] else ""
                info["vagal_match"] = "Y" if (pred_vagal and gt_vagal and pred_vagal == gt_vagal) else "N"

            model_info[name] = info

        # Cross-model consistency
        base_info = model_info[baseline_name]
        for name, info in model_info.items():
            if name == baseline_name:
                continue
            info[f"state_match_vs_{baseline_name}"] = (
                "Y" if base_info["state_label"] == info["state_label"] else "N"
            )
            info[f"learning_match_vs_{baseline_name}"] = (
                "Y" if base_info["learning_cat"] == info["learning_cat"] else "N"
            )

        # Build row
        row = {
            "subject": sid,
            "trial": tid,
            "gt_valence": gt["valence"],
            "gt_arousal": gt["arousal"],
            "gt_state": gt["gt_state"],
        }

        for name, info in model_info.items():
            prefix = name
            row[f"{prefix}_state"] = info["state"]
            row[f"{prefix}_state_label"] = info["state_label"]
            row[f"{prefix}_learning"] = info["learning"]
            row[f"{prefix}_learning_cat"] = info["learning_cat"]
            row[f"{prefix}_conf"] = info["conf"]
            row[f"{prefix}_gt_match"] = info["gt_match"]
            row[f"{prefix}_arousal_match"] = info["arousal_match"]
            row[f"{prefix}_vagal_match"] = info["vagal_match"]
            row[f"{prefix}_header"] = info["header"]
            row[f"{prefix}_has_zscore"] = info["has_zscore"]
            row[f"{prefix}_crc_score"] = info["crc_score"]
            row[f"{prefix}_crc_consistent"] = info["crc_consistent"]
            row[f"{prefix}_crc_inconsistent"] = info["crc_inconsistent"]
            row[f"{prefix}_crc_total"] = info["crc_total"]

            if name != baseline_name:
                row[f"{prefix}_state_match_vs_{baseline_name}"] = info[f"state_match_vs_{baseline_name}"]
                row[f"{prefix}_learning_match_vs_{baseline_name}"] = info[f"learning_match_vs_{baseline_name}"]

        rows.append(row)
    
    print(f"\nParsed {len(rows)} samples (skipped {missing} due to missing reports)")
    return rows


# =============================================================================
# Stage 2: Analysis and Metrics
# =============================================================================

@dataclass
class ClassMetrics:
    """Per-class precision/recall/F1 metrics."""
    label: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    support: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class WADResult:
    """Weighted Affective Distance result."""
    total_distance: float = 0.0
    sample_count: int = 0
    valence_errors: int = 0
    arousal_errors: int = 0
    both_errors: int = 0
    correct: int = 0

    @property
    def mean_wad(self) -> float:
        return self.total_distance / self.sample_count if self.sample_count > 0 else 0.0

    @property
    def normalized_wad(self) -> float:
        max_dist = math.sqrt(8)
        return self.mean_wad / max_dist if self.sample_count > 0 else 0.0


def safe_pct(num: int, den: int) -> float:
    return (num / den * 100.0) if den else 0.0


def normalize_label(val: str) -> str:
    return (val or "").strip()


def compute_per_class_metrics(confusion: Mapping[str, Mapping[str, int]]) -> Dict[str, ClassMetrics]:
    """Compute per-class metrics from confusion matrix."""
    metrics = {label: ClassMetrics(label=label) for label in CONFUSION_LABELS}

    for gt_label in CONFUSION_LABELS:
        gt_row = confusion.get(gt_label, {})
        for pred_label in CONFUSION_LABELS:
            count = gt_row.get(pred_label, 0)
            if gt_label == pred_label:
                metrics[gt_label].tp += count
            else:
                metrics[gt_label].fn += count
                metrics[pred_label].fp += count
        metrics[gt_label].support = sum(gt_row.get(p, 0) for p in CONFUSION_LABELS + [PRED_OTHER])

    return metrics


def compute_macro_f1(class_metrics: Dict[str, ClassMetrics]) -> Tuple[float, float, float]:
    """Compute macro precision, recall, F1."""
    valid_classes = [m for m in class_metrics.values() if m.support > 0]
    if not valid_classes:
        return 0.0, 0.0, 0.0

    macro_precision = sum(m.precision for m in valid_classes) / len(valid_classes)
    macro_recall = sum(m.recall for m in valid_classes) / len(valid_classes)
    macro_f1 = sum(m.f1 for m in valid_classes) / len(valid_classes)
    return macro_precision, macro_recall, macro_f1


def compute_weighted_f1(class_metrics: Dict[str, ClassMetrics]) -> float:
    """Compute weighted F1 score."""
    total_support = sum(m.support for m in class_metrics.values())
    if total_support == 0:
        return 0.0
    return sum(m.f1 * m.support for m in class_metrics.values()) / total_support


def compute_affective_distance(gt_label: str, pred_label: str) -> Tuple[float, str]:
    """Compute affective distance between two quadrants."""
    if gt_label not in QUADRANT_COORDS or pred_label not in QUADRANT_COORDS:
        return 0.0, "invalid"

    gt_v, gt_a = QUADRANT_COORDS[gt_label]
    pred_v, pred_a = QUADRANT_COORDS[pred_label]
    distance = math.sqrt((gt_v - pred_v) ** 2 + (gt_a - pred_a) ** 2)

    v_error = gt_v != pred_v
    a_error = gt_a != pred_a

    if not v_error and not a_error:
        error_type = "correct"
    elif v_error and a_error:
        error_type = "both"
    elif v_error:
        error_type = "valence"
    else:
        error_type = "arousal"

    return distance, error_type


def compute_wad_from_confusion(confusion: Mapping[str, Mapping[str, int]]) -> WADResult:
    """Compute WAD from confusion matrix."""
    result = WADResult()

    for gt_label in CONFUSION_LABELS:
        gt_row = confusion.get(gt_label, {})
        for pred_label in CONFUSION_LABELS:
            count = gt_row.get(pred_label, 0)
            if count == 0:
                continue

            distance, error_type = compute_affective_distance(gt_label, pred_label)
            result.total_distance += distance * count
            result.sample_count += count

            if error_type == "correct":
                result.correct += count
            elif error_type == "valence":
                result.valence_errors += count
            elif error_type == "arousal":
                result.arousal_errors += count
            elif error_type == "both":
                result.both_errors += count

    return result


def analyze_dimension_confusion(confusion: Mapping[str, Mapping[str, int]]) -> Dict:
    """Analyze High/Low dimension confusion."""
    hv_to_lv = lv_to_hv = ha_to_la = la_to_ha = 0

    hv_labels = {"HVHA", "HVLA"}
    lv_labels = {"LVHA", "LVLA"}
    ha_labels = {"HVHA", "LVHA"}
    la_labels = {"HVLA", "LVLA"}

    for gt_label in CONFUSION_LABELS:
        gt_row = confusion.get(gt_label, {})
        for pred_label in CONFUSION_LABELS:
            count = gt_row.get(pred_label, 0)
            if count == 0 or gt_label == pred_label:
                continue

            if gt_label in hv_labels and pred_label in lv_labels:
                hv_to_lv += count
            elif gt_label in lv_labels and pred_label in hv_labels:
                lv_to_hv += count

            if gt_label in ha_labels and pred_label in la_labels:
                ha_to_la += count
            elif gt_label in la_labels and pred_label in ha_labels:
                la_to_ha += count

    total_valence = hv_to_lv + lv_to_hv
    total_arousal = ha_to_la + la_to_ha

    return {
        "hv_to_lv": hv_to_lv,
        "lv_to_hv": lv_to_hv,
        "total_valence_confusion": total_valence,
        "ha_to_la": ha_to_la,
        "la_to_ha": la_to_ha,
        "total_arousal_confusion": total_arousal,
        "harder_dimension": "Valence" if total_valence > total_arousal else "Arousal",
    }


def summarize_model(rows: List[Mapping[str, str]], model: str, baseline: str) -> Dict:
    """Generate summary statistics for a model."""
    total = len(rows)
    gt_valid = gt_correct = 0
    arousal_valid = arousal_correct = 0
    vagal_valid = vagal_correct = 0
    header_yes = zscore_yes = 0
    state_filled = state_known = learning_filled = 0
    state_dist: Counter = Counter()
    learning_dist: Counter = Counter()
    conf_dist: Counter = Counter()
    confusion = defaultdict(lambda: defaultdict(int))
    
    crc_total_consistent = crc_total_inconsistent = crc_total_checked = crc_samples_with_data = 0

    state_agree = 0
    has_state_agree = False
    state_agree_key = f"{model}_state_match_vs_{baseline}"
    if baseline and baseline != model and rows:
        has_state_agree = state_agree_key in rows[0]

    for r in rows:
        gt_state = normalize_label(r.get("gt_state"))
        pred_state = normalize_label(r.get(f"{model}_state"))
        pred_state_label = normalize_label(r.get(f"{model}_state_label")).upper()
        learning_cat = normalize_label(r.get(f"{model}_learning_cat"))
        conf = normalize_label(r.get(f"{model}_conf"))

        if pred_state:
            state_filled += 1
        if learning_cat:
            learning_filled += 1
        if pred_state_label and pred_state_label not in ("UNKNOWN", "OTHER"):
            state_known += 1

        state_dist[pred_state_label or "UNKNOWN/EMPTY"] += 1
        learning_dist[learning_cat or "UNKNOWN/EMPTY"] += 1
        conf_dist[conf or "EMPTY"] += 1

        if r.get(f"{model}_header") == "Y":
            header_yes += 1
        if r.get(f"{model}_has_zscore") == "Y":
            zscore_yes += 1
        
        crc_checked = int(r.get(f"{model}_crc_total", 0) or 0)
        if crc_checked > 0:
            crc_samples_with_data += 1
            crc_total_checked += crc_checked
            crc_total_consistent += int(r.get(f"{model}_crc_consistent", 0) or 0)
            crc_total_inconsistent += int(r.get(f"{model}_crc_inconsistent", 0) or 0)

        if has_state_agree and r.get(state_agree_key) == "Y":
            state_agree += 1

        if gt_state and gt_state != "Neutral":
            gt_valid += 1
            if r.get(f"{model}_gt_match") == "Y":
                gt_correct += 1

            pred_key = pred_state_label if pred_state_label in CONFUSION_LABELS else PRED_OTHER
            gt_key = gt_state if gt_state in CONFUSION_LABELS else GT_OTHER
            confusion[gt_key][pred_key] += 1

            arousal_flag = r.get(f"{model}_arousal_match")
            if arousal_flag != "Neutral":
                arousal_valid += 1
                if arousal_flag == "Y":
                    arousal_correct += 1

            vagal_flag = r.get(f"{model}_vagal_match")
            if vagal_flag != "Neutral":
                vagal_valid += 1
                if vagal_flag == "Y":
                    vagal_correct += 1

    class_metrics = compute_per_class_metrics(confusion)
    macro_p, macro_r, macro_f1 = compute_macro_f1(class_metrics)
    weighted_f1 = compute_weighted_f1(class_metrics)
    wad_result = compute_wad_from_confusion(confusion)
    dim_confusion = analyze_dimension_confusion(confusion)
    crc_score = (crc_total_consistent / crc_total_checked * 100) if crc_total_checked > 0 else 0.0

    return {
        "model": model,
        "samples": total,
        "gt_valid": gt_valid,
        "gt_accuracy": safe_pct(gt_correct, gt_valid),
        "arousal_accuracy": safe_pct(arousal_correct, arousal_valid),
        "vagal_accuracy": safe_pct(vagal_correct, vagal_valid),
        "macro_precision": macro_p * 100,
        "macro_recall": macro_r * 100,
        "macro_f1": macro_f1 * 100,
        "weighted_f1": weighted_f1 * 100,
        "class_metrics": class_metrics,
        "wad": wad_result,
        "mean_wad": wad_result.mean_wad,
        "normalized_wad": wad_result.normalized_wad,
        "dim_confusion": dim_confusion,
        "crc_score": crc_score,
        "crc_consistent": crc_total_consistent,
        "crc_inconsistent": crc_total_inconsistent,
        "crc_total_checked": crc_total_checked,
        "crc_samples": crc_samples_with_data,
        "header_pct": safe_pct(header_yes, total),
        "zscore_pct": safe_pct(zscore_yes, total),
        "state_fill_pct": safe_pct(state_filled, total),
        "state_known_pct": safe_pct(state_known, total),
        "learning_fill_pct": safe_pct(learning_filled, total),
        "state_dist": state_dist,
        "learning_dist": learning_dist,
        "conf_dist": conf_dist,
        "state_agree_pct": safe_pct(state_agree, total) if has_state_agree else None,
        "confusion": confusion,
    }


def detect_models(fieldnames: Iterable[str]) -> List[str]:
    """Detect model names from CSV field names."""
    skip = {"gt"}
    models = set()
    for name in fieldnames:
        if not name.endswith("_state"):
            continue
        prefix = name[:-len("_state")]
        if prefix in skip:
            continue
        models.add(prefix)
    return sorted(models)


# =============================================================================
# Output Functions
# =============================================================================

def counter_to_str(counter: Counter) -> str:
    return "; ".join(f"{k}:{v}" for k, v in sorted(counter.items()))


def write_comparison_csv(rows: List[Dict], out_path: Path):
    """Write comparison CSV file."""
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_summary_csv(summary: List[Dict], out_path: Path):
    """Write summary CSV file."""
    if not summary:
        return
    fields = [
        "model", "samples", "gt_valid", "gt_accuracy",
        "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
        "mean_wad", "normalized_wad",
        "crc_score", "crc_consistent", "crc_inconsistent", "crc_total_checked",
        "arousal_accuracy", "vagal_accuracy",
        "state_agree_pct", "state_fill_pct", "state_known_pct",
        "learning_fill_pct", "header_pct", "zscore_pct",
        "valence_confusion", "arousal_confusion", "harder_dimension",
        "state_dist", "learning_dist", "conf_dist",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in summary:
            row = {k: m.get(k) for k in fields}
            row["state_dist"] = counter_to_str(m["state_dist"])
            row["learning_dist"] = counter_to_str(m["learning_dist"])
            row["conf_dist"] = counter_to_str(m["conf_dist"])
            row["valence_confusion"] = m["dim_confusion"]["total_valence_confusion"]
            row["arousal_confusion"] = m["dim_confusion"]["total_arousal_confusion"]
            row["harder_dimension"] = m["dim_confusion"]["harder_dimension"]
            writer.writerow(row)


def write_confusion_matrix(confusion: Mapping[str, Mapping[str, int]], out_path: Path):
    """Write confusion matrix to CSV."""
    gt_labels = CONFUSION_LABELS + [GT_OTHER]
    pred_labels = CONFUSION_LABELS + [PRED_OTHER]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + pred_labels)
        for gt in gt_labels:
            row = [gt]
            for pred in pred_labels:
                row.append(confusion.get(gt, {}).get(pred, 0))
            writer.writerow(row)


def print_summary(m: Dict, baseline: str):
    """Print summary for a single model."""
    print("=" * 80)
    print(f"Model: {m['model']}")
    print("-" * 80)
    print(f"Valid samples (GT≠Neutral): {m['gt_valid']}/{m['samples']}")
    
    print(f"\n[Accuracy]")
    print(f"  Overall: {m['gt_accuracy']:.1f}% | Arousal: {m['arousal_accuracy']:.1f}% | Vagal: {m['vagal_accuracy']:.1f}%")
    
    print(f"\n[F1 Scores (Class Imbalance Robust)]")
    print(f"  Macro F1:    {m['macro_f1']:.1f}% (P: {m['macro_precision']:.1f}%, R: {m['macro_recall']:.1f}%)")
    print(f"  Weighted F1: {m['weighted_f1']:.1f}%")
    
    print(f"\n[Per-Class F1]")
    for label in CONFUSION_LABELS:
        cm = m["class_metrics"][label]
        print(f"  {label}: P={cm.precision*100:.1f}%, R={cm.recall*100:.1f}%, F1={cm.f1*100:.1f}% (n={cm.support})")
    
    wad = m["wad"]
    print(f"\n[WAD (lower is better)]")
    print(f"  Mean WAD: {wad.mean_wad:.4f}, Normalized: {wad.normalized_wad:.4f}")
    print(f"  Errors: Correct={wad.correct}, Valence={wad.valence_errors}, Arousal={wad.arousal_errors}, Both={wad.both_errors}")
    
    dim = m["dim_confusion"]
    print(f"\n[Dimension Confusion]")
    print(f"  Valence (HV↔LV): {dim['total_valence_confusion']} | Arousal (HA↔LA): {dim['total_arousal_confusion']}")
    print(f"  Harder dimension: {dim['harder_dimension']}")
    
    if m.get("crc_total_checked", 0) > 0:
        print(f"\n[CRC (Clinical Reasoning Consistency)]")
        print(f"  Score: {m['crc_score']:.1f}% ({m['crc_consistent']}/{m['crc_total_checked']} consistent)")


def print_comparative_summary(summary: List[Dict]):
    """Print comparative summary table."""
    print("\n" + "=" * 110)
    print("[Model Comparison Summary]")
    print("=" * 110)
    
    header = f"{'Model':<25} {'Acc%':>6} {'MacroF1%':>9} {'MeanWAD':>8} {'CRC%':>6} {'ValConf':>8} {'AroConf':>8}"
    print(header)
    print("-" * 110)
    
    sorted_summary = sorted(summary, key=lambda x: x["macro_f1"], reverse=True)
    
    for m in sorted_summary:
        dim = m["dim_confusion"]
        crc = m.get("crc_score", 0)
        row = (f"{m['model']:<25} "
               f"{m['gt_accuracy']:>6.1f} "
               f"{m['macro_f1']:>9.1f} "
               f"{m['mean_wad']:>8.4f} "
               f"{crc:>6.1f} "
               f"{dim['total_valence_confusion']:>8} "
               f"{dim['total_arousal_confusion']:>8}")
        print(row)
    
    print("-" * 110)


# =============================================================================
# Visualization (Optional)
# =============================================================================

def setup_plot_style():
    """Setup Seaborn academic style."""
    if not HAS_SEABORN:
        return
    
    sns.set_style("whitegrid", {
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    plt.rcParams.update({
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })


def plot_accuracy(summary: List[Dict], out_dir: Path, dpi: int):
    """Plot accuracy comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    models = [m["model"] for m in summary]
    metrics = {
        "GT Accuracy": [m["gt_accuracy"] for m in summary],
        "Arousal Accuracy": [m["arousal_accuracy"] for m in summary],
        "Vagal Accuracy": [m["vagal_accuracy"] for m in summary],
    }
    
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(models))
    width = 0.25
    
    colors = sns.color_palette("Set2", 3) if HAS_SEABORN else ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i, (label, values) in enumerate(metrics.items()):
        ax.bar(x + i * width, values, width, label=label, color=colors[i])
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_dir / "accuracy.png", dpi=dpi)
    plt.close(fig)


def plot_f1_scores(summary: List[Dict], out_dir: Path, dpi: int):
    """Plot F1 score comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    models = [m["model"] for m in summary]
    
    fig, ax = plt.subplots(figsize=(7, 3))
    x = np.arange(len(models))
    width = 0.35
    
    colors = sns.color_palette("Set2", 2) if HAS_SEABORN else ['#1f77b4', '#ff7f0e']
    
    ax.bar(x - width/2, [m["macro_f1"] for m in summary], width, label="Macro F1", color=colors[0])
    ax.bar(x + width/2, [m["weighted_f1"] for m in summary], width, label="Weighted F1", color=colors[1])
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("F1 Score (%)")
    ax.set_title("F1 Score Comparison")
    ax.legend()
    
    plt.tight_layout()
    fig.savefig(out_dir / "f1_scores.png", dpi=dpi)
    plt.close(fig)


def plot_wad(summary: List[Dict], out_dir: Path, dpi: int):
    """Plot WAD comparison."""
    if not HAS_MATPLOTLIB:
        return
    
    models = [m["model"] for m in summary]
    wad_values = [m["mean_wad"] for m in summary]
    
    fig, ax = plt.subplots(figsize=(7, 3))
    
    colors = sns.color_palette("viridis", len(models)) if HAS_SEABORN else plt.cm.viridis(np.linspace(0, 1, len(models)))
    
    bars = ax.bar(range(len(models)), wad_values, color=colors)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel("Mean WAD (lower is better)")
    ax.set_title("Weighted Affective Distance")
    
    for bar, val in zip(bars, wad_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(out_dir / "wad_score.png", dpi=dpi)
    plt.close(fig)


def plot_all(summary: List[Dict], out_dir: Path, dpi: int):
    """Generate all plots."""
    if not HAS_MATPLOTLIB:
        print("Matplotlib not available, skipping plots")
        return
    
    setup_plot_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_accuracy(summary, out_dir, dpi)
    plot_f1_scores(summary, out_dir, dpi)
    plot_wad(summary, out_dir, dpi)
    
    print(f"Plots saved to: {out_dir}")


# =============================================================================
# CLI and Main Entry Point
# =============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CGRASP Model Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Using config file
    python model_evaluation.py --config models.yml
    
    # Using command line arguments
    python model_evaluation.py \\
        --model-dirs ./reports/model1 ./reports/model2 \\
        --model-names "Model A" "Model B" \\
        --baseline "Model A" \\
        --output ./analysis

Config file format (YAML):
    models:
      - name: "MedGemma-27B"
        path: "/path/to/reports"
      - name: "Qwen-V3-8B"
        path: "/path/to/reports"
    baseline_model: "Qwen-V3-8B"
    output_dir: "./analysis"
    generate_plots: true
        """
    )
    
    parser.add_argument("--config", "-c", type=str, help="Path to YAML config file")
    parser.add_argument("--model-dirs", nargs="+", help="Paths to model report directories")
    parser.add_argument("--model-names", nargs="+", help="Names for models (same order as dirs)")
    parser.add_argument("--baseline", "-b", type=str, help="Baseline model name for consistency")
    parser.add_argument("--output", "-o", type=str, default="./analysis", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for plots")
    
    return parser.parse_args()


def run(config: EvalConfig):
    """Run the complete evaluation pipeline."""
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = out_dir / "evaluation_log.txt"
    original_stdout = sys.stdout
    
    with open(log_path, "w", encoding="utf-8") as f:
        sys.stdout = Tee(original_stdout, f)
        
        try:
            print("=" * 60)
            print("CGRASP Model Evaluation Pipeline")
            print("=" * 60)
            print(f"Models: {[m.name for m in config.models]}")
            print(f"Baseline: {config.baseline_model}")
            print(f"Output: {config.output_dir}")
            print("=" * 60)
            
            # Stage 1: Parse reports
            print("\n[Stage 1] Parsing model reports...")
            rows = parse_all_reports(config)
            
            if not rows:
                print("No data to analyze")
                return
            
            # Write comparison CSV
            comparison_csv = out_dir / "model_comparison.csv"
            write_comparison_csv(rows, comparison_csv)
            print(f"Comparison CSV: {comparison_csv}")
            
            # Stage 2: Analyze
            print("\n[Stage 2] Computing metrics...")
            models = detect_models(rows[0].keys())
            
            summary_rows = []
            for model in models:
                metrics = summarize_model(rows, model, config.baseline_model)
                summary_rows.append(metrics)
                print_summary(metrics, config.baseline_model)
                
                # Write confusion matrix
                conf_path = out_dir / f"confusion_{model}.csv"
                write_confusion_matrix(metrics["confusion"], conf_path)
            
            print_comparative_summary(summary_rows)
            
            # Write summary CSV
            summary_csv = out_dir / "analysis_summary.csv"
            write_summary_csv(summary_rows, summary_csv)
            print(f"\nSummary CSV: {summary_csv}")
            
            # Stage 3: Visualization
            if config.generate_plots:
                print("\n[Stage 3] Generating plots...")
                plot_all(summary_rows, out_dir, config.dpi)
            
            print("\n" + "=" * 60)
            print("Evaluation complete!")
            print(f"Results saved to: {out_dir}")
            print("=" * 60)
            
        finally:
            sys.stdout = original_stdout


def main():
    """Main entry point."""
    args = parse_args()
    
    # Validate arguments
    if not args.config and not args.model_dirs:
        print("Error: Either --config or --model-dirs is required")
        print("Use --help for usage information")
        sys.exit(1)
    
    config = EvalConfig.from_args(args)
    
    if not config.models:
        print("Error: No models configured")
        sys.exit(1)
    
    run(config)


if __name__ == "__main__":
    main()
