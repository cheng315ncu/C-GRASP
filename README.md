# C-GRASP
# CGRASP - Clinical Grade RAG System for Psychophysiology

<p align="center">
  <img src="docs/source/images/logo/PIKE-RAG_icon.svg" alt="CGRASP Logo" width="120"/>
</p>

**CGRASP** is a multimodal AI system for automated Heart Rate Variability (HRV) analysis with clinical-grade reasoning. It combines large language models (LLMs) with Retrieval-Augmented Generation (RAG) to provide evidence-based psychophysiological assessments.

## Features

- **Multi-step Clinical Reasoning**: 8-step pipeline covering signal quality, time/frequency domain analysis, Poincaré geometry, complexity metrics, and final integration
- **RAG-Enhanced Analysis**: Retrieves relevant clinical literature from your PDF knowledge base to ground LLM reasoning
- **Within-Subject Baseline Comparison**: Uses individual Z-scores and Delta features for personalized assessment
- **Multimodal Input**: Processes numerical HRV features + Poincaré plot images
- **Structured Output**: Generates standardized reports with State (HVHA/HVLA/LVHA/LVLA), Learning State, and Confidence levels

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/CGRASP.git
cd CGRASP

# Install uv (recommended)
# https://docs.astral.sh/uv/
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Prepare Your Data

```bash
# Create required directories
mkdir -p data clinical_pdfs outputs

# Place your HRV CSV file
cp /path/to/your/HRV_data.csv data/HRV_all_final.csv

# (Optional) Add clinical PDFs for RAG
cp /path/to/clinical_papers/*.pdf clinical_pdfs/
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

### 4. Run Analysis

```bash
# Basic run (without RAG)
python main.py --no-rag

# With RAG enabled
python main.py --rag

# Specify custom paths
CGRASP_CSV_PATH=/path/to/data.csv python main.py
```

## Data Preprocessing (Optional)

This repo also includes an **HRV preprocessing pipeline** (`preprocess2.py`) intended for datasets like DREAMER (`DREAMER.mat`). It can generate per-trial features and a final CSV compatible with `main.py`.

```bash
# Generate processed HRV CSV + images (Poincaré/PSD/etc.)
python preprocess2.py --mat-path /path/to/DREAMER.mat --output-dir ./data/processed

# Quick smoke test on a small subset (faster)
python preprocess2.py --mat-path /path/to/DREAMER.mat --output-dir ./data/processed --subjects 1 2 --trials 1 3 --no-plots
```

Environment variables supported:

- `CGRASP_MAT_PATH`: path to `DREAMER.mat`
- `CGRASP_PREPROCESS_OUTPUT`: output directory for preprocessing results

## Model Evaluation (Optional)

If you have multiple model report folders (e.g. `report_xxx/S01/T01.txt`) and want to compare them, use `model_evaluation.py`.  
It integrates the previous `test2.py` + `ana.py` workflow into one script and produces:

- `model_comparison.csv` (per-sample)
- `analysis_summary.csv` (per-model metrics)
- `confusion_*.csv`
- optional plots (`accuracy.png`, `f1_scores.png`, `wad_score.png`)

### CLI usage

```bash
python model_evaluation.py \
  --model-dirs /path/to/report_modelA /path/to/report_modelB \
  --model-names "Model-A" "Model-B" \
  --baseline "Model-A" \
  --output ./analysis
```

### YAML config usage

Create `models.yml`:

```yaml
models:
  - name: "Model-A"
    path: "/path/to/report_modelA"
  - name: "Model-B"
    path: "/path/to/report_modelB"
baseline_model: "Model-A"
output_dir: "./analysis"
generate_plots: true
dpi: 300
```

Then run:

```bash
python model_evaluation.py --config models.yml
```

Notes:
- Plotting requires optional dependencies: `matplotlib`, `seaborn`, `pandas`
- YAML config requires `pyyaml`

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CGRASP_CSV_PATH` | Path to input HRV CSV file | `data/HRV_all_final.csv` |
| `CGRASP_OUTPUT_DIR` | Output directory for reports | `outputs/reports` |
| `CGRASP_PDF_DIR` | Directory containing clinical PDFs | `clinical_pdfs` |
| `CGRASP_CHROMA_DIR` | ChromaDB vector store path | `chroma_db` |
| `CGRASP_MODEL_ID` | HuggingFace model ID | `Qwen/Qwen3-VL-8B-Instruct` |

### Config File (`config.py`)

Key parameters you may want to adjust:

```python
# RAG Settings
ENABLE_RAG = True                    # Enable/disable RAG
RAG_TOP_K = 5                        # Number of documents to retrieve
RAG_SCORE_THRESHOLD = 0.25           # Relevance threshold
RAG_EMBEDDING_MODEL = "FremyCompany/BioLORD-2023"  # Embedding model

# Generation Settings
TEMPERATURE = 0.3                    # Lower = more deterministic
TOP_P = 0.85                         # Nucleus sampling parameter
INFERENCE_OUTPUT_LENGTH = 1024       # Max tokens for Step 1-7
SUMMARY_OUTPUT_LENGTH = 4096         # Max tokens for Step 8

# Feature Toggles
ENABLE_EEG_ANALYSIS = False          # Enable EEG multimodal analysis
ENABLE_GUARDRAILS = True             # Enable methodological guardrails
ENABLE_DELTA_ZSCORE = True           # Use Delta Z-scores for baseline
USE_4BIT = False                     # Enable 4-bit quantization (saves VRAM)
```

## Input Data Format

Your CSV should contain the following columns:

### Required Columns
- `subject`, `trial` - Identifiers
- `MeanRR_ms`, `SDNN_ms`, `RMSSD_ms`, `MeanHR_bpm` - Time domain features
- `LF_HF`, `HF_ratio`, `LF_ratio` - Frequency domain features
- `SampEn`, `DFA_alpha` - Nonlinear features
- `SD1_ms`, `SD2_ms` - Poincaré features
- `img_path` - Path to Poincaré plot image

### Optional Columns
- `*_zscore` - Z-score normalized features
- `Delta_*` - Baseline difference features
- `valence`, `arousal` - Ground truth labels (for evaluation)
- `EEG_*` - EEG features (if multimodal)
- `overall_quality`, `spectral_quality` - Quality indicators

## Output Format

Reports are saved to `{OUTPUT_DIR}/{subject}/{trial}.txt` with:

1. **Step 1-7 Sub-reports**: Signal quality, time domain, frequency domain, Poincaré/complexity, baseline delta, within-subject, EEG analysis
2. **Step 8 Integrated Report**: Final clinical reasoning with:
   - `State: HVHA|HVLA|LVHA|LVLA`
   - `Learning: Engaged/Curious|Focused/Flow|Anxious/Stressed|Disengaged/Confused`
   - `Confidence: High|Medium|Low`

## Project Structure

```
CGRASP/
├── main.py                      # Main entry point
├── config.py                    # Configuration settings
├── dataset.py                   # HRV dataset loader
├── inference_steps.py           # Step 1-7 inference functions
├── rag_system.py                # RAG initialization
├── pikerag_medical_integration.py  # Clinical knowledge retriever
├── system_prompt.py             # LLM system prompts
├── utils.py                     # Utility functions
├── preprocess2.py               # Optional: DREAMER-style preprocessing pipeline (MAT → CSV + images)
├── model_evaluation.py          # Optional: multi-model report evaluation (comparison + metrics + plots)
├── pikerag/                     # PIKE-RAG framework modules
│   ├── knowledge_retrievers/    # Retrieval implementations
│   ├── document_loaders/        # PDF loading utilities
│   ├── prompts/                 # Prompt templates
│   └── workflows/               # Processing pipelines
├── clinical_pdfs/               # Your clinical PDF papers
├── chroma_db/                   # Vector database (generated)
├── data/                        # Input data directory
└── outputs/                     # Generated reports
```

## Supported Models

| Model | VRAM Required | Notes |
|-------|---------------|-------|
| `Qwen/Qwen3-VL-4B-Instruct` | ~8GB | Fast, good for testing |
| `Qwen/Qwen3-VL-8B-Instruct` | ~16GB | Recommended balance |
| `google/medgemma-4b-it` | ~8GB | Medical-specialized |
| `Cannae-AI/MedicalQwen3-Reasoning-14B-IT` | ~28GB | Best quality, slower |

For limited VRAM, enable 4-bit quantization:
```python
USE_4BIT = True  # in config.py
```

## Clinical Knowledge Base (RAG)

To enable RAG-enhanced analysis:

1. Place relevant clinical PDFs in `clinical_pdfs/`:
   - HRV interpretation guidelines
   - Psychophysiology research papers
   - Autonomic nervous system studies

2. The system will automatically:
   - Extract and chunk text from PDFs
   - Build a vector database using sentence embeddings
   - Retrieve relevant passages during analysis

3. Pre-configured literature weighting handles known methodological issues (e.g., LF/HF ratio limitations).

### Suggested Clinical Papers for `clinical_pdfs/`

If you want a starting set of reference PDFs to place in `clinical_pdfs/`, you can download and include the following widely cited works:

- **HRV standards (Task Force 1996)**: *Heart rate variability: Standards of measurement, physiological interpretation, and clinical use* (Task Force of the European Society of Cardiology and the North American Society of Pacing and Electrophysiology).  
  Link: [ESC PDF](https://www.escardio.org/static-file/Escardio/Guidelines/Scientific-Statements/guidelines-Heart-Rate-Variability-FT-1996.pdf)

- **Psychophysiology HR/HRV publication guidelines (Part 1, 2024)**: Committee report on physiological underpinnings and foundations of measurement for HR and HRV in psychophysiology research.  
  Link: [PDF](https://knowledge.uchicago.edu/record/12658/files/Publication-guidelines-for-human-heart-rate-and-heart-rate-variability-studies-in-psychophysiology.pdf)

- **HRV standardisation checklist (Masterclass)**: *Heart rate variability: are you using it properly? Standardisation checklist of procedures* – practical checklist for experimental and clinical HRV use.  
  Link: [Article](https://www.sciencedirect.com/science/article/abs/pii/S1413355518307974)

## Citation

If you use CGRASP in your research, please cite:

```bibtex
@software{cgrasp2024,
  title = {CGRASP: Clinical Grade RAG System for Psychophysiology},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/CGRASP}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [PIKE-RAG](https://github.com/microsoft/PIKE-RAG) framework
- Uses [LangChain](https://github.com/langchain-ai/langchain) for RAG pipeline
- Embedding models from [HuggingFace](https://huggingface.co/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
