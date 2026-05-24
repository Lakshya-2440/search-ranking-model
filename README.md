# Search Ranking Model — LambdaMART Learning-to-Rank

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/la7shya/search-ranking-model)

A production-grade learning-to-rank system that uses **LambdaMART** (XGBoost) to rank search results by relevance, incorporating user signals (CTR, dwell time, skip rate) alongside traditional text-matching features.

## Key Results

| Metric | BM25 Baseline | LambdaMART | Improvement |
|--------|:------------:|:----------:|:-----------:|
| **NDCG@10** | 0.4020 | 0.6292 | **+56.5%** |
| **MRR** | 0.3498 | 0.6289 | **+79.8%** |
| **MAP** | 0.3307 | 0.6283 | **+89.9%** |

## Project Structure

```
search_ranking/
├── configs/config.yaml           # All hyperparameters
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py # 2M+ query-doc pair generation
│   │   ├── label_extraction.py   # Click → relevance grade (0-4) + IPW
│   │   └── splitter.py           # Time-based train/val/test split
│   ├── features/
│   │   ├── query_features.py     # TF-IDF, BM25, query length/type
│   │   ├── doc_features.py       # PageRank, freshness, title match
│   │   ├── interaction_features.py # CTR, dwell, skip rate, IPW
│   │   └── feature_store.py      # Feature registry + pipeline
│   ├── models/
│   │   ├── bm25_baseline.py      # BM25 keyword baseline
│   │   ├── lambdamart.py         # XGBRanker with rank:ndcg
│   │   └── model_utils.py        # Save/load, comparison tables
│   ├── evaluation/
│   │   ├── metrics.py            # NDCG@k, MAP, MRR
│   │   ├── ablation.py           # Feature group ablation study
│   │   └── bias_analysis.py      # Position bias + IPW analysis
│   ├── serving/
│   │   ├── shadow_deploy.py      # Shadow deployment simulation
│   │   └── ab_testing.py         # A/B test framework
│   └── visualization/
│       └── plots.py              # All charts and visualizations
├── tests/                        # Unit tests
├── main.py                       # Full pipeline orchestrator
├── Makefile                      # Convenience commands
└── requirements.txt
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run faster (skip ablation)
python main.py --skip-ablation

# Run tests
python -m pytest tests/ -v
```

## Pipeline Steps

1. **Data Generation** — 2M+ synthetic query-doc pairs with realistic click signals
2. **Label Extraction** — Map click/dwell/skip → relevance grades (0-4) with IPW correction
3. **Data Splitting** — Time-based 70/15/15 split (prevents leakage)
4. **Feature Engineering** — 22 features across 3 groups:
   - **Query**: TF-IDF, BM25, length, type encoding
   - **Document**: PageRank, freshness, title match, anchor text, URL depth
   - **Interaction**: CTR, dwell time, skip rate, IPW weights, click entropy
5. **BM25 Baseline** — Keyword-based ranking baseline
6. **LambdaMART Training** — XGBoost `rank:ndcg` with early stopping
7. **Offline Evaluation** — NDCG@1/5/10, MAP, MRR on held-out test set
8. **Ablation Study** — Remove each feature group, measure NDCG drop
9. **Bias Analysis** — Quantify position bias, evaluate IPW effectiveness
10. **Shadow Deployment** — Side-by-side comparison without user impact
11. **A/B Test Simulation** — 5% traffic split, statistical significance testing

## Model Details

### LambdaMART Configuration
- **Objective**: `rank:ndcg` (LambdaRank loss, optimizes NDCG directly)
- **Trees**: Up to 1000, early stopping on validation NDCG@10
- **Max depth**: 6
- **Learning rate**: 0.05
- **Regularization**: subsample=0.8, colsample=0.8, L1=0.1, L2=1.0

### Position Bias Correction
Uses Inverse Propensity Weighting (IPW) to correct for examination bias:
- Propensity estimated from CTR-by-position ratios
- Weights normalized and capped at 99th percentile
- Comparison: model with vs without IPW correction

## Outputs

After running, find results in `outputs/`:
- `outputs/models/` — Saved LambdaMART model
- `outputs/plots/` — All visualizations (PNG)
- `outputs/results/` — Metric JSONs
