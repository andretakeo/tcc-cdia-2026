# 12 — Code Architecture and Model Specifications

## Repository Structure

```
tcc-cdia-2026/
├── 1.news/
│   └── extractor.py              # ExtratorDeNoticias class (InfoMoney API)
├── 2.stocks/
│   ├── yahoo_finance.py          # MarketData class (yfinance wrapper)
│   ├── news_embedder.py          # NewsEmbedder class (Ollama embeddings)
│   └── dataset_full.csv          # Merged dataset
├── 3.model_traning/
│   ├── lstm_classifier.py        # BiLSTM implementation (Stage 3)
│   ├── transformer_classifier.py # Transformer implementation (Stage 3)
│   └── xgboost_baseline.py       # XGBoost implementation (Stage 3)
├── 4.finbert-br/
│   ├── index.ipynb               # Sentiment extraction pipeline
│   ├── model_training.ipynb      # Model retraining with sentiment
│   ├── FinBERT-PT-BR/            # Pre-trained model weights
│   └── *_daily_sentiment.csv     # Daily sentiment per ticker
├── 5.threshold-tuning/           # Horizon and threshold experiments
├── 6.17years-news/
│   ├── yahoo_finance.py          # Extended MarketData wrapper
│   └── itub4_daily_sentiment_17y.csv
├── 7.model-evaluation/
│   ├── shared/
│   │   ├── models.py             # ALL 6 model architectures
│   │   ├── data_loader.py        # Data loading for stages 3-6
│   │   ├── metrics.py            # Comprehensive metrics
│   │   ├── trainer.py            # Training orchestration
│   │   └── plots.py              # Visualization functions
│   └── results/                  # 48 diagnostic plots
├── 8.multi-source-news/
│   ├── cvm_collector.py          # CVM regulatory data
│   ├── google_news_collector.py  # Google News collection
│   └── results/                  # Multi-source analysis plots
├── 9.baselines/
│   ├── eval_utils.py             # Shared evaluation utilities
│   ├── *.ipynb                   # 15+ experiment notebooks
│   ├── *.csv                     # Raw and aggregated results
│   └── *.png                     # Chapter 5 figures
├── docs/
│   ├── tcc.tex                   # Main thesis LaTeX file
│   ├── referencias.bib           # Bibliography (21 entries)
│   ├── capitulo_4.md             # Chapter 4 draft (Markdown)
│   └── capitulo_5.md             # Chapter 5 draft (Markdown)
└── full-docs/                    # This documentation
```

## Model Architectures

### BiLSTM (`LSTMClassifier`)

```
Input (batch, seq_len=30, n_features)
  → LSTM(input_size=n_features, hidden_size=128, num_layers=2,
         bidirectional=True, dropout=0.3, batch_first=True)
  → Take last hidden state
  → Linear(256, 1)
  → Sigmoid
Output: probability ∈ [0, 1]
```

- **Parameters**: ~200K (varies with input features)
- **Loss**: `nn.BCELoss()` (sigmoid already in forward)
- **Optimizer**: Adam (lr=1e-3, weight_decay=1e-4)
- **Training**: Early stopping (patience=10), gradient clipping (max_norm=1.0)

### Transformer (`TransformerClassifier`)

```
Input (batch, seq_len=30, n_features)
  → Linear(n_features, d_model=64)         # Input projection
  → PositionalEncoding(d_model=64)          # Sinusoidal
  → TransformerEncoder(
        num_layers=2,
        nhead=4,
        d_model=64,
        dim_feedforward=256,
        dropout=0.2
    )
  → Global mean pooling over sequence
  → Linear(64, 32) → ReLU → Dropout(0.2)
  → Linear(32, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

- **Parameters**: ~35K
- **Loss**: `nn.BCELoss()` (models.py version) or `nn.BCEWithLogitsLoss()` (multi_seed version — no sigmoid)
- **Key finding**: This model exhibits bimodal collapse in low-data regimes

### TCN (`TCNClassifier`)

```
Input (batch, seq_len=30, n_features)
  → Permute to (batch, n_features, seq_len)
  → CausalConv1dBlock(n_features, 32, kernel_size=3, dilation=1)
  → CausalConv1dBlock(32, 32, kernel_size=3, dilation=2)
  → Global mean pooling over time
  → Linear(32, 1) → Sigmoid
Output: probability ∈ [0, 1]
```

Each `CausalConv1dBlock` contains:
- `Conv1d` → chomp (remove padding) → ReLU → Dropout
- `Conv1d` → chomp → ReLU → Dropout
- Residual connection (with 1×1 conv downsample if channels differ)

- **Parameters**: ~5K
- **Loss**: `nn.BCELoss()` (sigmoid in forward — do NOT use BCEWithLogitsLoss)
- **Key property**: Causal convolutions prevent information leakage by construction

### XGBoost (`build_xgboost`)

```python
XGBClassifier(
    n_estimators=200,       # 300 in Stage 3-4 code, 200 in shared models.py
    max_depth=4,
    learning_rate=0.05,
    scale_pos_weight=...,   # (1 - y_mean) / y_mean
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=seed
)
```

**Note**: `n_estimators` differs between Stage 3-4 code (300) and the shared models.py (200). This is documented in the thesis.

### Logistic Regression

```python
LogisticRegression(
    class_weight='balanced',
    solver='lbfgs',
    C=1.0,
    max_iter=1000,
    random_state=seed
)
```

### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_features='sqrt',
    random_state=seed
)
```

## Evaluation Utilities (`eval_utils.py`)

### `walk_forward_split(df, train_frac=0.70, val_frac=0.15)`

Chronological split without shuffle:
- Train: first 70% of rows
- Validation: next 15%
- Test: remaining 15%

### `bootstrap_auc_ci(y_true, y_score, n_boot=1000, alpha=0.05, seed=42)`

Bootstrap confidence interval for ROC-AUC:
1. Draw `n` samples with replacement (1,000 times)
2. Compute AUC on each resample (skip if only one class present)
3. Return point estimate + percentile CI [α/2, 1-α/2]

### `make_binary_target(df, horizon)`

Creates target variable: `1 if Close[t+horizon] > Close[t]`, else `0`.

## Seed Management Convention

```python
torch.manual_seed(seed)
np.random.seed(seed)
```

Applied at the start of every experiment. Stage 9 experiments iterate over multiple seeds (typically 5, 10, or 20).

## Data Flow Summary

```
InfoMoney API → JSON articles
                    ↓
            FinBERT-PT-BR inference
                    ↓
            5 daily sentiment features
                    ↓
Yahoo Finance → 11 price features ──→ Left join by date
                                          ↓
                                   Forward-fill (ffill)
                                          ↓
                                   16 features × N days
                                          ↓
                              30-day sliding windows
                                          ↓
                              Binary target (h=5 or h=21)
                                          ↓
                              Walk-forward or expanding-window split
                                          ↓
                              Model training + evaluation
                                          ↓
                              AUC + bootstrap CI
```
