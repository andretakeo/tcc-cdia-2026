import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

# ── TCN Architecture ────────────────────────────────────────────────────────
class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = padding
        self.chomp2 = padding
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        if self.chomp1 > 0: out = out[:, :, :-self.chomp1]
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        if self.chomp2 > 0: out = out[:, :, :-self.chomp2]
        out = self.relu(out)
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNClassifier(nn.Module):
    def __init__(self, input_size, num_channels=[32, 32], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            layers.append(CausalConv1dBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(num_channels[-1], 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid())

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.network(x)
        x = x.mean(dim=2)
        return self.head(x).squeeze(1)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# ── Training Logic ──────────────────────────────────────────────────────────
def prepare_sequences(df, features, window=30, horizon=21):
    X_raw = df[features].values
    # Target: Close[t+horizon] > Close[t]
    y_raw = (df['close'].shift(-horizon) > df['close']).astype(int).values
    
    X_seq, y_seq = [], []
    for i in range(window, len(df) - horizon):
        X_seq.append(X_raw[i-window:i])
        y_seq.append(y_raw[i])
    return np.array(X_seq), np.array(y_seq)

def train_tcn(X_train, y_train, X_val, y_val, input_size, seed):
    torch.manual_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_ds = TimeSeriesDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    
    model = TCNClassifier(input_size=input_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    best_val_auc = 0
    for epoch in range(30):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_val, dtype=torch.float32).to(device)).cpu().numpy()
        auc = roc_auc_score(y_val, preds)
    return auc

# ── Main Scrutiny Loop ──────────────────────────────────────────────────────
def run_scrutiny():
    df = pd.read_csv('purified_itub4_dataset.csv', parse_dates=['date'])
    features = ['close', 'news_count', 'avg_sentiment', 'avg_logit_pos', 'avg_logit_neg']
    
    # 1. Baseline Inércia (Price only)
    df['inertia_pred'] = (df['close'] > df['close'].shift(21)).astype(int)
    y_true = (df['close'].shift(-21) > df['close']).astype(int)
    common = df.dropna().index
    baseline_auc = roc_auc_score(y_true.loc[common], df.loc[common, 'inertia_pred'])
    print(f"Baseline Inertia AUC: {baseline_auc:.4f}")

    # 2. TCN Scrutiny (Multi-seed)
    seeds = range(40, 60) # 20 seeds
    tcn_results = []
    
    # Simple temporal split for multi-seed stability test
    split = int(len(df) * 0.8)
    sc = StandardScaler()
    df_sc = df.copy()
    df_sc[features] = sc.fit_transform(df[features])
    
    X_train, y_train = prepare_sequences(df_sc.iloc[:split], features)
    X_test, y_test = prepare_sequences(df_sc.iloc[split-30:], features)

    print(f"Running TCN on {len(seeds)} seeds...")
    for seed in seeds:
        auc = train_tcn(X_train, y_train, X_test, y_test, len(features), seed)
        tcn_results.append(auc)
    
    avg_tcn = np.mean(tcn_results)
    std_tcn = np.std(tcn_results)
    print(f"TCN Purified (V2) AUC: {avg_tcn:.4f} (+/- {std_tcn:.4f})")

    # 3. Statistical Test
    # Compare TCN seed distribution against the constant Baseline
    # (Simplified: check how many seeds beat the baseline)
    beats = sum(1 for r in tcn_results if r > baseline_auc)
    print(f"Seeds beating baseline: {beats}/{len(seeds)} ({beats/len(seeds)*100:.1f}%)")

if __name__ == "__main__":
    run_scrutiny()
