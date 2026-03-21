"""
models/base_models.py
=====================
7 base learners:
  B1 - XGBoost              tabular gradient boosting
  B2 - LightGBM             fast leaf-wise gradient boosting
  B3 - CatBoost             categorical-aware gradient boosting (new)
  B4 - Random Forest        bagging ensemble, different bias to boosters
  B5 - CNN-BiLSTM           temporal pattern expert
  B6 - LSTM-Attention       key-event focus
  B7 - WaveNet              dilated causal CNN (new) - long context window

All PyTorch deep models - full native Windows CUDA.
GPU: CUDA -> MPS -> CPU.

All expose: .fit(X, y, X_val=None, y_val=None) / .predict_proba(X) / .save() / .load()
No Sigmoid in forward pass -- BCEWithLogitsLoss + torch.sigmoid at inference.
"""

import logging
import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DEVICE
# ─────────────────────────────────────────────

def get_device():
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            logger.info(f"[GPU] Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = torch.device("mps")
            logger.info("[GPU] Using Apple MPS")
        else:
            dev = torch.device("cpu")
            logger.info("[GPU] No GPU found - using CPU")
        return dev
    except ImportError:
        return None

_DEVICE = None
def _device():
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = get_device()
    return _DEVICE

# ─────────────────────────────────────────────
# BASE INTERFACE
# ─────────────────────────────────────────────

class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X, y, X_val=None, y_val=None) -> "BaseModel": ...

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray: ...

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved {self.name} -> {path}")

    @classmethod
    def load(cls, path: Path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded {obj.name} <- {path}")
        return obj

# =====================================================================
# B1 -- XGBoost
# =====================================================================

class XGBoostModel(BaseModel):
    name = "xgboost"
    def __init__(self, params):
        self.params = {k: v for k, v in params.items() if k != "use_label_encoder"}
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        from xgboost import XGBClassifier
        self.model = XGBClassifier(**self.params)
        kw = {}
        if X_val is not None and y_val is not None:
            kw["eval_set"] = [(X_val, y_val)]
            kw["verbose"] = False
        self.model.fit(X, y, **kw)
        logger.info(f"XGBoost trained on {len(y)} samples")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        return self.model.feature_importances_

# =====================================================================
# B2 -- LightGBM
# =====================================================================

class LightGBMModel(BaseModel):
    name = "lightgbm"
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        self.model = LGBMClassifier(**self.params)
        kw = {}
        if X_val is not None and y_val is not None:
            kw["eval_set"] = [(X_val, y_val)]
            kw["callbacks"] = [early_stopping(50, verbose=False), log_evaluation(-1)]
        self.model.fit(X, y, **kw)
        logger.info(f"LightGBM trained on {len(y)} samples")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        return self.model.feature_importances_

# =====================================================================
# B3 -- CatBoost  (new - handles feature interactions differently)
# =====================================================================

class CatBoostModel(BaseModel):
    name = "catboost"
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError:
            raise ImportError("pip install catboost")
        self.model = CatBoostClassifier(**self.params)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val)
        self.model.fit(X, y, eval_set=eval_set, use_best_model=(eval_set is not None))
        logger.info(f"CatBoost trained on {len(y)} samples")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        return self.model.get_feature_importance()

# =====================================================================
# B4 -- Random Forest  (new - bagging diversity vs boosting)
# =====================================================================

class RandomForestModel(BaseModel):
    name = "random_forest"
    def __init__(self, params):
        self.params = params
        self.model = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(**self.params)
        self.model.fit(X, y)
        logger.info(f"RandomForest trained on {len(y)} samples")
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def get_feature_importance(self):
        return self.model.feature_importances_

# =====================================================================
# SHARED PYTORCH UTILITIES
# =====================================================================

def _to_sequences(X, y, seq_len):
    n, f = X.shape
    n_seq = n - seq_len
    if n_seq <= 0:
        raise ValueError(f"Not enough rows ({n}) for seq_len={seq_len}.")
    Xs = np.zeros((n_seq, seq_len, f), dtype=np.float32)
    ys = np.zeros(n_seq, dtype=np.float32)
    for i in range(n_seq):
        Xs[i] = X[i:i+seq_len]
        ys[i] = y[i+seq_len]
    return Xs, ys

def _to_sequences_infer(X, seq_len):
    n, f = X.shape
    n_seq = max(0, n - seq_len + 1)
    out = np.zeros((n_seq, seq_len, f), dtype=np.float32)
    for i in range(n_seq):
        out[i] = X[i:i+seq_len]
    return out

def _run_training(net, device, X_seq, y_seq, X_val_seq, y_val_seq,
                  lr, batch_size, epochs, patience):
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    net = net.to(device)
    # NO class weighting in loss -- probability calibration handles bias
    # (loss reweighting caused oscillation between 68% SELL and 94% BUY)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, epochs // 5), eta_min=lr * 0.01)

    Xt = torch.tensor(X_seq).to(device)
    yt = torch.tensor(y_seq).to(device)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=False)

    has_val = X_val_seq is not None and len(X_val_seq) > 0
    if has_val:
        Xv = torch.tensor(X_val_seq).to(device)
        yv = torch.tensor(y_val_seq).to(device)

    best_loss, best_state, no_improve = float("inf"), None, 0

    for epoch in range(epochs):
        net.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            criterion(net(xb).squeeze(-1), yb).backward()
            optimizer.step()
        scheduler.step()

        if has_val:
            net.eval()
            with torch.no_grad():
                val_loss = criterion(net(Xv).squeeze(-1), yv).item()
            if val_loss < best_loss - 1e-5:
                best_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state:
        net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return epoch + 1

def _prep_sequences(scaler, X, y, X_val, y_val, seq_len):
    Xs = scaler.transform(X)
    X_seq, y_seq = _to_sequences(Xs, y.astype(np.float32), seq_len)
    Xv_seq = yv_seq = None
    if X_val is not None and y_val is not None and len(X_val) > seq_len:
        Xvs = scaler.transform(X_val)
        Xv_seq, yv_seq = _to_sequences(Xvs, y_val.astype(np.float32), seq_len)
    return X_seq, y_seq, Xv_seq, yv_seq

def _infer_proba(net, scaler, X, seq_len, device):
    import torch
    Xs = scaler.transform(X)
    X_seq = _to_sequences_infer(Xs, seq_len)
    if len(X_seq) == 0:
        return np.full(len(X), 0.5, dtype=np.float32)
    net.eval()
    with torch.no_grad():
        logits = net(torch.tensor(X_seq).to(device)).squeeze(-1)
        out = torch.sigmoid(logits).cpu().numpy()
    pad = np.full(len(X) - len(out), 0.5, dtype=np.float32)
    return np.concatenate([pad, out])

class _DeepBase(BaseModel):
    def predict_proba(self, X):
        return _infer_proba(self.net, self.scaler, X, self._seq_len, _device())

# =====================================================================
# B5 -- CNN-BiLSTM  (module-level Net for pickle)
# =====================================================================

import torch.nn as nn

class _CNNBiLSTMNetModule(nn.Module):
    def __init__(self, n_feat, cnn_filters, cnn_kernel_size, lstm_units,
                 dense_units, dropout_rate, recurrent_dropout):
        super().__init__()
        cnn_layers, in_ch = [], n_feat
        for out_ch in cnn_filters:
            cnn_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=cnn_kernel_size,
                          padding=cnn_kernel_size // 2),
                nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(dropout_rate)]
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        n_layers = len(lstm_units)
        self.bilstm = nn.LSTM(in_ch, lstm_units[0], num_layers=n_layers,
                              batch_first=True, bidirectional=True,
                              dropout=recurrent_dropout if n_layers > 1 else 0.0)
        lstm_out = lstm_units[0] * 2
        dense_layers, d_in = [], lstm_out
        for d_out in dense_units:
            dense_layers += [nn.Linear(d_in, d_out), nn.GELU(), nn.Dropout(dropout_rate)]
            d_in = d_out
        dense_layers.append(nn.Linear(d_in, 1))   # raw logit
        self.head = nn.Sequential(*dense_layers)

    def forward(self, x):
        import torch
        x = self.cnn(x.permute(0, 2, 1)).permute(0, 2, 1)
        _, (h, _) = self.bilstm(x)
        return self.head(torch.cat([h[-2], h[-1]], dim=-1))

class CNNBiLSTMModel(_DeepBase):
    name = "cnn_bilstm"
    def __init__(self, params, n_features=1):
        self.params = params; self.n_features = n_features
        self._seq_len = params["sequence_length"]
        self.net = None; self.scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler
        p = self.params; self.n_features = X.shape[1]
        self.scaler = StandardScaler(); self.scaler.fit(X)
        self.net = _CNNBiLSTMNetModule(self.n_features, p["cnn_filters"],
            p["cnn_kernel_size"], p["lstm_units"], p["dense_units"],
            p["dropout_rate"], p.get("recurrent_dropout", 0.2))
        X_seq, y_seq, Xv, yv = _prep_sequences(self.scaler, X, y, X_val, y_val, self._seq_len)
        dev = _device()
        ep = _run_training(self.net, dev, X_seq, y_seq, Xv, yv,
                           p["learning_rate"], p["batch_size"], p["epochs"],
                           p["early_stopping_patience"])
        logger.info(f"CNN-BiLSTM {ep} epochs | {len(y_seq)} seq | device={dev}")
        return self

# =====================================================================
# B6 -- LSTM-Attention  (module-level)
# =====================================================================

class _LSTMAttnNetModule(nn.Module):
    def __init__(self, n_feat, lstm_units, attention_units, dropout_rate):
        super().__init__()
        n_layers = len(lstm_units)
        self.lstm = nn.LSTM(n_feat, lstm_units[0], num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout_rate if n_layers > 1 else 0.0)
        H = lstm_units[0]
        self.attn_fc = nn.Linear(H, attention_units)
        self.attn_v  = nn.Linear(attention_units, 1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Sequential(
            nn.Linear(H, 64), nn.GELU(), nn.Dropout(dropout_rate),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))  # raw logit

    def forward(self, x):
        import torch.nn.functional as F
        out, _ = self.lstm(x)
        weights = F.softmax(self.attn_v(F.tanh(self.attn_fc(out))), dim=1)
        return self.head(self.dropout((out * weights).sum(1)))

class LSTMAttentionModel(_DeepBase):
    name = "lstm_attention"
    def __init__(self, params, n_features=1):
        self.params = params; self.n_features = n_features
        self._seq_len = params["sequence_length"]
        self.net = None; self.scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler
        p = self.params; self.n_features = X.shape[1]
        self.scaler = StandardScaler(); self.scaler.fit(X)
        self.net = _LSTMAttnNetModule(self.n_features, p["lstm_units"],
                                      p.get("attention_units", 32), p["dropout_rate"])
        X_seq, y_seq, Xv, yv = _prep_sequences(self.scaler, X, y, X_val, y_val, self._seq_len)
        dev = _device()
        ep = _run_training(self.net, dev, X_seq, y_seq, Xv, yv,
                           p["learning_rate"], p["batch_size"], p["epochs"],
                           p["early_stopping_patience"])
        logger.info(f"LSTM-Attention {ep} epochs | {len(y_seq)} seq | device={dev}")
        return self

# =====================================================================
# B7 -- WaveNet-style Dilated Causal CNN  (new - long context window)
# =====================================================================

class _ResidualDilatedBlock(nn.Module):
    """WaveNet residual block: dilated causal conv + gated activation + skip."""
    def __init__(self, n_filters, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(n_filters, n_filters * 2,
                              kernel_size=kernel_size,
                              dilation=dilation, padding=pad)
        self.skip_conv  = nn.Conv1d(n_filters, n_filters, 1)
        self.res_conv   = nn.Conv1d(n_filters, n_filters, 1)
        self.drop       = nn.Dropout(dropout)
        self._pad       = pad

    def forward(self, x):
        import torch
        h = self.conv(x)
        h = h[:, :, :x.shape[2]]     # causal: strip future padding
        # Gated activation: tanh(a) * sigmoid(b)
        h_tanh  = torch.tanh(h[:, :h.shape[1]//2, :])
        h_sig   = torch.sigmoid(h[:, h.shape[1]//2:, :])
        h = self.drop(h_tanh * h_sig)
        return self.res_conv(h) + x,  self.skip_conv(h)

class _WaveNetModule(nn.Module):
    def __init__(self, n_feat, n_filters, dilations, kernel_size, dropout):
        super().__init__()
        self.input_conv = nn.Conv1d(n_feat, n_filters, 1)
        self.blocks = nn.ModuleList([
            _ResidualDilatedBlock(n_filters, kernel_size, d, dropout)
            for d in dilations
        ])
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_filters, n_filters, 1), nn.ReLU(),
            nn.Conv1d(n_filters, 1, 1))

    def forward(self, x):
        import torch
        # x: (B, T, F) -> (B, F, T)
        x = self.input_conv(x.permute(0, 2, 1))
        skip_sum = 0
        for block in self.blocks:
            x, skip = block(x)
            skip_sum = skip_sum + skip
        out = self.head(skip_sum)   # (B, 1, T)
        return out[:, :, -1]        # last time step logit (B, 1)

class WaveNetModel(_DeepBase):
    name = "wavenet"
    def __init__(self, params, n_features=1):
        self.params = params; self.n_features = n_features
        self._seq_len = params["sequence_length"]
        self.net = None; self.scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler
        p = self.params; self.n_features = X.shape[1]
        self.scaler = StandardScaler(); self.scaler.fit(X)
        self.net = _WaveNetModule(self.n_features, p["n_filters"],
                                  p["dilations"], p["kernel_size"], p["dropout_rate"])
        X_seq, y_seq, Xv, yv = _prep_sequences(self.scaler, X, y, X_val, y_val, self._seq_len)
        dev = _device()
        ep = _run_training(self.net, dev, X_seq, y_seq, Xv, yv,
                           p["learning_rate"], p["batch_size"], p["epochs"],
                           p["early_stopping_patience"])
        logger.info(f"WaveNet {ep} epochs | {len(y_seq)} seq | device={dev}")
        return self

    def predict_proba(self, X):
        # WaveNet outputs (B,1) not (B,) -- handle reshape
        import torch
        Xs = self.scaler.transform(X)
        X_seq = _to_sequences_infer(Xs, self._seq_len)
        if len(X_seq) == 0:
            return np.full(len(X), 0.5, dtype=np.float32)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.tensor(X_seq).to(_device())).squeeze()
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
            out = torch.sigmoid(logits).cpu().numpy()
        pad = np.full(len(X) - len(out), 0.5, dtype=np.float32)
        return np.concatenate([pad, out])

# =====================================================================
# TFT -- kept for compatibility (renamed to index 5 slot in old code)
# =====================================================================

class _GRNModule(nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, out_dim)
        self.fc2  = nn.Linear(out_dim, out_dim)
        self.gate = nn.Sequential(nn.Linear(in_dim, out_dim), nn.Sigmoid())
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        import torch
        h = self.drop(self.fc2(torch.relu(self.fc1(x))))
        return self.norm(self.skip(x) + self.gate(x) * h)

class _TFTNetModule(nn.Module):
    def __init__(self, n_feat, H, heads, drop, Hc):
        super().__init__()
        self.proj  = nn.Linear(n_feat, H)
        self.grn1  = _GRNModule(H, H, drop)
        self.attn  = nn.MultiheadAttention(H, num_heads=heads, dropout=drop, batch_first=True)
        self.norm1 = nn.LayerNorm(H)
        self.grn2  = _GRNModule(H, H, drop)
        self.lstm  = nn.LSTM(H, H, batch_first=True)
        self.norm2 = nn.LayerNorm(H)
        self.head  = nn.Sequential(nn.Linear(H, Hc), nn.GELU(),
                                   nn.Dropout(drop), nn.Linear(Hc, 1))

    def forward(self, x):
        x = self.grn1(self.proj(x))
        a, _ = self.attn(x, x, x)
        x = self.norm1(x + a)
        x = self.grn2(x)
        _, (h, _) = self.lstm(x)
        return self.head(self.norm2(h[-1]))

class TFTModel(_DeepBase):
    name = "tft"
    def __init__(self, params, backend="pytorch", n_features=1):
        self.params = params; self.backend = backend; self.n_features = n_features
        self._seq_len = params["sequence_length"]
        self.net = None; self.scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        from sklearn.preprocessing import StandardScaler
        p = self.params; self.n_features = X.shape[1]
        self.scaler = StandardScaler(); self.scaler.fit(X)
        self.net = _TFTNetModule(self.n_features, p["hidden_size"],
                                 p["attention_head_size"], p["dropout"],
                                 p["hidden_continuous_size"])
        X_seq, y_seq, Xv, yv = _prep_sequences(self.scaler, X, y, X_val, y_val, self._seq_len)
        dev = _device()
        ep = _run_training(self.net, dev, X_seq, y_seq, Xv, yv,
                           p["learning_rate"], p["batch_size"], p["max_epochs"], 15)
        logger.info(f"TFT {ep} epochs | {len(y_seq)} seq | device={dev}")
        return self

# =====================================================================
# FACTORY
# =====================================================================

def build_all_base_models(config) -> list:
    return [
        XGBoostModel(config.XGBOOST_PARAMS),
        LightGBMModel(config.LIGHTGBM_PARAMS),
        CatBoostModel(config.CATBOOST_PARAMS),
        RandomForestModel(config.RANDOM_FOREST_PARAMS),
        CNNBiLSTMModel(config.CNN_BILSTM_PARAMS),
        LSTMAttentionModel(config.LSTM_ATTENTION_PARAMS),
        WaveNetModel(config.WAVENET_PARAMS),
    ]
