"""
=======================================================================
  AIT-204 Assignment: RNN Temperature Forecasting
  Dataset : Jena Climate 2009-2016
  Model   : Vanilla RNN (NumPy), trained with Truncated BPTT + Adam
=======================================================================
"""
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

# ══════════════════════════════════════════════════════════════════════
#  Configuration (Hyperparameters)
# ══════════════════════════════════════════════════════════════════════
SEQ_LEN       = 720    # 5 days × 144 readings/day (10-min intervals)
HIDDEN_SIZE   = 32     # Number of hidden units in the RNN layer
LEARNING_RATE = 0.001  # Adam optimizer learning rate
EPOCHS        = 30     # Number of training epochs
BATCH_SIZE    = 64     # Mini-batch size
BPTT_STEPS    = 50     # Truncated BPTT window (last N steps for gradients)
TRAIN_RATIO   = 0.8    # 80 % train / 20 % test (chronological split)
SAMPLE_STEP   = 10     # Use every 10th sequence to keep memory manageable
SEED          = 42

cwd = os.getcwd()
DATASET_PATH  = (f'{cwd}/dataset/jena_climate_2009_2016.csv')
SAVE_DIR      = f'{cwd}'

np.random.seed(SEED)
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  1. Data Loading
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  AIT-204 — RNN Temperature Forecasting (Jena Climate)")
print("=" * 65)

print("\n[1] Loading dataset …")
df = pd.read_csv(DATASET_PATH)
temperature = df['T (degC)'].values.astype(np.float64)

print(f"    Rows              : {len(df):,}")
print(f"    Columns           : {list(df.columns)}")
print(f"    Temperature range : {temperature.min():.1f} °C  →  "
      f"{temperature.max():.1f} °C  (mean = {temperature.mean():.1f} °C)")


# ══════════════════════════════════════════════════════════════════════
#  2. Normalization — Min-Max Scaling
# ══════════════════════════════════════════════════════════════════════
print("\n[2] Applying Min-Max normalization …")
T_min, T_max = temperature.min(), temperature.max()
temp_norm = (temperature - T_min) / (T_max - T_min)
print(f"    T_min = {T_min:.2f} °C,  T_max = {T_max:.2f} °C")
print(f"    Normalized range  : [{temp_norm.min():.4f}, {temp_norm.max():.4f}]")


# ══════════════════════════════════════════════════════════════════════
#  3. Sequence Creation — Sliding Window
# ══════════════════════════════════════════════════════════════════════
print(f"\n[3] Creating sequences  (window = {SEQ_LEN}, sample_step = {SAMPLE_STEP}) …")

n_possible = len(temp_norm) - SEQ_LEN
split_idx  = int(n_possible * TRAIN_RATIO)

train_idx  = range(0,          split_idx,  SAMPLE_STEP)
test_idx   = range(split_idx,  n_possible, SAMPLE_STEP)

# Build arrays  — shape: (N, SEQ_LEN, 1)  and  (N, 1)
X_train = np.array([temp_norm[i : i + SEQ_LEN] for i in train_idx],
                   dtype=np.float64)[:, :, np.newaxis]
y_train = np.array([temp_norm[i + SEQ_LEN]     for i in train_idx],
                   dtype=np.float64)[:, np.newaxis]

X_test  = np.array([temp_norm[i : i + SEQ_LEN] for i in test_idx],
                   dtype=np.float64)[:, :, np.newaxis]
y_test  = np.array([temp_norm[i + SEQ_LEN]     for i in test_idx],
                   dtype=np.float64)[:, np.newaxis]

print(f"    Training samples  : {X_train.shape[0]:,}")
print(f"    Test samples      : {X_test.shape[0]:,}")
print(f"    X_train shape     : {X_train.shape}  →  (samples, time_steps, features)")
print(f"    y_train shape     : {y_train.shape}  →  (samples, 1)")


# ══════════════════════════════════════════════════════════════════════
#  4. RNN Model  (NumPy Vanilla RNN)
# ══════════════════════════════════════════════════════════════════════
class VanillaRNN:
    """
    Single-layer Vanilla RNN for sequence-to-one regression.

    Recurrence:
        pre_h  = X[:,t,:] @ W_xh + h @ W_hh + b_h      (preactivation)
        h      = tanh(pre_h)                             (hidden state)

    Output (final hidden state only):
        y_pred = h_T @ W_hy + b_y

    Training uses Truncated BPTT (last `bptt_steps` time steps) and the
    Adam optimiser with gradient clipping to stabilise learning.
    """

    def __init__(self, input_size: int, hidden_size: int,
                 output_size: int, lr: float = 0.001):
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr          = lr

        # ── Xavier / Glorot initialisation ──────────────────────────
        s_xh = np.sqrt(2.0 / (input_size  + hidden_size))
        s_hh = np.sqrt(2.0 / (hidden_size + hidden_size))
        s_hy = np.sqrt(2.0 / (hidden_size + output_size))

        self.W_xh = np.random.randn(input_size,  hidden_size) * s_xh
        self.W_hh = np.random.randn(hidden_size, hidden_size) * s_hh
        self.b_h  = np.zeros((1, hidden_size))
        self.W_hy = np.random.randn(hidden_size, output_size) * s_hy
        self.b_y  = np.zeros((1, output_size))

        # ── Adam state ──────────────────────────────────────────────
        self._adam_t = 0
        self._adam_m = {k: np.zeros_like(v) for k, v in self._params.items()}
        self._adam_v = {k: np.zeros_like(v) for k, v in self._params.items()}

    # ── helpers ─────────────────────────────────────────────────────
    @property
    def _params(self):
        return dict(W_xh=self.W_xh, W_hh=self.W_hh, b_h=self.b_h,
                    W_hy=self.W_hy, b_y=self.b_y)

    # ── forward pass ────────────────────────────────────────────────
    def forward(self, X: np.ndarray, bptt_steps: int = 50):
        """
        Parameters
        ----------
        X           : (batch, seq_len, input_size)
        bptt_steps  : store only the last N hidden states for backprop

        Returns
        -------
        y_pred  : (batch, output_size)
        cache   : dict needed by backward()
        """
        batch, seq_len, _ = X.shape
        start = max(0, seq_len - bptt_steps)

        # Precompute input projections for ALL time steps — vectorised
        XW = X @ self.W_xh + self.b_h          # (batch, seq_len, hidden)

        h      = np.zeros((batch, self.hidden_size))
        h_prev = np.zeros((batch, self.hidden_size))   # state before window
        h_list = []
        h_tm1_init = None

        for t in range(seq_len):
            h = np.tanh(XW[:, t, :] + h @ self.W_hh)
            if t == start - 1:
                h_tm1_init = h.copy()          # h just before the BPTT window
            if t >= start:
                h_list.append(h.copy())

        y_pred = h @ self.W_hy + self.b_y       # (batch, output_size)

        cache = {
            'h_list'    : h_list,              # list of len <= bptt_steps
            'h_tm1_init': h_tm1_init if h_tm1_init is not None
                          else np.zeros((batch, self.hidden_size)),
            'XW_window' : XW[:, start:, :],    # input projections in window
        }
        return y_pred, cache

    # ── backward pass ───────────────────────────────────────────────
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray,
                 cache: dict) -> dict:
        """
        Truncated BPTT through the stored window.
        Loss = MSE  →  dL/dy_pred = 2*(y_pred - y_true) / batch
        """
        h_list     = cache['h_list']
        h_tm1_init = cache['h_tm1_init']
        XW_window  = cache['XW_window']
        batch      = y_pred.shape[0]
        bptt_len   = len(h_list)

        # ── output layer ────────────────────────────────────────────
        dL_dy  = 2.0 * (y_pred - y_true) / batch   # MSE derivative
        dW_hy  = h_list[-1].T @ dL_dy
        db_y   = dL_dy.sum(axis=0, keepdims=True)

        # ── propagate gradient into hidden layer ────────────────────
        dh     = dL_dy @ self.W_hy.T               # (batch, hidden)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)

        for t in reversed(range(bptt_len)):
            h_t   = h_list[t]
            h_tm1 = h_list[t - 1] if t > 0 else h_tm1_init

            dtanh  = (1.0 - h_t ** 2) * dh        # tanh back-prop
            dW_xh += XW_window[:, t, :].T @ dtanh  # note: XW = x@W_xh; dx handled implicitly
            # Correct gradient w.r.t W_xh requires original x, not xW
            # We'll compute it from h_list residual via chain rule below
            dW_hh += h_tm1.T @ dtanh
            db_h  += dtanh.sum(axis=0, keepdims=True)
            dh     = dtanh @ self.W_hh.T

        # Fix dW_xh: we stored XW not X, so recompute properly
        # XW[:,t,:] = X[:,t,:] @ W_xh + b_h  →  d(XW)/dW_xh needs X
        # We approximate here by noting that dXW = dtanh (already summed above),
        # but store X[:,t,:] is unavailable in this compact cache.
        # Instead, use the relationship: dW_xh = X.T @ dtanh_sum
        # Re-run a mini forward to get X window (fast, no storage needed)
        # For correctness, pass X into cache as well:
        dW_xh = cache.get('dW_xh_override', dW_xh)

        grads = dict(W_xh=dW_xh, W_hh=dW_hh, b_h=db_h,
                     W_hy=dW_hy, b_y=db_y)

        # ── gradient clipping (max global norm) ─────────────────────
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        clip_val   = 5.0
        if total_norm > clip_val:
            scale  = clip_val / (total_norm + 1e-8)
            grads  = {k: v * scale for k, v in grads.items()}

        return grads

    # ── Adam update ─────────────────────────────────────────────────
    def _adam_step(self, grads: dict,
                   beta1: float = 0.9, beta2: float = 0.999,
                   eps: float = 1e-8):
        self._adam_t += 1
        params = self._params
        for k in params:
            g = grads[k]
            self._adam_m[k] = beta1 * self._adam_m[k] + (1 - beta1) * g
            self._adam_v[k] = beta2 * self._adam_v[k] + (1 - beta2) * g**2
            m_hat = self._adam_m[k] / (1 - beta1 ** self._adam_t)
            v_hat = self._adam_v[k] / (1 - beta2 ** self._adam_t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + eps)

    # ── one training epoch ──────────────────────────────────────────
    def train_epoch(self, X: np.ndarray, y: np.ndarray,
                    batch_size: int = 64,
                    bptt_steps: int = 50) -> float:
        n       = len(X)
        indices = np.random.permutation(n)
        total_loss, n_batches = 0.0, 0

        for start in range(0, n, batch_size):
            idx       = indices[start : start + batch_size]
            Xb, yb    = X[idx], y[idx]

            y_pred, cache = self.forward(Xb, bptt_steps=bptt_steps)
            loss          = np.mean((y_pred - yb) ** 2)
            total_loss   += loss
            n_batches    += 1

            # Store X window in cache for correct dW_xh
            _, seq_len, _ = Xb.shape
            win_start = max(0, seq_len - bptt_steps)
            Xb_win    = Xb[:, win_start:, :]
            # Recompute dW_xh properly using stored X
            grads = self._compute_grads_with_X(y_pred, yb, cache, Xb_win)
            self._adam_step(grads)

        return total_loss / max(n_batches, 1)

    def _compute_grads_with_X(self, y_pred, y_true, cache, X_win):
        """Recompute all gradients with access to the raw input window."""
        h_list     = cache['h_list']
        h_tm1_init = cache['h_tm1_init']
        batch      = y_pred.shape[0]
        bptt_len   = len(h_list)

        dL_dy  = 2.0 * (y_pred - y_true) / batch
        dW_hy  = h_list[-1].T @ dL_dy
        db_y   = dL_dy.sum(axis=0, keepdims=True)

        dh    = dL_dy @ self.W_hy.T
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h  = np.zeros_like(self.b_h)

        for t in reversed(range(bptt_len)):
            h_t   = h_list[t]
            h_tm1 = h_list[t - 1] if t > 0 else h_tm1_init
            x_t   = X_win[:, t, :]

            dtanh  = (1.0 - h_t ** 2) * dh
            dW_xh += x_t.T @ dtanh
            dW_hh += h_tm1.T @ dtanh
            db_h  += dtanh.sum(axis=0, keepdims=True)
            dh     = dtanh @ self.W_hh.T

        grads = dict(W_xh=dW_xh, W_hh=dW_hh, b_h=db_h,
                     W_hy=dW_hy, b_y=db_y)
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
        clip_val   = 5.0
        if total_norm > clip_val:
            scale = clip_val / (total_norm + 1e-8)
            grads = {k: v * scale for k, v in grads.items()}
        return grads

    # ── inference ───────────────────────────────────────────────────
    def predict(self, X: np.ndarray, batch_size: int = 128) -> np.ndarray:
        preds = []
        for start in range(0, len(X), batch_size):
            Xb         = X[start : start + batch_size]
            y_pred, _  = self.forward(Xb, bptt_steps=SEQ_LEN)
            preds.append(y_pred)
        return np.vstack(preds)

# ══════════════════════════════════════════════════════════════════════
#  5. Instantiate & Train
# ══════════════════════════════════════════════════════════════════════
print(f"\n[4] Instantiating RNN …")
model = VanillaRNN(input_size=1, hidden_size=HIDDEN_SIZE,
                   output_size=1, lr=LEARNING_RATE)
print(f"    Input size   : 1  (temperature at each time step)")
print(f"    Hidden size  : {HIDDEN_SIZE}")
print(f"    Output size  : 1  (next temperature)")
print(f"    BPTT window  : {BPTT_STEPS} steps")
print(f"    Optimiser    : Adam  (lr={LEARNING_RATE})")

total_params = (1 * HIDDEN_SIZE +           # W_xh
                HIDDEN_SIZE * HIDDEN_SIZE + # W_hh
                HIDDEN_SIZE +               # b_h
                HIDDEN_SIZE * 1 +           # W_hy
                1)                          # b_y
print(f"    Total params : {total_params:,}")

print(f"\n[5] Training for {EPOCHS} epochs …")
print(f"    {'Epoch':>6}  {'Train MSE':>12}  {'Time/epoch':>12}")
print(f"    {'-'*6}  {'-'*12}  {'-'*12}")

train_losses = []
t_start = time.time()

for epoch in range(1, EPOCHS + 1):
    t0   = time.time()
    loss = model.train_epoch(X_train, y_train,
                             batch_size=BATCH_SIZE,
                             bptt_steps=BPTT_STEPS)
    elapsed = time.time() - t0
    train_losses.append(loss)
    if epoch == 1 or epoch % 5 == 0:
        print(f"    {epoch:>6}  {loss:>12.6f}  {elapsed:>10.2f}s")


total_time = time.time() - t_start
print(f"\n    Total training time : {total_time:.1f}s")

# Save model weights
weights = {
    "W_xh": model.W_xh.tolist(),
    "W_hh": model.W_hh.tolist(),
    "b_h": model.b_h.flatten().tolist(),
    "W_hy": model.W_hy.tolist(),
    "b_y": model.b_y.flatten().tolist()
}

weights_json = json.dumps(weights)
with(open("rnn-app/app/api/predict/weights.json", "w")) as f:
    f.write(weights_json)

# ══════════════════════════════════════════════════════════════════════
#  6. Evaluation
# ══════════════════════════════════════════════════════════════════════
print("\n[6] Evaluating on test set …")
y_pred_norm = model.predict(X_test, batch_size=128)

# Inverse Min-Max transform → degrees Celsius
y_pred_c = y_pred_norm * (T_max - T_min) + T_min
y_true_c = y_test       * (T_max - T_min) + T_min

rmse = np.sqrt(np.mean((y_pred_c - y_true_c) ** 2))
mae  = np.mean(np.abs(y_pred_c  - y_true_c))
mse  = np.mean((y_pred_c - y_true_c) ** 2)

print(f"    Test MSE  : {mse:.4f}")
print(f"    Test RMSE : {rmse:.4f} °C")
print(f"    Test MAE  : {mae:.4f} °C")


# ══════════════════════════════════════════════════════════════════════
#  7. Visualisation
# ══════════════════════════════════════════════════════════════════════
print("\n[7] Generating visualisations …")

fig, axes = plt.subplots(2, 1, figsize=(13, 10))
fig.suptitle(
    "RNN Temperature Forecasting — Jena Climate Dataset\n"
    f"Hidden={HIDDEN_SIZE}, LR={LEARNING_RATE}, Epochs={EPOCHS}, "
    f"SeqLen={SEQ_LEN}, BPTT={BPTT_STEPS}",
    fontsize=13, fontweight='bold'
)

# ── (a) Training loss curve ──────────────────────────────────────────
ax1 = axes[0]
ax1.plot(range(1, EPOCHS + 1), train_losses,
         color='steelblue', linewidth=2, marker='o', markersize=4)
ax1.set_title("Training Loss (MSE) over Epochs", fontsize=11)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("MSE Loss (normalised scale)")
ax1.set_yscale('log')
ax1.grid(True, alpha=0.35)
ax1.set_xlim(1, EPOCHS)
# Annotate final loss
ax1.annotate(f"Final: {train_losses[-1]:.5f}",
             xy=(EPOCHS, train_losses[-1]),
             xytext=(-60, 12), textcoords='offset points',
             fontsize=9, color='steelblue',
             arrowprops=dict(arrowstyle='->', color='steelblue'))

# ── (b) Actual vs Predicted ─────────────────────────────────────────
ax2 = axes[1]
n_plot = min(500, len(y_true_c))
x_axis = np.arange(n_plot)
ax2.plot(x_axis, y_true_c[:n_plot].ravel(),
         color='royalblue', linewidth=1.4, label='Actual Temperature', alpha=0.9)
ax2.plot(x_axis, y_pred_c[:n_plot].ravel(),
         color='tomato',    linewidth=1.4, label='Predicted Temperature',
         linestyle='--', alpha=0.9)
ax2.fill_between(x_axis,
                 y_true_c[:n_plot].ravel(),
                 y_pred_c[:n_plot].ravel(),
                 alpha=0.15, color='orange', label='Error')
ax2.set_title(
    f"Actual vs. Predicted Temperature — first {n_plot} test samples\n"
    f"RMSE = {rmse:.3f} °C  |  MAE = {mae:.3f} °C",
    fontsize=11
)
ax2.set_xlabel("Test Sample Index")
ax2.set_ylabel("Temperature (°C)")
ax2.legend(loc='upper right', fontsize=9)
ax2.grid(True, alpha=0.35)

plt.tight_layout()
fig_path = os.path.join(SAVE_DIR, 'rnn_results.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {fig_path}")


# ── Scatter plot: Predicted vs Actual ───────────────────────────────
fig2, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true_c.ravel(), y_pred_c.ravel(),
           alpha=0.2, s=8, color='steelblue', label='Predictions')
# Perfect-prediction line
lim = [min(y_true_c.min(), y_pred_c.min()) - 1,
       max(y_true_c.max(), y_pred_c.max()) + 1]
ax.plot(lim, lim, 'r--', linewidth=1.5, label='Perfect prediction')
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_title(f"Predicted vs. Actual Temperature\nRMSE={rmse:.3f} °C", fontsize=11)
ax.set_xlabel("Actual Temperature (°C)")
ax.set_ylabel("Predicted Temperature (°C)")
ax.legend(); ax.grid(True, alpha=0.35)
fig2.tight_layout()
scatter_path = os.path.join(SAVE_DIR, 'rnn_scatter.png')
plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"    Saved: {scatter_path}")


# ══════════════════════════════════════════════════════════════════════
#  Summary
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  SUMMARY")
print("=" * 65)
print(f"  Dataset         : Jena Climate 2009-2016  ({len(df):,} rows)")
print(f"  Feature used    : T (degC)  — temperature only")
print(f"  Sequence length : {SEQ_LEN} steps  (= 5 days × 144 readings/day)")
print(f"  Normalisation   : Min-Max  [{T_min:.2f} °C, {T_max:.2f} °C] → [0, 1]")
print(f"  Train / Test    : {X_train.shape[0]:,}  /  {X_test.shape[0]:,} sequences")
print(f"  Model           : Vanilla RNN  (hidden={HIDDEN_SIZE})")
print(f"  Optimiser       : Adam  (lr={LEARNING_RATE})")
print(f"  Epochs          : {EPOCHS}  (BPTT window={BPTT_STEPS} steps)")
print(f"  ── Results ─────────────────────────────────────")
print(f"  Test MSE        : {mse:.4f}")
print(f"  Test RMSE       : {rmse:.4f} °C")
print(f"  Test MAE        : {mae:.4f} °C")
print(f"  Training time   : {total_time:.1f} s")
print(f"  Outputs saved to: {SAVE_DIR}/")
print("=" * 65)
