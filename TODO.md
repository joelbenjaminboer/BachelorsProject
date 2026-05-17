# Encoder Performance Enhancement Recommendations

## Context

The user's current setup achieves **~8° RMSE** on continuous knee-angle regression from 6-axis thigh IMU on the ENABL3S dataset using a transformer encoder with MAE pretraining. The goal is to identify high-leverage improvements across data, model, training, and post-processing.

**Current state observed (from exploration):**
- **Architecture:** d_model=128, num_heads=4, num_layers=3, ff_dim=512, dropout=0.1 → **~668k params** (note: user said ~5M — possibly a different runtime config; recommendations scale either way)
- **Inputs:** 6 channels, single thigh IMU, context=125 samples (1.25s @ 100 Hz), forecast_horizon=10 samples (0.1s)
- **Pretraining:** Contiguous-block MAE, mask ratio ~12–16%, MSE on masked positions
- **Fine-tuning:** 5 total epochs (2 frozen + 3 unfrozen), AdamW lr=1e-4, MSE loss, ReduceLROnPlateau (patience=5 — never triggers within 5 epochs)
- **No** data augmentation, **no** anti-aliasing before downsampling, **no** LR warmup, **no** early stopping, **no** gradient clipping, **no** mixed precision by default, **no** EMA, **no** test-time augmentation, **no** output smoothing

The list below is roughly ordered by expected impact-per-effort.

---

## 1. Data (highest leverage — model is starved, not over-parameterized)


### 1a. Anti-aliased downsampling [HIGH][turn on and off in config to confirm impact]
[src/data/preprocessing.py:100-102] uses `scipy.signal.resample()` (FFT-based) with no lowpass pre-filter when going 500 Hz → 100 Hz. Locomotion IMU has energy above 50 Hz (impacts, vibration) that aliases into the band you keep.
- Replace with `scipy.signal.decimate(x, q=5, ftype='iir', zero_phase=True)` (Chebyshev type I + filtfilt), or apply Butterworth lowpass at ~40 Hz before `resample`.
- Likely worth 0.3–0.8° RMSE on its own.

### 1b. Data augmentation [HIGH][MAYBE?][turn on and off in config to confirm impact]
There is **zero** augmentation. For a LOSO setup on 10 subjects this is the single biggest source of overfitting.
- **Gaussian jitter** on accel & gyro (σ ≈ 0.02 × channel std)
- **Magnitude scaling** (uniform [0.9, 1.1])
- **Time warping** (small DTW-style stretches, ±5%)
- **Sensor rotation augmentation** — apply a random small 3D rotation to the accel & gyro vectors (simulates IMU mounting variation, very common in IMU literature; e.g., Um et al. 2017). Especially impactful because LOSO test subjects have different sensor placements.
- **Channel dropout / masking at train time** (force model not to rely on a single axis).
Add these in [src/data/dataloader.py:35-56] inside `__getitem__`.

### 1e. Activity-stratified validation split [MEDIUM][]
[src/data/dataloader.py] does a random 90/10 within remaining subjects. Stratify by activity class so validation isn't dominated by level walking (which is the easiest).

### 1f. Output target normalization sanity-check [LOW-MEDIUM][]
Confirm `y_mean`/`y_std` are computed **after** target reshaping and only over the *flat* knee-angle distribution per fold. A leak here masks real generalization.

---

## 2. Model size & encoder layers

### 2a. Drop CLS token, use temporal pooling [MEDIUM]
[src/models/encoder.py:73] regresses from CLS only. For dense forecasting, CLS pooling forces 125 tokens of information through a single 128-d vector.
- **Better:** mean/attention-pool over the last K tokens of the encoder (e.g., last 25 tokens), or concat CLS with mean-pool.

### 2c. Rotary positional embeddings (RoPE) or ALiBi [MEDIUM]
The current "rotary-like sine/cosine" is additive sinusoidal PE (per exploration). True RoPE applied inside attention generalizes better to sequence-length changes and consistently outperforms additive sinusoidal in time-series transformers.

### 2f. PatchTST-style patch embedding [MEDIUM-HIGH]
Instead of 1 token per timestep, patch into windows of 8–16 samples. Lets you use longer context cheaply. Particularly well-validated for IMU/time-series regression.

---

## 3. Regression head

### 3c. Per-step heads vs. one head outputting H values [LOW-MEDIUM]
Currently one Linear outputs all H steps jointly. Try a small MLP per step with shared trunk, OR an autoregressive decoder with H≤10 steps (cheap at this length). For very short H this rarely matters; for longer H it does.

### 3d. GELU not ReLU [LOW]
Standard upgrade in transformer heads.

---

## 4. Training procedure (cheap wins, several stacked)

### 4g. Longer / better pretraining [HIGH][Set to percentage of window instead of absolute steps]
Mask ratio ~12–16% is way below MAE best practice. Vision/audio MAE typically uses 50–75%. Increase to **40–60%** with mixed random + block masking. Pretrain for many more epochs (you only need *unlabeled* IMU — even data from outside ENABL3S would help here, e.g., MotionSense, UCI HAR).

---

## 5. Post-processing & evaluation

### 5a. Per-activity error reporting [LOW] (for thesis quality, not RMSE itself)
You already do per-subject; add per-activity. This will reveal that stair errors are likely 2–3× walking errors, which then justifies the activity-stratified split (1f) and the mounting-rotation augmentation (1b).

---

## 6. Things you might be missing

- **Pretraining data leakage check:** confirm the LOSO held-out subject is also excluded from MAE pretraining — otherwise pretraining "sees" the test subject. Audit [src/training/pretrain.py] vs the fold setup.
- **Determinism:** seed `numpy`, `torch`, `torch.backends.cudnn.deterministic`. With <30 subjects, variance across runs may be > effect size you're trying to measure.
- **Report mean ± std over LOSO folds**, not just averaged RMSE. A 0.5° improvement is only meaningful if it exceeds across-fold std.

---

## Critical files for any of these changes

- [src/data/preprocessing.py](src/data/preprocessing.py) — anti-aliased downsample, context length
- [src/data/dataloader.py](src/data/dataloader.py) — augmentation in `__getitem__`, stratified val split
- [src/models/encoder.py](src/models/encoder.py) — pooling, RoPE, residual head, conv stem
- [src/models/factory.py](src/models/factory.py) — switch loss to Huber, build warmup-cosine scheduler
- [src/training/train.py](src/training/train.py) — early stopping, EMA, longer training
- [src/training/pretrain.py](src/training/pretrain.py) — higher mask ratio, more epochs
- [src/training/eval.py](src/training/eval.py) — overlap-average smoothing, per-activity metrics
- [conf/training/default.yaml](conf/training/default.yaml) — epochs, freeze schedule, context_length
- [conf/model/default.yaml](conf/model/default.yaml) — d_model, layers, loss, lr schedule
- [conf/gpu/default.yaml](conf/gpu/default.yaml) — `clip_grad_norm: 1.0`