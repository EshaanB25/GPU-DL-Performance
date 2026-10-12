"""
Multi-Branch Audio Deepfake Detector
======================================
Custom Architecture:

  INPUT AUDIO
      │
      ├─── RAW WAVEFORM ──→ 1D CNN Block
      ├─── LOG-MEL SPECTROGRAM ──→ 2D CNN Block
      └─── MFCC ──→ 2D CNN Block
                │
      FEATURE CONCATENATION (Fusion)
                │
      RESIDUAL FEATURE REFINEMENT (ResNet-style)
                │
      TEMPORAL MODELING (BiLSTM)
                │
      ┌─────────┴──────────┐
      │                    │
  AUTOENCODER          TEMPORAL FEATURE
  BRANCH               VECTOR
  (Recon Error)            │
      └─────────┬──────────┘
                │
      FEATURE + ERROR FUSION
                │
      REGULARIZATION (Dropout + BatchNorm)
                │
      FULLY CONNECTED LAYERS
                │
      OUTPUT (REAL / FAKE)

Dataset: ASVspoof 2019 LA / PA
  Download: https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset

  Dataset is heavily imbalanced (~1:9 real:fake).
  Handled via:
    1. Weighted CrossEntropyLoss (bonafide weighted 9×)
    2. EER as PRIMARY metric (ASVspoof benchmark standard)
    3. Accuracy reported last with imbalance warning

INSTALL DEPENDENCIES (run once):
  conda activate dl
  pip install matplotlib seaborn scikit-learn numpy librosa tqdm scipy

HOW TO RUN IN VS CODE:
  1. Set DATA_ROOT below to your ASVspoof 2019 folder
  2. Set TASK to "LA", "PA", or "both"
  3. Press ▶
"""

# ── GPU fix — must be BEFORE torch import ────────────────────────────────────
import os
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   ── USER SETTINGS ──                            ║
# ║   Edit ONLY this section, then press ▶                           ║
# ╚══════════════════════════════════════════════════════════════════╝

MODE = "train"
#   "train"   → train from scratch
#   "eval"    → evaluate saved checkpoint on eval split
#   "predict" → classify a single audio file

# Root folder that CONTAINS the LA/ and PA/ subfolders
# Example:  r"D:\SEM 6\datasets\asvspoof2019"
DATA_ROOT = r"C:\SEM 6\DL\DL_Project\Dataset_2"

# Task:  "LA" | "PA" | "both"
TASK = "LA"

BATCH_SIZE    = 32
EPOCHS        = 50
LEARNING_RATE = 1e-3
CHECKPOINT    = "best_multibranch_model.pt"

# Class imbalance weights
# ASVspoof 2019 LA training is ~1:9 real:fake
# real_weight=9.0 compensates for the imbalance
REAL_WEIGHT = 9.0
FAKE_WEIGHT = 1.0

# Only for MODE = "predict"
AUDIO_FILE = r"C:\path\to\your\audio.wav"

# ╔══════════════════════════════════════════════════════════════════╗
# ║                END OF USER SETTINGS                              ║
# ╚══════════════════════════════════════════════════════════════════╝


# ─────────────────────────────────────────────────────────────────────
#  ASVspoof 2019 protocol paths  (do not edit)
# ─────────────────────────────────────────────────────────────────────
_TASK_CFG = {
    "LA": {
        "train": {
            "audio": os.path.join("LA", "ASVspoof2019_LA_train", "flac"),
            "proto": os.path.join("LA", "ASVspoof2019_LA_cm_protocols",
                                  "ASVspoof2019.LA.cm.train.trn.txt"),
        },
        "dev": {
            "audio": os.path.join("LA", "ASVspoof2019_LA_dev", "flac"),
            "proto": os.path.join("LA", "ASVspoof2019_LA_cm_protocols",
                                  "ASVspoof2019.LA.cm.dev.trl.txt"),
        },
        "eval": {
            "audio": os.path.join("LA", "ASVspoof2019_LA_eval", "flac"),
            "proto": os.path.join("LA", "ASVspoof2019_LA_cm_protocols",
                                  "ASVspoof2019.LA.cm.eval.trl.txt"),
        },
    },
    "PA": {
        "train": {
            "audio": os.path.join("PA", "ASVspoof2019_PA_train", "flac"),
            "proto": os.path.join("PA", "ASVspoof2019_PA_cm_protocols",
                                  "ASVspoof2019.PA.cm.train.trn.txt"),
        },
        "dev": {
            "audio": os.path.join("PA", "ASVspoof2019_PA_dev", "flac"),
            "proto": os.path.join("PA", "ASVspoof2019_PA_cm_protocols",
                                  "ASVspoof2019.PA.cm.dev.trl.txt"),
        },
        "eval": {
            "audio": os.path.join("PA", "ASVspoof2019_PA_eval", "flac"),
            "proto": os.path.join("PA", "ASVspoof2019_PA_cm_protocols",
                                  "ASVspoof2019.PA.cm.eval.trl.txt"),
        },
    },
}


# ─────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE   = 16_000
DURATION      = 4.0
N_MFCC        = 40
N_MELS        = 64
N_FFT         = 512
HOP_LENGTH    = 160
WIN_LENGTH    = 400
MAX_FRAMES    = 128
RAW_LENGTH    = int(SAMPLE_RATE * DURATION)

BRANCH_DIM    = 128
BILSTM_HIDDEN = 128     # per direction → 256 total
AE_LATENT     = 64

SEED          = 42
USE_AMP       = True
NUM_WORKERS   = 0

torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True


# ═════════════════════════════════════════════════════════════════════
#  FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════
def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m
    if len(y) < RAW_LENGTH:
        y = np.pad(y, (0, RAW_LENGTH - len(y)))
    else:
        y = y[:RAW_LENGTH]
    return y.astype(np.float32)


def _fix_frames(feat, max_frames):
    if feat.shape[1] < max_frames:
        feat = np.pad(feat, ((0, 0), (0, max_frames - feat.shape[1])))
    else:
        feat = feat[:, :max_frames]
    return feat


def extract_features(y):
    """
    Returns three representations:
      1. Raw waveform         → (1, RAW_LENGTH)
      2. Log-Mel Spectrogram  → (1, N_MELS, MAX_FRAMES)
      3. MFCC                 → (1, N_MFCC, MAX_FRAMES)
    """
    raw = y[np.newaxis, :]

    mel     = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = _fix_frames(log_mel, MAX_FRAMES)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-9)
    log_mel = log_mel[np.newaxis, :, :]

    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH
    )
    mfcc = _fix_frames(mfcc, MAX_FRAMES)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
    mfcc = mfcc[np.newaxis, :, :]

    return (raw.astype(np.float32),
            log_mel.astype(np.float32),
            mfcc.astype(np.float32))


# ═════════════════════════════════════════════════════════════════════
#  MODEL BLOCKS
# ═════════════════════════════════════════════════════════════════════

class RawWaveformBranch(nn.Module):
    """1D CNN on raw waveform → (B, BRANCH_DIM)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1,   32,  kernel_size=64, stride=4,  padding=32),
            nn.BatchNorm1d(32),  nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(32,  64,  kernel_size=32, stride=2,  padding=16),
            nn.BatchNorm1d(64),  nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(64,  128, kernel_size=16, stride=2,  padding=8),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.MaxPool1d(4),
            nn.Conv1d(128, BRANCH_DIM, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(BRANCH_DIM), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LogMelBranch(nn.Module):
    """2D CNN on Log-Mel Spectrogram → (B, BRANCH_DIM)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, BRANCH_DIM, kernel_size=3, padding=1),
            nn.BatchNorm2d(BRANCH_DIM), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class MFCCBranch(nn.Module):
    """2D CNN on MFCC → (B, BRANCH_DIM)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  32,  kernel_size=3, padding=1),
            nn.BatchNorm2d(32),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64,  kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, BRANCH_DIM, kernel_size=3, padding=1),
            nn.BatchNorm2d(BRANCH_DIM), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1).squeeze(-1)


class ResidualBlock(nn.Module):
    """ResNet-style skip connection — refines fused features."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
            nn.Linear(dim, dim), nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


class AutoencoderBranch(nn.Module):
    """
    Autoencoder for anomaly detection.
    Real audio → low reconstruction error.
    Fake audio → high reconstruction error.
    """
    def __init__(self, in_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
            nn.Linear(256, AE_LATENT), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(AE_LATENT, 256), nn.ReLU(inplace=True),
            nn.Linear(256, in_dim),
        )

    def forward(self, x):
        latent = self.encoder(x)
        recon  = self.decoder(latent)
        return recon, latent


# ═════════════════════════════════════════════════════════════════════
#  FULL MODEL
# ═════════════════════════════════════════════════════════════════════
class MultiBranchDeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()

        self.raw_branch    = RawWaveformBranch()
        self.logmel_branch = LogMelBranch()
        self.mfcc_branch   = MFCCBranch()

        fused_dim = BRANCH_DIM * 3   # 384

        self.res_block = ResidualBlock(fused_dim)

        self.bilstm = nn.LSTM(
            input_size=fused_dim, hidden_size=BILSTM_HIDDEN,
            num_layers=1, batch_first=True, bidirectional=True,
        )
        bilstm_out_dim = BILSTM_HIDDEN * 2   # 256

        self.autoencoder = AutoencoderBranch(in_dim=fused_dim)

        fusion_in = bilstm_out_dim + AE_LATENT + 1   # 321

        self.fc = nn.Sequential(
            nn.Linear(fusion_in, 256),
            nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, raw, log_mel, mfcc):
        f_raw    = self.raw_branch(raw)
        f_logmel = self.logmel_branch(log_mel)
        f_mfcc   = self.mfcc_branch(mfcc)

        fused = torch.cat([f_raw, f_logmel, f_mfcc], dim=1)  # (B, 384)
        fused = self.res_block(fused)

        lstm_out, _ = self.bilstm(fused.unsqueeze(1))
        temporal    = lstm_out.squeeze(1)                     # (B, 256)

        recon, latent = self.autoencoder(fused)
        recon_error   = F.mse_loss(
            recon, fused.detach(), reduction="none"
        ).mean(dim=1, keepdim=True)                           # (B, 1)

        combined = torch.cat([temporal, latent, recon_error], dim=1)
        return self.fc(combined)


# ═════════════════════════════════════════════════════════════════════
#  DATASET
# ═════════════════════════════════════════════════════════════════════
class ASVspoofMultiFeatureDataset(Dataset):
    def __init__(self, audio_dir, proto_file):
        self.samples = []
        self._parse(audio_dir, proto_file)

    def _parse(self, audio_dir, proto_file):
        if not os.path.isfile(proto_file):
            raise FileNotFoundError(
                f"\nProtocol file not found:\n  {proto_file}\n"
                f"Check DATA_ROOT='{DATA_ROOT}'\n"
                f"It should be the folder that CONTAINS LA/ and PA/"
            )
        missing = 0
        with open(proto_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                file_id = parts[1]
                label   = 0 if parts[4].lower() == "bonafide" else 1
                found   = False
                for ext in (".flac", ".wav"):
                    fp = os.path.join(audio_dir, file_id + ext)
                    if os.path.isfile(fp):
                        self.samples.append((fp, label))
                        found = True
                        break
                if not found:
                    missing += 1
        if missing:
            print(f"  [WARN] {missing} files in protocol not found on disk.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            y = load_audio(path)
            raw, log_mel, mfcc = extract_features(y)
        except Exception:
            raw     = np.zeros((1, RAW_LENGTH),         dtype=np.float32)
            log_mel = np.zeros((1, N_MELS, MAX_FRAMES), dtype=np.float32)
            mfcc    = np.zeros((1, N_MFCC, MAX_FRAMES), dtype=np.float32)
        return (
            torch.from_numpy(raw),
            torch.from_numpy(log_mel),
            torch.from_numpy(mfcc),
            label,
        )


def _make_split(task_key, split_name):
    cfg    = _TASK_CFG[task_key][split_name]
    a_dir  = os.path.join(DATA_ROOT, cfg["audio"])
    p_file = os.path.join(DATA_ROOT, cfg["proto"])
    ds     = ASVspoofMultiFeatureDataset(a_dir, p_file)
    n_real = sum(1 for _, l in ds.samples if l == 0)
    n_fake = sum(1 for _, l in ds.samples if l == 1)
    ratio  = n_fake / max(n_real, 1)
    print(f"  [{task_key}] {split_name:6s}: {len(ds):>7} files  "
          f"(bonafide={n_real}, spoof={n_fake}, ratio=1:{ratio:.1f})")
    return ds


def get_loaders():
    if TASK not in ("LA", "PA", "both"):
        raise ValueError(f"TASK must be 'LA', 'PA', or 'both'. Got: '{TASK}'")

    tasks = ["LA", "PA"] if TASK == "both" else [TASK]

    print(f"\n── Loading ASVspoof 2019  [{TASK}] ───────────────────────────")
    print(f"  ⚠  Dataset is imbalanced (~1:9 real:fake)")
    print(f"  ✔  Using weighted CrossEntropyLoss "
          f"(real={REAL_WEIGHT}×, fake={FAKE_WEIGHT}×)")
    print(f"  ✔  EER is primary metric (standard for ASVspoof)\n")

    tr_list, va_list, te_list = [], [], []
    for t in tasks:
        tr_list.append(_make_split(t, "train"))
        va_list.append(_make_split(t, "dev"))
        te_list.append(_make_split(t, "eval"))

    train_ds = ConcatDataset(tr_list)
    dev_ds   = ConcatDataset(va_list)
    eval_ds  = ConcatDataset(te_list)

    print(f"\n  Total train : {len(train_ds)}")
    print(f"  Total dev   : {len(dev_ds)}")
    print(f"  Total eval  : {len(eval_ds)}\n")

    kw = dict(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"), persistent_workers=False,
    )
    train_dl = DataLoader(train_ds, shuffle=True,  **kw)
    dev_dl   = DataLoader(dev_ds,   shuffle=False, **kw)
    eval_dl  = DataLoader(eval_ds,  shuffle=False, **kw)
    return train_dl, dev_dl, eval_dl


# ═════════════════════════════════════════════════════════════════════
#  EER  (primary metric)
# ═════════════════════════════════════════════════════════════════════
def compute_eer(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    eer = ((fpr[idx] + fnr[idx]) / 2) * 100
    return eer, thresholds[idx]


# ═════════════════════════════════════════════════════════════════════
#  EVAL PASS  (used by both training loop and standalone eval)
# ═════════════════════════════════════════════════════════════════════
def _eval_pass(model, dl, criterion, amp_on):
    """
    Full forward pass over a DataLoader.
    FIX: output is cast to float32 before loss to avoid
         AMP half-precision conflict with weighted CrossEntropyLoss.
    """
    model.eval()
    tl, tc, tt = 0.0, 0, 0
    preds, labels, probs = [], [], []

    with torch.no_grad():
        for raw, log_mel, mfcc, y in dl:
            raw     = raw    .to(DEVICE)
            log_mel = log_mel.to(DEVICE)
            mfcc    = mfcc   .to(DEVICE)
            y       = y      .to(DEVICE)

            with autocast(enabled=amp_on):
                out = model(raw, log_mel, mfcc)

            # ── FIX: cast to float32 before loss ──────────────────────
            # AMP produces float16 output inside autocast.
            # Weighted CrossEntropyLoss requires float32.
            out_f = out.float()

            tl += criterion(out_f, y).item() * y.size(0)
            tc += (out_f.argmax(1) == y).sum().item()
            tt += y.size(0)

            p = F.softmax(out_f, dim=1)[:, 1]
            preds .extend(out_f.argmax(1).cpu().tolist())
            labels.extend(y.cpu().tolist())
            probs .extend(p.cpu().tolist())

    return tl, tc, tt, preds, labels, probs


# ═════════════════════════════════════════════════════════════════════
#  TRAINING
# ═════════════════════════════════════════════════════════════════════
def train_model():
    print("\n── Training Multi-Branch Detector ───────────────────────────")
    train_dl, dev_dl, eval_dl = get_loaders()

    model = MultiBranchDeepfakeDetector().to(DEVICE)

    # Weighted loss — float32 explicit to avoid AMP dtype conflicts
    weights   = torch.tensor(
        [REAL_WEIGHT, FAKE_WEIGHT], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS
    )
    amp_on   = USE_AMP and DEVICE == "cuda"
    scaler   = GradScaler(enabled=amp_on)
    history  = {"tl": [], "ta": [], "vl": [], "va": [], "eer": []}
    best_eer = float("inf")

    for ep in range(1, EPOCHS + 1):
        model.train()
        tl, tc, tt = 0.0, 0, 0

        for raw, log_mel, mfcc, y in tqdm(
                train_dl, desc=f"Ep {ep:3d}/{EPOCHS}", leave=False):
            raw     = raw    .to(DEVICE)
            log_mel = log_mel.to(DEVICE)
            mfcc    = mfcc   .to(DEVICE)
            y       = y      .to(DEVICE)

            optimizer.zero_grad()

            with autocast(enabled=amp_on):
                out = model(raw, log_mel, mfcc)

            # ── FIX: cast to float32 before loss ──────────────────────
            out_f = out.float()
            loss  = criterion(out_f, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tl += loss.item() * y.size(0)
            tc += (out_f.argmax(1) == y).sum().item()
            tt += y.size(0)

        scheduler.step()

        vl, vc, vt, _, v_labels, v_probs = _eval_pass(
            model, dev_dl, criterion, amp_on
        )
        ta = tc / tt
        va = vc / vt
        try:
            v_eer, _ = compute_eer(v_labels, v_probs)
        except Exception:
            v_eer = 50.0

        history["tl"].append(tl / tt); history["ta"].append(ta)
        history["vl"].append(vl / vt); history["va"].append(va)
        history["eer"].append(v_eer)

        print(f"Ep {ep:3d}  "
              f"train loss={tl/tt:.4f} acc={ta:.4f}  "
              f"dev loss={vl/vt:.4f} acc={va:.4f}  "
              f"EER={v_eer:.2f}%")

        # Save best model based on EER (lower = better)
        if v_eer < best_eer:
            best_eer = v_eer
            torch.save(model.state_dict(), CHECKPOINT)
            print(f"        ✔ Best model saved  (dev_EER={v_eer:.2f}%)")

    # ── Final evaluation on eval split ────────────────────────────────
    print("\n── Final Evaluation on Eval Split ───────────────────────────")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    _, _, _, preds, labels, probs = _eval_pass(
        model, eval_dl, criterion, amp_on
    )
    _print_metrics(labels, preds, probs)
    _plot_curves(history)
    _plot_confusion_matrix(labels, preds)
    _plot_roc_curve(labels, probs)
    _plot_eer_curve(history)


# ═════════════════════════════════════════════════════════════════════
#  EVALUATE CHECKPOINT
# ═════════════════════════════════════════════════════════════════════
def eval_model():
    print(f"\n── Evaluating: {CHECKPOINT}  [{TASK}] ────────────────────────")
    _, _, eval_dl = get_loaders()
    model = MultiBranchDeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    weights   = torch.tensor(
        [REAL_WEIGHT, FAKE_WEIGHT], dtype=torch.float32
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    amp_on    = USE_AMP and DEVICE == "cuda"
    _, _, _, preds, labels, probs = _eval_pass(
        model, eval_dl, criterion, amp_on
    )
    _print_metrics(labels, preds, probs)
    _plot_confusion_matrix(labels, preds)
    _plot_roc_curve(labels, probs)


# ═════════════════════════════════════════════════════════════════════
#  SINGLE FILE PREDICTION
# ═════════════════════════════════════════════════════════════════════
def predict_single():
    print(f"\n── Predicting: {AUDIO_FILE} ─────────────────────────────────")
    if not os.path.isfile(AUDIO_FILE):
        print(f"ERROR: File not found → {AUDIO_FILE}"); return
    model = MultiBranchDeepfakeDetector().to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.eval()
    try:
        y = load_audio(AUDIO_FILE)
        raw, log_mel, mfcc = extract_features(y)
        raw_t    = torch.from_numpy(raw    ).unsqueeze(0).to(DEVICE)
        logmel_t = torch.from_numpy(log_mel).unsqueeze(0).to(DEVICE)
        mfcc_t   = torch.from_numpy(mfcc   ).unsqueeze(0).to(DEVICE)
    except Exception as e:
        print(f"ERROR loading audio: {e}"); return
    with torch.no_grad():
        out   = model(raw_t, logmel_t, mfcc_t)
        probs = F.softmax(out.float(), dim=1)[0]
    label = "FAKE (Spoof)" if probs[1].item() > 0.5 else "REAL (Bonafide)"
    conf  = max(probs[0].item(), probs[1].item()) * 100
    print(f"\n  Result      : {label}  (confidence {conf:.1f}%)")
    print(f"  P(Bonafide) : {probs[0].item():.4f}")
    print(f"  P(Spoof)    : {probs[1].item():.4f}")


# ═════════════════════════════════════════════════════════════════════
#  METRICS + PLOTS
# ═════════════════════════════════════════════════════════════════════
def _print_metrics(labels, preds, probs):
    print("\n" + "─" * 56)
    print(f"  METRICS  [Task={TASK}]")
    print("─" * 56)

    # PRIMARY: EER
    try:
        eer, threshold = compute_eer(labels, probs)
        print(f"\n  EER (primary)  : {eer:.2f}%")
        print(f"  EER threshold  : {threshold:.4f}")
        print(f"  → Standard metric for ASVspoof benchmark")
        print(f"  → Not affected by 1:9 class imbalance")
    except Exception:
        pass

    # SECONDARY: ROC-AUC
    try:
        auc = roc_auc_score(labels, probs)
        print(f"\n  ROC-AUC        : {auc:.4f}")
    except Exception:
        pass

    # TERTIARY: Accuracy (with imbalance warning)
    acc = accuracy_score(labels, preds)
    print(f"\n  Accuracy       : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ⚠  Treat with caution — dataset is ~1:9 imbalanced")
    print(f"     A model predicting all-fake gets ~90% accuracy")

    print("\n── Classification Report ────────────────────────────────────")
    print(classification_report(
        labels, preds, target_names=["Bonafide (Real)", "Spoof (Fake)"]
    ))
    print("─" * 56)


def _plot_curves(h):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(
        f"Multi-Branch Detector  [ASVspoof 2019 — {TASK}]", fontsize=12
    )
    axes[0].plot(h["tl"], label="Train"); axes[0].plot(h["vl"], label="Dev")
    axes[0].set_title("Loss (Weighted CE)")
    axes[0].set_xlabel("Epoch"); axes[0].legend()

    axes[1].plot(h["ta"], label="Train"); axes[1].plot(h["va"], label="Dev")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].legend()

    axes[2].plot(h["eer"], color="red", label="Dev EER")
    axes[2].set_title("EER %  (↓ lower is better)")
    axes[2].set_xlabel("Epoch"); axes[2].legend()

    plt.tight_layout()
    out = f"training_curves_multibranch_{TASK}.png"
    plt.savefig(out, dpi=150); plt.show()
    print(f"  Saved → {out}")


def _plot_confusion_matrix(labels, preds):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bonafide", "Spoof"],
                yticklabels=["Bonafide", "Spoof"])
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.title(f"Multi-Branch Confusion Matrix [{TASK}]")
    plt.tight_layout()
    out = f"confusion_matrix_multibranch_{TASK}.png"
    plt.savefig(out, dpi=150); plt.show()
    print(f"  Saved → {out}")


def _plot_roc_curve(labels, probs):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc  = roc_auc_score(labels, probs)
    fnr  = 1 - tpr
    idx  = np.argmin(np.abs(fpr - fnr))
    eer  = ((fpr[idx] + fnr[idx]) / 2) * 100
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="steelblue",
             label=f"ROC Curve (AUC = {auc:.2f})")
    plt.scatter(fpr[idx], tpr[idx], color="red", zorder=5,
                label=f"EER point ({eer:.2f}%)")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve — Multi-Branch  [{TASK}]")
    plt.legend(loc="lower right"); plt.tight_layout()
    out = f"roc_curve_multibranch_{TASK}.png"
    plt.savefig(out, dpi=150); plt.show()
    print(f"  Saved → {out}")


def _plot_eer_curve(h):
    plt.figure(figsize=(8, 4))
    plt.plot(h["eer"], color="red", linewidth=1.5, label="Dev EER %")
    plt.axhline(y=min(h["eer"]), color="darkred", linestyle="--",
                linewidth=0.8, label=f"Best EER: {min(h['eer']):.2f}%")
    plt.title(
        f"EER Progress  [ASVspoof 2019 — {TASK}]  ↓ lower is better"
    )
    plt.xlabel("Epoch"); plt.ylabel("EER %")
    plt.legend(); plt.tight_layout()
    out = f"eer_progress_multibranch_{TASK}.png"
    plt.savefig(out, dpi=150); plt.show()
    print(f"  Saved → {out}")


# ═════════════════════════════════════════════════════════════════════
#  ENTRY POINT  (if __name__ guard required on Windows)
# ═════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if DEVICE == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"GPU    : {p.name}")
        print(f"VRAM   : {p.total_memory / 1e9:.1f} GB")
        print(f"AMP    : {'enabled' if USE_AMP else 'disabled'}")
    else:
        print("No CUDA GPU — running on CPU.")

    model    = MultiBranchDeepfakeDetector()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : Multi-Branch (Raw+LogMel+MFCC → ResNet → BiLSTM → AE)")
    print(f"Params : {n_params:,}")
    print(f"Loss   : Weighted CrossEntropy "
          f"(real={REAL_WEIGHT}×, fake={FAKE_WEIGHT}×)  [float32]")
    print(f"Metric : EER (primary) — standard for ASVspoof benchmark")
    print(f"Task   : ASVspoof 2019  [{TASK}]  |  SR={SAMPLE_RATE} Hz")

    if   MODE == "train":   train_model()
    elif MODE == "eval":    eval_model()
    elif MODE == "predict": predict_single()
    else:
        print(f'ERROR: Unknown MODE "{MODE}". '
              f'Choose "train", "eval", or "predict".')