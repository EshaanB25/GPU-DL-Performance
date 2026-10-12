"""Part 2c: does SAFE-Net's data pipeline starve the GPU?

SAFE-Net computes its inputs on the CPU inside Dataset.__getitem__: decode FLAC,
resample, mel spectrogram and MFCC with librosa. Its training script uses
NUM_WORKERS = 0, so the GPU waits while one CPU core prepares every batch.

This script trains SAFE-Net (BF16 autocast) for a fixed number of batches with
    - on-the-fly features with 0, 2, 4 and 8 DataLoader workers
    - features precomputed into memory (the GPU-only upper bound)
and reports samples/s and how much of each step the GPU spent waiting.

Usage:
    python dataloader_test.py --data-root "C:\\SEM 6\\DL\\DL_Project\\Dataset_2"
    python dataloader_test.py                 # no dataset: synthetic audio, same
                                              # librosa work, no disk decode
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

import common as C
import safenet_model as sn


class SyntheticAudio(Dataset):
    """Random 4 s waveforms run through SAFE-Net's real feature extraction."""
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(i)
        y = (rng.standard_normal(sn.RAW_LENGTH) * 0.1).astype(np.float32)
        raw, log_mel, mfcc = sn.extract_features(y)
        label = 0 if i % 10 == 0 else 1
        return torch.from_numpy(raw), torch.from_numpy(log_mel), torch.from_numpy(mfcc), label


def make_dataset(data_root, n_samples):
    if data_root:
        sn.DATA_ROOT = data_root
        cfg = sn._TASK_CFG["LA"]["train"]
        ds = sn.ASVspoofMultiFeatureDataset(os.path.join(data_root, cfg["audio"]),
                                            os.path.join(data_root, cfg["proto"]))
        idx = np.random.default_rng(0).permutation(len(ds))[:n_samples]
        return Subset(ds, idx.tolist()), "ASVspoof 2019 LA train (FLAC from disk)"
    return SyntheticAudio(n_samples), "synthetic audio (librosa features, no disk)"


def _one_thread_per_worker(_):
    # Each worker is its own process; without this every worker spins up a full
    # set of BLAS/OpenMP threads and they fight over the CPU cores.
    torch.set_num_threads(1)


def warmup_batches(workers):
    # Windows starts workers as fresh processes (re-import torch + librosa, and
    # librosa JIT-compiles on first use). Let every worker deliver a couple of
    # batches before timing so we measure steady state, not start-up.
    return 2 * workers + 1


def run_loader(ds, workers, pin, batch, n_batches, prec, device):
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers,
                    pin_memory=pin, persistent_workers=workers > 0, drop_last=True,
                    worker_init_fn=_one_thread_per_worker if workers > 0 else None)
    model = C.build_model(device)
    opt, loss_fn, scaler = C.make_training(model, prec, device)
    it = iter(dl)
    for _ in range(warmup_batches(workers)):
        b = [t.to(device, non_blocking=pin) for t in next(it)]
        C.train_step(model, b, opt, loss_fn, scaler, prec, device)
    C.sync(device)
    wait = 0.0
    t0 = time.perf_counter()
    for _ in range(n_batches):
        tw = time.perf_counter()
        b = next(it)                  # time blocked here = GPU starved of data
        wait += time.perf_counter() - tw
        b = [t.to(device, non_blocking=pin) for t in b]
        C.train_step(model, b, opt, loss_fn, scaler, prec, device)
    C.sync(device)
    total = time.perf_counter() - t0
    return total, wait


def run_precomputed(ds, batch, n_batches, prec, device):
    """Features already in GPU memory: what the model alone can sustain."""
    items = [ds[i] for i in range(batch)]
    data = [torch.stack([it[k] for it in items]).to(device) for k in range(3)]
    data.append(torch.tensor([it[3] for it in items], device=device))
    model = C.build_model(device)
    opt, loss_fn, scaler = C.make_training(model, prec, device)
    step = lambda: C.train_step(model, data, opt, loss_fn, scaler, prec, device)
    for _ in range(3):
        step()
    return C.timed_steps(step, n_batches, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=None, help="folder that contains LA/")
    p.add_argument("--batch", type=int, default=32, help="SAFE-Net trains with 32")
    p.add_argument("--batches", type=int, default=100, help="timed batches per setting")
    p.add_argument("--workers", type=int, nargs="+", default=[0, 2, 4, 8])
    p.add_argument("--precision", default="bf16", choices=list(C.PRECISIONS))
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="dataloader_results.csv")
    a = p.parse_args()

    device = C.pick_device(a.device)
    C.setup_backends(device)
    pin = device == "cuda"
    n = a.batch * (a.batches + warmup_batches(max(a.workers)) + 1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")   # inherited by spawned workers
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    ds, source = make_dataset(a.data_root, n)
    print(f"Device: {C.device_name(device)} | data: {source} | batch {a.batch} | "
          f"{a.batches} batches | {a.precision} | CPU cores: {os.cpu_count()}")

    rows = []
    t_gpu = run_precomputed(ds, a.batch, a.batches, a.precision, device)
    gpu_rate = a.batch * a.batches / t_gpu
    print(f"precomputed features   : {t_gpu:6.2f} s  {gpu_rate:8.1f} samples/s  (GPU-only bound)")
    rows.append({"setting": "precomputed", "workers": "", "seconds": round(t_gpu, 2),
                 "samples_per_s": round(gpu_rate, 1), "data_wait_share": "0%",
                 "vs_gpu_bound": "100%", "source": source})

    for w in a.workers:
        if w > (os.cpu_count() or 1) * 2:
            continue
        t, wait = run_loader(ds, w, pin, a.batch, a.batches, a.precision, device)
        rate = a.batch * a.batches / t
        rows.append({"setting": "on_the_fly", "workers": w, "seconds": round(t, 2),
                     "samples_per_s": round(rate, 1), "data_wait_share": f"{wait / t:.0%}",
                     "vs_gpu_bound": f"{rate / gpu_rate:.0%}", "source": source})
        print(f"num_workers={w:<2d} pin={pin!s:5s}: {t:6.2f} s  {rate:8.1f} samples/s  "
              f"waiting on data {wait / t:4.0%} of the time  ({rate / gpu_rate:.0%} of GPU bound)")

    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=rows[0].keys())
        wr.writeheader()
        wr.writerows(rows)
    print("Saved", a.out)


if __name__ == "__main__":   # required on Windows when num_workers > 0
    main()
