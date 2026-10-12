"""Part 1: SAFE-Net training throughput and memory across precision and batch size.

Synthetic inputs isolate the GPU: no disk, no librosa, so only the model's own
forward, backward and optimiser time is measured (Part 2c measures data loading).

Usage:
    python bench.py                                         # full sweep -> results.csv
    python bench.py --precision fp16 --batch 128 --steps 30 --nvtx   # one config (Nsight)
"""
import argparse
import csv
import os
import statistics
import subprocess
import time

import torch

import common as C

DEFAULT_BATCHES = [32, 64, 128, 256]


def gpu_status():
    """(temperature C, power W, SM clock MHz) from nvidia-smi, or Nones if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        vals = []
        for v in out.split(","):
            try:
                vals.append(float(v))
            except ValueError:
                vals.append(None)
        return tuple(vals[:3])
    except Exception:
        return (None, None, None)


def wait_until_cool(max_temp, cooldown):
    """Pause between configs so a laptop GPU never runs flat-out for long."""
    if cooldown:
        time.sleep(cooldown)
    t0 = time.time()
    while True:
        temp, power, _ = gpu_status()
        if temp is None or temp < max_temp - 10:
            return
        if time.time() - t0 > 300:
            print(f"  GPU still at {temp:.0f} C after 5 min of cooling - continuing")
            return
        print(f"  cooling down: GPU at {temp:.0f} C, waiting for < {max_temp - 10} C")
        time.sleep(10)


def run_config(precision, batch, device, steps, warmup, repeats, use_nvtx=False,
               channels_last=False, native_bn=False):
    model = C.build_model(device)
    data = C.synthetic_batch(batch, device)
    if channels_last:
        model, data = C.to_channels_last(model, data)
    if native_bn:
        C.use_native_batchnorm2d(model)
    opt, loss_fn, scaler = C.make_training(model, precision, device)
    step = lambda: C.train_step(model, data, opt, loss_fn, scaler, precision, device, use_nvtx)

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):            # one-time costs: allocation, cuDNN algorithm search
        step()
    # Several timed windows; the median resists laptop clock/thermal jitter
    rates = []
    for _ in range(repeats):
        with C.nvtx(f"{precision}_b{batch}", use_nvtx):
            elapsed = C.timed_steps(step, steps, device)
        rates.append(batch * steps / elapsed)
    tput = statistics.median(rates)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    spread = (max(rates) - min(rates)) / tput if len(rates) > 1 else 0.0
    del model, opt, data
    return tput, mem, spread


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=list(C.PRECISIONS), help="run one precision only")
    p.add_argument("--batch", type=int, help="run one batch size only")
    p.add_argument("--batches", type=int, nargs="+", default=DEFAULT_BATCHES)
    p.add_argument("--steps", type=int, default=30, help="timed steps per window")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3, help="timed windows per config")
    p.add_argument("--device", default=None)
    p.add_argument("--nvtx", action="store_true", help="add NVTX ranges for Nsight Systems")
    p.add_argument("--out", default=None,
                   help="default results.csv, or results_channels_last.csv with --channels-last")
    p.add_argument("--channels-last", action="store_true",
                   help="store 2D convolution weights and inputs in NHWC (channels_last)")
    p.add_argument("--native-bn", action="store_true",
                   help="run BatchNorm2d with PyTorch's kernel instead of cuDNN's")
    p.add_argument("--cooldown", type=float, default=15,
                   help="seconds to rest the GPU between configs (laptop safety)")
    p.add_argument("--max-temp", type=float, default=87,
                   help="stop the sweep if the GPU reaches this temperature (C)")
    a = p.parse_args()
    if a.out is None:
        a.out = "results" + ("_channels_last" if a.channels_last else "") \
                + ("_nativebn" if a.native_bn else "") + ".csv"

    device = C.pick_device(a.device)
    C.setup_backends(device)
    print(f"Device: {C.device_name(device)}"
          f"{'  |  memory format: channels_last (NHWC)' if a.channels_last else ''}"
          f"{'  |  BatchNorm2d: PyTorch native (cuDNN off)' if a.native_bn else ''}")
    print(f"Model : SAFE-Net multi-branch, {C.count_params(C.build_model('cpu')):,} params")

    precisions = [a.precision] if a.precision else list(C.PRECISIONS)
    batches = [a.batch] if a.batch else a.batches

    sweep = not (a.precision and a.batch)
    fields = ["device", "memory_format", "precision", "batch", "samples_per_s", "step_ms", "peak_mem_mb",
              "run_spread", "gpu_temp_c", "gpu_power_w", "sm_clock_mhz", "status"]
    if sweep:                          # write the header now; rows are appended as they finish
        with open(a.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()
    print(f"Safety: {a.cooldown:.0f} s rest between configs, stop at {a.max_temp:.0f} C. "
          f"Start temp: {gpu_status()[0]} C")

    rows = []
    too_hot = False
    for prec in precisions:
        oom_hit = False
        for b in batches:
            row = {k: "" for k in fields}
            row.update(device=C.device_name(device), precision=prec, batch=b,
                       memory_format=("channels_last" if a.channels_last else "nchw")
                                     + ("+native_bn" if a.native_bn else ""))
            if too_hot:
                row["status"] = "skipped (too hot)"
                rows.append(row)
                continue
            if device == "cuda" and sweep and rows:
                wait_until_cool(a.max_temp, a.cooldown)
            if oom_hit:                 # larger batches will not fit either
                row["status"] = "OOM"
                rows.append(row)
                print(f"{prec:5s} batch={b:5d}  skipped (smaller batch already OOM)")
                continue
            try:
                tput, mem, spread = run_config(prec, b, device, a.steps, a.warmup,
                                               a.repeats, a.nvtx, a.channels_last, a.native_bn)
                row.update(samples_per_s=round(tput, 1), step_ms=round(1000 * b / tput, 2),
                           peak_mem_mb=round(mem) if mem == mem else "",
                           run_spread=f"{spread:.1%}", status="ok")
                print(f"{prec:5s} batch={b:5d}  {tput:9.1f} samples/s  "
                      f"{1000*b/tput:7.2f} ms/step  peak {mem:7.0f} MB  (spread {spread:.1%})")
            except torch.cuda.OutOfMemoryError:
                oom_hit = True
                row["status"] = "OOM"
                print(f"{prec:5s} batch={b:5d}  out of memory")
            except RuntimeError as e:   # e.g. an op without a kernel for this dtype
                row["status"] = "error: " + str(e).splitlines()[0][:80]
                print(f"{prec:5s} batch={b:5d}  {row['status']}")
            if device == "cuda":
                temp, power, clock = gpu_status()     # reading right after the timed run
                row.update(gpu_temp_c=temp, gpu_power_w=power, sm_clock_mhz=clock)
                if temp is not None:
                    print(f"        GPU {temp:.0f} C, {power} W, SM clock {clock} MHz")
                    if temp >= a.max_temp:
                        too_hot = True
                        print(f"  STOPPING: GPU reached {temp:.0f} C (limit {a.max_temp:.0f} C). "
                              f"Results so far are saved.")
                torch.cuda.empty_cache()
            rows.append(row)
            if sweep:
                with open(a.out, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fields).writerow(row)

    if sweep:
        print("Saved", a.out)


if __name__ == "__main__":
    main()
