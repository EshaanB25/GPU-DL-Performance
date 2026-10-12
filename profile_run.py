"""Part 2a: where does SAFE-Net's GPU time go? PyTorch Profiler, FP32 vs FP16 vs BF16.

Produces, per precision:
    profile_<prec>.txt          top 20 operators by GPU time
    branch_breakdown_<prec>.csv forward GPU time per SAFE-Net block
    trace_<prec>.json           timeline -> open in https://ui.perfetto.dev

Usage:
    python profile_run.py                         # fp32, fp16, bf16 at batch 128
    python profile_run.py --precision fp16 --batch 256
"""
import argparse
import csv

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function

import common as C

# SAFE-Net blocks to label, in forward order
BLOCKS = ["raw_branch", "logmel_branch", "mfcc_branch", "res_block",
          "bilstm", "autoencoder", "fc"]


class Labeled(nn.Module):
    """Wraps a block so its forward shows up as a named range in the profile."""
    def __init__(self, inner, name):
        super().__init__()
        self.inner, self.name = inner, name

    def forward(self, *args, **kw):
        with record_function(f"SAFE::{self.name}"):
            return self.inner(*args, **kw)


def device_time_us(evt):
    for attr in ("device_time_total", "cuda_time_total"):   # name changed across versions
        v = getattr(evt, attr, None)
        if v:
            return v
    return 0.0


def profile_precision(prec, batch, device, steps, channels_last=False, native_bn=False):
    tag = prec + ("_cl" if channels_last else "") + ("_nbn" if native_bn else "")  # file suffix
    model = C.build_model(device)
    data = C.synthetic_batch(batch, device)
    if channels_last:
        model, data = C.to_channels_last(model, data)
    if native_bn:
        C.use_native_batchnorm2d(model)
    for name in BLOCKS:
        setattr(model, name, Labeled(getattr(model, name), name))
    opt, loss_fn, scaler = C.make_training(model, prec, device)
    step = lambda: C.train_step(model, data, opt, loss_fn, scaler, prec, device)

    for _ in range(10):                  # warm up so one-time setup isn't profiled
        step()
    C.sync(device)

    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with profile(activities=acts) as prof:
        for _ in range(steps):
            with record_function("SAFE::train_step"):
                step()
        C.sync(device)

    sort_key = "device_time_total" if device == "cuda" else "cpu_time_total"
    try:
        table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
    except (KeyError, AttributeError):   # older PyTorch
        sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
        table = prof.key_averages().table(sort_by=sort_key, row_limit=20)
    with open(f"profile_{tag}.txt", "w", encoding="utf-8") as f:
        f.write(f"SAFE-Net, {prec}, batch {batch}, {steps} steps, {C.device_name(device)}\n\n")
        f.write(table)
    prof.export_chrome_trace(f"trace_{tag}.json")

    # Work from the raw events. Each SAFE:: label appears twice: a CPU-side range
    # (whose device time = the GPU kernels launched inside it) and a GPU-side
    # annotation (a wall-clock span that can include gaps). Use the CPU-side one.
    from torch.autograd import DeviceType
    events = prof.events()
    on_gpu = device == "cuda"

    def dev_time(e):
        for attr in ("device_time_total", "cuda_time_total"):
            v = getattr(e, attr, None)
            if v:
                return v
        return 0.0

    if on_gpu:
        kernels = [e for e in events if e.device_type == DeviceType.CUDA
                   and not e.name.startswith(("SAFE::", "ProfilerStep"))]
        dur = lambda e: e.time_range.elapsed_us()
    else:
        kernels = [e for e in events if e.device_type == DeviceType.CPU
                   and not e.name.startswith(("SAFE::", "ProfilerStep"))
                   and e.cpu_parent is not None and e.cpu_parent.name.startswith("SAFE::")]
        dur = lambda e: e.cpu_time_total
    gpu_total_ms = sum(dur(e) for e in kernels) / 1000 / steps   # all GPU work per step

    # Top kernels by GPU time -> top_kernels_<prec>.csv
    by_name = {}
    for e in kernels:
        by_name[e.name] = by_name.get(e.name, 0.0) + dur(e)
    top = sorted(by_name.items(), key=lambda kv: -kv[1])[:15]
    with open(f"top_kernels_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "name", "time_ms_per_step", "share"])
        for i, (k, t) in enumerate(top, 1):
            ms = t / 1000 / steps
            w.writerow([i, k[:120], round(ms, 3), f"{ms / gpu_total_ms:.1%}" if gpu_total_ms else ""])

    # Layout-conversion kernels (NCHW <-> NHWC) -> layout_<tag>.csv
    conv_names = [k for k in by_name if "nchwToNhwc" in k or "nhwcToNchw" in k]
    layout_ms = sum(by_name[k] for k in conv_names) / 1000 / steps
    bn_names = [k for k in by_name if ("bn_" in k and "cudnn" in k) or "batch_norm" in k]
    bn_ms = sum(by_name[k] for k in bn_names) / 1000 / steps
    with open(f"layout_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["precision", "memory_format", "layout_conversion_ms_per_step",
                    "batchnorm_ms_per_step", "all_gpu_ms_per_step", "share", "conversion_kernel_types"])
        w.writerow([prec, ("channels_last" if channels_last else "nchw")
                    + ("+native_bn" if native_bn else ""), round(layout_ms, 3), round(bn_ms, 3),
                    round(gpu_total_ms, 3),
                    f"{layout_ms / gpu_total_ms:.1%}" if gpu_total_ms else "", len(conv_names)])

    # Forward GPU time per SAFE-Net block (CPU-side ranges only)
    rows = [{"precision": prec, "block": "all_gpu_work", "time_ms_per_step": round(gpu_total_ms, 3)}]
    for name in BLOCKS:
        evs = [e for e in events if e.name == f"SAFE::{name}" and e.device_type == DeviceType.CPU]
        if not evs:
            continue
        t = sum(dev_time(e) if on_gpu else e.cpu_time_total for e in evs)
        rows.append({"precision": prec, "block": name, "time_ms_per_step": round(t / 1000 / steps, 3)})
    fwd = sum(r["time_ms_per_step"] for r in rows[1:])
    rows.append({"precision": prec, "block": "backward_loss_adam",
                 "time_ms_per_step": round(gpu_total_ms - fwd, 3)})
    for r in rows:
        r["share_of_step"] = f"{r['time_ms_per_step'] / gpu_total_ms:.1%}" if gpu_total_ms else ""
    with open(f"branch_breakdown_{tag}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== {prec}  batch {batch}{'  channels_last' if channels_last else ''}{'  native-bn' if native_bn else ''} ===")
    print(table)
    print("GPU time per training step, by SAFE-Net block (forward) and the rest:")
    for r in rows:
        print(f"  {r['block']:20s} {r['time_ms_per_step']:9.3f} ms  {r['share_of_step']:>6s}")
    print(f"Batch-norm kernels: {bn_ms:.3f} ms/step" if gpu_total_ms else "")
    print(f"Layout conversion (NCHW<->NHWC) kernels: {layout_ms:.3f} ms/step "
          f"({layout_ms / gpu_total_ms:.1%} of GPU time)" if gpu_total_ms else "")
    print("Top 5 kernels by GPU time:")
    for i, (k, t) in enumerate(top[:5], 1):
        print(f"  {i}. {t / 1000 / steps:7.3f} ms  {k[:90]}")
    print(f"Saved profile_{tag}.txt, branch_breakdown_{tag}.csv, top_kernels_{tag}.csv, layout_{tag}.csv, trace_{tag}.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precision", choices=list(C.PRECISIONS) + ["all"], default="all")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--device", default=None)
    p.add_argument("--channels-last", action="store_true",
                   help="store 2D convolution weights and inputs in NHWC (channels_last)")
    p.add_argument("--native-bn", action="store_true",
                   help="run BatchNorm2d with PyTorch's kernel instead of cuDNN's")
    a = p.parse_args()

    device = C.pick_device(a.device)
    C.setup_backends(device)
    print("Device:", C.device_name(device))
    precs = list(C.PRECISIONS) if a.precision == "all" else [a.precision]
    for prec in precs:
        try:
            profile_precision(prec, a.batch, device, a.steps, a.channels_last, a.native_bn)
        except RuntimeError as e:
            print(f"{prec}: failed - {str(e).splitlines()[0]}")
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
