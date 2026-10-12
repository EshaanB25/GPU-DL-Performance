"""Make every plot and fill in the results table from whatever CSVs exist.

    python report.py      -> *.png and RESULTS.md
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"fp32": "#6b7280", "fp16": "#2563eb", "bf16": "#d97706",
          "naive": "#9ca3af", "tiled": "#2563eb", "cublas": "#059669"}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "font.size": 10})
lines = ["# Results\n"]


def md(df, **kw):
    try:
        return df.to_markdown(**kw)
    except ImportError:                 # tabulate not installed
        return "```\n" + df.to_string(**kw) + "\n```"


def add(label, value, note=""):
    lines.append(f"| {label} | {value} | {note} |")


def part1():
    if not os.path.exists("results.csv"):
        return
    df = pd.read_csv("results.csv")
    dev = df["device"].iloc[0]
    ok = df[df["status"] == "ok"].copy()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for prec, g in ok.groupby("precision", sort=False):
        ax[0].plot(g["batch"], g["samples_per_s"], marker="o", label=prec, color=COLORS.get(prec))
        if g["peak_mem_mb"].notna().any():
            ax[1].plot(g["batch"], g["peak_mem_mb"], marker="o", label=prec, color=COLORS.get(prec))
    ax[0].set(title="SAFE-Net training throughput", xlabel="Batch size", ylabel="Samples / s")
    ax[1].set(title="Peak GPU memory", xlabel="Batch size", ylabel="MB")
    for a in ax:
        a.set_xscale("log", base=2)
        a.legend(frameon=False)
    fig.suptitle(dev, fontsize=9, color="#6b7280")
    fig.tight_layout()
    fig.savefig("throughput_memory.png", dpi=150)
    plt.close(fig)

    lines.append(f"\n## Part 1: precision x batch size ({dev})\n")
    lines.append(ok.pivot(index="batch", columns="precision", values="samples_per_s")
                 .pipe(md) + "\n\n*samples/s*\n")
    lines.append("| Placeholder | Value | Note |\n|---|---|---|")
    add("[A] smallest batch tested", int(df["batch"].min()))
    for prec in ["fp32", "fp16", "bf16"]:
        g = ok[ok["precision"] == prec]
        if len(g):
            oom = (df[(df["precision"] == prec)]["status"] == "OOM").any()
            add(f"[B] largest batch that fit, {prec}", int(g["batch"].max()),
                "next size ran out of memory" if oom else "no OOM in the sweep")
    base = ok[ok["precision"] == "fp32"].set_index("batch")
    for prec in ["fp16", "bf16"]:
        g = ok[ok["precision"] == prec].set_index("batch")
        common = g.index.intersection(base.index)
        if len(common) == 0:
            continue
        b = common.max()           # largest batch every precision ran at
        sp = g.loc[b, "samples_per_s"] / base.loc[b, "samples_per_s"]
        add(f"[X] {prec} speed-up over fp32", f"{sp:.2f}x", f"batch {b}")
        if pd.notna(g.loc[b, "peak_mem_mb"]) and pd.notna(base.loc[b, "peak_mem_mb"]):
            sv = 1 - g.loc[b, "peak_mem_mb"] / base.loc[b, "peak_mem_mb"]
            add(f"[Y] {prec} memory saving vs fp32", f"{sv:.0%}", f"batch {b}")
        best = max(g.loc[c, "samples_per_s"] / base.loc[c, "samples_per_s"] for c in common)
        add(f"{prec} best speed-up at any common batch", f"{best:.2f}x")


def part2a():
    files = sorted(glob.glob("branch_breakdown_*.csv"))
    if not files:
        return
    df = pd.concat(pd.read_csv(f) for f in files)
    blocks = df[df["block"] != "all_gpu_work"]
    precs = [p for p in ["fp32", "fp16", "bf16"] if p in set(df["precision"])]
    order = list(dict.fromkeys(blocks["block"]))
    fig, ax = plt.subplots(figsize=(8, 4))
    h = 0.8 / max(len(precs), 1)
    for i, p in enumerate(precs):
        g = blocks[blocks["precision"] == p].set_index("block").reindex(order)
        ax.barh([j + i * h for j in range(len(order))], g["time_ms_per_step"], height=h,
                label=p, color=COLORS.get(p))
    ax.set_yticks([j + h * (len(precs) - 1) / 2 for j in range(len(order))], order)
    ax.invert_yaxis()
    ax.set(title="SAFE-Net GPU time per training step", xlabel="ms per step (forward per block, then backward + loss + Adam)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("branch_breakdown.png", dpi=150)
    plt.close(fig)

    lines.append("\n## Part 2a: where the time goes\n")
    lines.append(df.pivot(index="block", columns="precision", values="time_ms_per_step")
                 .reindex(["all_gpu_work"] + order).pipe(md) + "\n\n*ms per step*\n")
    for p in precs:
        f = f"top_kernels_{p}.csv"
        if os.path.exists(f):
            k = pd.read_csv(f).head(3)
            lines.append(f"\n**Top 3 kernels, {p}:**\n")
            for _, r in k.iterrows():
                lines.append(f"{r['rank']}. `{r['name'][:90]}` — {r['time_ms_per_step']} ms/step ({r['share']})")
            lines.append("")


def part2c():
    if not os.path.exists("dataloader_results.csv"):
        return
    df = pd.read_csv("dataloader_results.csv")
    labels = ["precomputed" if s == "precomputed" else f"{int(w)} workers"
              for s, w in zip(df["setting"], df["workers"].fillna(-1))]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    cols = ["#059669" if l == "precomputed" else "#2563eb" for l in labels]
    ax.bar(labels, df["samples_per_s"], color=cols)
    for i, v in enumerate(df["samples_per_s"]):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    ax.set(title="SAFE-Net data pipeline vs GPU-only bound", ylabel="Samples / s")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig("dataloader.png", dpi=150)
    plt.close(fig)

    lines.append(f"\n## Part 2c: data loading ({df['source'].iloc[0]})\n")
    lines.append(md(df.drop(columns=["source"]), index=False) + "\n")
    fly = df[df["setting"] == "on_the_fly"]
    if len(fly):
        w0 = fly[fly["workers"] == 0]
        best = fly.loc[fly["samples_per_s"].idxmax()]
        gpu = df[df["setting"] == "precomputed"]["samples_per_s"].iloc[0]
        lines.append("| Placeholder | Value | Note |\n|---|---|---|")
        if len(w0):
            w0 = w0.iloc[0]
            add("0 workers (SAFE-Net default)", f"{w0['seconds']} s, {w0['samples_per_s']} samples/s",
                f"waiting on data {w0['data_wait_share']}")
            add("Best worker count", f"{int(best['workers'])} workers, {best['samples_per_s']} samples/s",
                f"{best['samples_per_s'] / w0['samples_per_s']:.1f}x faster than 0 workers")
        add("GPU-only bound (precomputed features)", f"{gpu} samples/s",
            f"best loader reaches {best['samples_per_s'] / gpu:.0%} of it")


def part3():
    if not os.path.exists("matmul_results.csv"):
        return
    df = pd.read_csv("matmul_results.csv")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ["naive", "tiled", "cublas"]:
        g = df[df["kernel"] == name]
        ax.plot(g["size"], g["gflops"], marker="o", label=name, color=COLORS[name])
    ax.set(title="FP32 matrix multiplication", xlabel="Matrix size N (N x N)", ylabel="GFLOPS")
    ax.set_xscale("log", base=2)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("matmul_gflops.png", dpi=150)
    plt.close(fig)

    piv = df.pivot(index="size", columns="kernel", values="gflops")
    lines.append("\n## Part 3: CUDA matmul vs cuBLAS\n")
    lines.append(md(piv) + "\n\n*GFLOPS*\n")
    lines.append("| Placeholder | Value | Note |\n|---|---|---|")
    t = df[df["kernel"] == "tiled"]
    add("[Z] best tiled GFLOPS", f"{t['gflops'].max():.0f}", f"N = {int(t.loc[t['gflops'].idxmax(), 'size'])}")
    n = piv.index.max()
    add(f"Tiled vs naive at N={n}", f"{piv.loc[n, 'tiled'] / piv.loc[n, 'naive']:.1f}x")
    add(f"Tiled as % of cuBLAS at N={n}", f"{piv.loc[n, 'tiled'] / piv.loc[n, 'cublas']:.0%}")
    err = df[df["kernel"] != "cublas"]["max_abs_error_vs_cublas"].astype(float).max()
    add("Max error vs cuBLAS", f"{err:.1e}", "OK" if err < 1e-2 else "CHECK FOR A BUG")


def part_channels_last():
    """Default NCHW vs channels_last vs channels_last + PyTorch-native BatchNorm2d."""
    variants = [("results.csv", "NCHW (default)", "#9ca3af"),
                ("results_channels_last.csv", "channels_last", "#dc2626"),
                ("results_channels_last_nativebn.csv", "channels_last + native BN", "#059669")]
    variants = [v for v in variants if os.path.exists(v[0])]
    if len(variants) < 2:
        return
    frames = []
    for f, label, _ in variants:
        d = pd.read_csv(f)
        d = d[d["status"] == "ok"][["precision", "batch", "samples_per_s", "peak_mem_mb"]]
        d["variant"] = label
        frames.append(d)
    df = pd.concat(frames)
    precs = [p for p in ["fp32", "fp16", "bf16"] if p in set(df["precision"])]

    fig, ax = plt.subplots(1, len(precs), figsize=(4.4 * len(precs), 3.7), squeeze=False)
    for a, p in zip(ax[0], precs):
        for f, label, color in variants:
            g = df[(df["precision"] == p) & (df["variant"] == label)]
            if len(g):
                a.plot(g["batch"], g["samples_per_s"], marker="o", color=color, label=label)
        a.set(title=p, xlabel="Batch size", ylabel="Samples / s")
        a.set_xscale("log", base=2)
    ax[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("SAFE-Net training: memory format and batch-norm implementation", fontsize=11)
    fig.tight_layout()
    fig.savefig("channels_last.png", dpi=150)
    plt.close(fig)

    lines.append("\n## Optimisation: channels_last memory format\n")
    piv = df.pivot_table(index=["precision", "batch"], columns="variant", values="samples_per_s")
    piv = piv[[v[1] for v in variants if v[1] in piv.columns]]
    base = variants[0][1]
    for col in piv.columns[1:]:
        piv[f"{col} vs default"] = (piv[col] / piv[base]).map(lambda v: f"{v:.2f}x")
    lines.append(md(piv.reset_index(), index=False) + "\n\n*samples/s*\n")

    rows = []
    for p in ["fp32", "fp16", "bf16"]:
        for suffix, label in [("", "NCHW"), ("_cl", "channels_last"), ("_cl_nbn", "channels_last + native BN")]:
            f = f"layout_{p}{suffix}.csv"
            if os.path.exists(f):
                r = pd.read_csv(f).iloc[0]
                rows.append({"precision": p, "variant": label,
                             "GPU ms/step": r["all_gpu_ms_per_step"],
                             "batch-norm ms/step": r.get("batchnorm_ms_per_step", float("nan")),
                             "layout conversion ms/step": r["layout_conversion_ms_per_step"]})
    if rows:
        lines.append("\n**Profiler breakdown (batch 128):**\n")
        lines.append(md(pd.DataFrame(rows), index=False) + "\n")


if __name__ == "__main__":
    for part in (part1, part_channels_last, part2a, part2c, part3):
        try:
            part()
        except Exception as e:          # one missing/odd CSV shouldn't stop the rest
            print(f"{part.__name__}: skipped ({e})")
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\nSaved RESULTS.md and plots:", ", ".join(sorted(glob.glob("*.png"))))
