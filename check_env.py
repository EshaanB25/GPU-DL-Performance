"""Check the environment before running anything. Exits non-zero if the GPU isn't usable."""
import shutil
import sys

ok = True
try:
    import torch
    print(f"PyTorch       : {torch.__version__} (CUDA {torch.version.cuda})")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU           : {p.name}, {p.total_memory / 2**30:.1f} GB, "
              f"compute capability {p.major}.{p.minor}, {p.multi_processor_count} SMs")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")
        arch = f"sm_{p.major}{p.minor}"
        if arch not in torch.cuda.get_arch_list():
            print(f"WARNING: this PyTorch build has no kernels for {arch} "
                  f"({torch.cuda.get_arch_list()}). Install a CUDA 12.8+ build.")
            ok = False
        torch.ones(8, device="cuda").sum().item()       # a real kernel launch
        print(f"Use -arch={arch} when building matmul.cu")
    else:
        print("ERROR: torch.cuda.is_available() is False")
        ok = False
except ImportError as e:
    print("ERROR:", e)
    ok = False

for mod in ["torchvision", "librosa", "pandas", "matplotlib", "sklearn", "seaborn", "tqdm"]:
    try:
        __import__(mod)
    except ImportError:
        print(f"missing Python package: {mod}")
        ok = False
for tool in ["nvcc", "nsys"]:
    print(f"{tool:14s}: {shutil.which(tool) or 'not on PATH (only needed for Part 3 / Part 2b)'}")
sys.exit(0 if ok else 1)
