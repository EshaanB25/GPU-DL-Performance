"""Shared helpers: build SAFE-Net, make synthetic batches, run one training step.

The model under test is SAFE-Net (MultiBranchDeepfakeDetector) from
safenet_model.py, imported unchanged. Its three inputs are:
    raw      (B, 1, 64000)     4 s of 16 kHz audio
    log_mel  (B, 1, 64, 128)   64 mel bands x 128 frames
    mfcc     (B, 1, 40, 128)   40 MFCCs x 128 frames
"""
import contextlib
import time

import torch
import torch.nn as nn

import safenet_model as sn

PRECISIONS = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def pick_device(requested=None):
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def setup_backends(device):
    """Make FP32 a true FP32 baseline and let cuDNN pick fast algorithms."""
    if device == "cuda":
        # Newer GPUs may silently run "FP32" maths on TF32 tensor cores
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = True


def device_name(device):
    if device == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def sync(device):
    """GPU work is asynchronous: wait for it before reading a timer."""
    if device == "cuda":
        torch.cuda.synchronize()


def build_model(device):
    torch.manual_seed(0)
    return sn.MultiBranchDeepfakeDetector().to(device).train()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def synthetic_batch(batch, device):
    """Random tensors with SAFE-Net's input shapes; the 1:9 real:fake label mix
    matches ASVspoof 2019 LA so the weighted loss behaves as in real training."""
    g = torch.Generator().manual_seed(0)
    raw = torch.randn(batch, 1, sn.RAW_LENGTH, generator=g)
    log_mel = torch.randn(batch, 1, sn.N_MELS, sn.MAX_FRAMES, generator=g)
    mfcc = torch.randn(batch, 1, sn.N_MFCC, sn.MAX_FRAMES, generator=g)
    y = (torch.rand(batch, generator=g) > 0.1).long()      # ~10% bonafide (0)
    return [t.to(device) for t in (raw, log_mel, mfcc, y)]


def to_channels_last(model, batch):
    """Store SAFE-Net's 2D tensors in NHWC (channels_last) order.

    Tensor-core convolution kernels want NHWC. With the default NCHW layout,
    cuDNN inserts nchwToNhwc / nhwcToNchw conversion kernels around every 2D
    convolution. Converting the conv2d weights and the two 2D inputs (log-mel,
    MFCC) once removes those conversions. The raw-waveform branch is 1D (3D
    tensors), which has no channels_last format, so it is left unchanged.
    """
    model.to(memory_format=torch.channels_last)   # affects 4D params only (conv2d weights)
    raw, log_mel, mfcc, y = batch
    log_mel = log_mel.contiguous(memory_format=torch.channels_last)
    mfcc = mfcc.contiguous(memory_format=torch.channels_last)
    return model, [raw, log_mel, mfcc, y]


class NativeBatchNorm2d(nn.Module):
    """Runs a BatchNorm2d with cuDNN switched off, so PyTorch's own kernel is used.

    With channels_last (NHWC) activations, cuDNN's batch-norm picks much slower
    kernels (bn_fw_tr_1C11_singleread / bn_bw_1C11_singleread). PyTorch's native
    batch-norm has dedicated channels_last kernels. Autograd records which
    implementation ran in the forward pass, so the backward pass follows it.
    """
    def __init__(self, bn):
        super().__init__()
        self.bn = bn

    def forward(self, x):
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False
        try:
            return self.bn(x)
        finally:
            torch.backends.cudnn.enabled = prev


def use_native_batchnorm2d(model):
    """Wrap every BatchNorm2d in the model; returns how many were wrapped."""
    targets = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.BatchNorm2d)]
    for name, m in targets:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, NativeBatchNorm2d(m))
    return len(targets)


def make_training(model, precision, device):
    """Optimiser, weighted loss and GradScaler exactly as SAFE-Net trains."""
    opt = torch.optim.Adam(model.parameters(), lr=sn.LEARNING_RATE, weight_decay=1e-4)
    weights = torch.tensor([sn.REAL_WEIGHT, sn.FAKE_WEIGHT], dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    # Only FP16 needs loss scaling (small exponent range -> gradients underflow)
    scaler = torch.amp.GradScaler(device, enabled=(precision == "fp16" and device == "cuda"))
    return opt, loss_fn, scaler


def nvtx(name, on):
    if on and torch.cuda.is_available():
        return torch.cuda.nvtx.range(name)
    return contextlib.nullcontext()


def train_step(model, batch, opt, loss_fn, scaler, precision, device, use_nvtx=False):
    raw, log_mel, mfcc, y = batch
    dtype = PRECISIONS[precision]
    opt.zero_grad(set_to_none=True)
    with nvtx("forward", use_nvtx):
        with torch.autocast(device, dtype=dtype, enabled=dtype is not None):
            out = model(raw, log_mel, mfcc)
        loss = loss_fn(out.float(), y)          # loss in FP32, as in SAFE-Net
    with nvtx("backward", use_nvtx):
        scaler.scale(loss).backward()
    with nvtx("optimizer", use_nvtx):
        scaler.step(opt)
        scaler.update()
    return loss


def timed_steps(step_fn, steps, device):
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        step_fn()
    sync(device)
    return time.perf_counter() - t0
