# Runs the whole study on the GPU and writes logs\ , CSVs, plots and RESULTS.md.
#
# From an Anaconda PowerShell Prompt (or VS Code terminal) in this folder:
#     conda activate dl
#     powershell -ExecutionPolicy Bypass -File run_all.ps1
#
# Optional: point Part 2c at real ASVspoof audio (otherwise it uses synthetic audio)
#     powershell -ExecutionPolicy Bypass -File run_all.ps1 -DataRoot "C:\path\to\ASVspoof2019"
#
# Before running: charger plugged in, Windows power mode "Balanced", GPU clock locked
# (admin PowerShell: nvidia-smi -lgc 210,1500; undo with nvidia-smi -rgc),
# other GPU-heavy apps closed. Laptop GPUs throttle on battery and when hot.

param([string]$DataRoot = "")

$ErrorActionPreference = "Continue"
$env:PYTHONUNBUFFERED = "1"   # show Python output live instead of in buffered bursts
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Step($name, $cmd) {
    Write-Host "`n==== $name ====" -ForegroundColor Cyan
    $t = Get-Date
    Invoke-Expression "$cmd 2>&1" | Tee-Object -FilePath "logs\$name.log"
    Write-Host ("---- {0} took {1:N0} s" -f $name, ((Get-Date) - $t).TotalSeconds)
}

Step "00_env" "python check_env.py"
if ($LASTEXITCODE -ne 0) { Write-Host "Environment check failed - fix the above first." -ForegroundColor Red; exit 1 }
nvidia-smi --query-gpu=name,compute_cap,driver_version,power.limit,temperature.gpu --format=csv | Tee-Object -FilePath logs\00_nvidia_smi.log

# Part 1: precision x batch size (about 10-20 min)
Step "01_bench" "python bench.py"

# Part 2a: kernel and per-block breakdown, fp32 / fp16 / bf16
Step "02a_profile" "python profile_run.py --batch 128"

# Part 2b: Nsight Systems timelines (skipped if nsys isn't on PATH)
if (Get-Command nsys -ErrorAction SilentlyContinue) {
    Step "02b_nsys_fp16" "nsys profile -o safenet_fp16 --force-overwrite true --trace=cuda,nvtx,osrt python bench.py --precision fp16 --batch 128 --steps 20 --repeats 1 --nvtx"
    Step "02b_nsys_fp32" "nsys profile -o safenet_fp32 --force-overwrite true --trace=cuda,nvtx,osrt python bench.py --precision fp32 --batch 128 --steps 20 --repeats 1 --nvtx"
} else { Write-Host "nsys not found - skipping Part 2b" -ForegroundColor Yellow }

# Part 2c: data loading
if ($DataRoot) { Step "02c_dataloader" "python dataloader_test.py --data-root `"$DataRoot`"" }
else           { Step "02c_dataloader" "python dataloader_test.py" }

# Part 3: CUDA kernels. Try nvcc inside the VS 2022 C++ environment; else CuPy.
$built = $false
if (Get-Command nvcc -ErrorAction SilentlyContinue) {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    $vcvars = $null
    if (Test-Path $vswhere) {
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($vs) { $vcvars = Join-Path $vs "VC\Auxiliary\Build\vcvars64.bat" }
    }
    $build = "nvcc -O3 -arch=sm_120 matmul.cu -lcublas -o matmul"
    if ($vcvars -and (Test-Path $vcvars)) { cmd /c "`"$vcvars`" >nul && $build" 2>&1 | Tee-Object logs\03_build.log }
    else { Invoke-Expression "$build 2>&1" | Tee-Object logs\03_build.log }
    if (Test-Path .\matmul.exe) {
        Write-Host "`n==== 03_matmul ====" -ForegroundColor Cyan
        .\matmul.exe > matmul_results.csv
        Get-Content matmul_results.csv | Tee-Object logs\03_matmul.log
        $built = $true
    }
}
if (-not $built) {
    Write-Host "nvcc build unavailable - using the CuPy fallback (pip install cupy-cuda12x)" -ForegroundColor Yellow
    Step "03_matmul_cupy" "python matmul_cupy.py"
}

# Plots + results table
Step "04_report" "python report.py"
Write-Host "`nDone. Open RESULTS.md and the .png files. Traces: trace_*.json (ui.perfetto.dev), safenet_*.nsys-rep (Nsight Systems)." -ForegroundColor Green
