param(
    [string]$CudaBinDir = "",
    [string]$DesktopPython = "",
    [string]$AsrPython = "",
    [string]$ModelPath = "",
    [switch]$AllowCpuFallback,
    [switch]$DebugAnalysis
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path -LiteralPath (Join-Path $ScriptDir "..")
Set-Location $Root

if (-not $AsrPython) {
    $candidate = Join-Path $Root ".venv-asr\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $AsrPython = $candidate
    }
}
if (-not $AsrPython -or -not (Test-Path -LiteralPath $AsrPython)) {
    throw "ASR Python not found. Expected .venv-asr\Scripts\python.exe or pass -AsrPython."
}

if (-not $DesktopPython) {
    $candidate = Join-Path $Root ".venv-py311\Scripts\python.exe"
    if (Test-Path -LiteralPath $candidate) {
        $DesktopPython = $candidate
    } else {
        $candidate = Join-Path $Root ".venv\Scripts\python.exe"
        $DesktopPython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
    }
}

if (-not $ModelPath) {
    $ModelPath = Join-Path $Root "models\faster-whisper-small"
}

function Add-PathPrefix([string]$PathToAdd) {
    if ($PathToAdd -and (Test-Path -LiteralPath $PathToAdd)) {
        $resolved = (Resolve-Path -LiteralPath $PathToAdd).Path
        $parts = ($env:PATH -split ';') | Where-Object { $_ }
        if ($parts -notcontains $resolved) {
            $env:PATH = $resolved + ";" + $env:PATH
        }
    }
}

function Find-CudaBinDir {
    param([string]$Explicit)

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($Explicit) { $candidates.Add($Explicit) }
    if ($env:DSO_CUDA_BIN_DIR) { $candidates.Add($env:DSO_CUDA_BIN_DIR) }
    if ($env:CUDA_PATH) { $candidates.Add((Join-Path $env:CUDA_PATH "bin")) }
    $localCuda = Join-Path $Root "third_party\cuda12\bin"
    $candidates.Add($localCuda)
    $toolkitRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (Test-Path -LiteralPath $toolkitRoot) {
        Get-ChildItem -LiteralPath $toolkitRoot -Directory -Filter "v12*" |
            Sort-Object Name -Descending |
            ForEach-Object { $candidates.Add((Join-Path $_.FullName "bin")) }
    }

    foreach ($candidate in $candidates) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate)) { continue }
        if (Test-Path -LiteralPath (Join-Path $candidate "cublas64_12.dll")) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return ""
}

$ct2Dir = Join-Path $Root ".venv-asr\Lib\site-packages\ctranslate2"
Add-PathPrefix $ct2Dir

$resolvedCudaBin = Find-CudaBinDir -Explicit $CudaBinDir
if ($resolvedCudaBin) {
    Add-PathPrefix $resolvedCudaBin
}

$hasCublas = $false
$hasCudnn = $false
foreach ($pathPart in ($env:PATH -split ';')) {
    if (-not $pathPart) { continue }
    if (-not $hasCublas -and (Test-Path -LiteralPath (Join-Path $pathPart "cublas64_12.dll"))) {
        $hasCublas = $true
    }
    if (
        -not $hasCudnn -and (
            (Test-Path -LiteralPath (Join-Path $pathPart "cudnn64_9.dll")) -or
            (Test-Path -LiteralPath (Join-Path $pathPart "cudnn_ops64_9.dll"))
        )
    ) {
        $hasCudnn = $true
    }
}

if (-not $AllowCpuFallback -and (-not $hasCublas -or -not $hasCudnn)) {
    Write-Error @"
CUDA ASR dependencies are incomplete.
Required for faster-whisper/CTranslate2 GPU:
  - cublas64_12.dll
  - cudnn64_9.dll or cudnn_ops64_9.dll

Pass -CudaBinDir <dir> where these DLLs live, install CUDA 12 + cuDNN 9, or place them under third_party\cuda12\bin.
Current cublas64_12.dll found: $hasCublas
Current cuDNN 9 DLL found: $hasCudnn
"@
    exit 1
}

$env:DSO_ASR_PYTHON = (Resolve-Path -LiteralPath $AsrPython).Path
$env:DSO_ASR_MODEL = (Resolve-Path -LiteralPath $ModelPath).Path
$env:DSO_ASR_MODE = if ($env:DSO_ASR_MODE) { $env:DSO_ASR_MODE } else { "whole" }
$env:DSO_ASR_CHUNK_SECONDS = if ($env:DSO_ASR_CHUNK_SECONDS) { $env:DSO_ASR_CHUNK_SECONDS } else { "600" }
$env:DSO_ASR_DEVICE = if ($AllowCpuFallback) { "cpu" } elseif ($env:DSO_ASR_DEVICE) { $env:DSO_ASR_DEVICE } else { "cuda" }
$env:DSO_ASR_COMPUTE_TYPE = if ($AllowCpuFallback) { "int8" } elseif ($env:DSO_ASR_COMPUTE_TYPE) { $env:DSO_ASR_COMPUTE_TYPE } else { "int8_float16" }
$env:DSO_ASR_STRICT_GPU = if ($AllowCpuFallback) { "0" } elseif ($env:DSO_ASR_STRICT_GPU) { $env:DSO_ASR_STRICT_GPU } else { "1" }
if ($DebugAnalysis) {
    $env:DSO_DEBUG_ANALYSIS = "1"
    $env:DSO_REQUIRE_ASR = "1"
    $env:DSO_ASR_STRICT_GPU = "1"
}

Write-Host "Launching workbench with GPU ASR settings:"
Write-Host "  ASR Python: $env:DSO_ASR_PYTHON"
Write-Host "  ASR Model:  $env:DSO_ASR_MODEL"
Write-Host "  ASR Mode:   $env:DSO_ASR_MODE"
Write-Host "  Device:     $env:DSO_ASR_DEVICE / $env:DSO_ASR_COMPUTE_TYPE"
Write-Host "  CUDA Bin:   $resolvedCudaBin"
Write-Host "  Debug:      $env:DSO_DEBUG_ANALYSIS"
Write-Host "  Require ASR:$env:DSO_REQUIRE_ASR"

& $DesktopPython (Join-Path $Root "scripts\launch_workbench_desktop.py")
exit $LASTEXITCODE
