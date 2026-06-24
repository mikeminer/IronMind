param(
    [switch]$SkipRuntimeBuild,
    [switch]$SkipModelBuild
)

$ErrorActionPreference = "Stop"

$Repo = "mikeminer/IronMind"
$Branch = "iurexa"
$IkLlamaCommit = "d5507e33"
$InstallDir = Join-Path $env:LOCALAPPDATA "IronMind"
$UserRoot = Join-Path $env:USERPROFILE ".ironmind"
$BinDir = Join-Path $UserRoot "bin"
$ConfigPath = Join-Path $UserRoot "ironmind.json"
$RuntimeDir = Join-Path $UserRoot "runtimes\ik_llama.cpp"
$RuntimeBin = Join-Path $RuntimeDir "build\bin\Release"
$EmbeddedRuntimeDir = Join-Path $InstallDir "third_party\ik_llama.cpp"
$EmbeddedRunner = Join-Path $InstallDir "build-ik\Release\ironmind-ik-native.exe"
$ModelDir = Join-Path $UserRoot "models\iurexa"
$BaseModelPath = Join-Path $ModelDir "Qwen3-1.7B-F16.gguf"
$ImatrixPath = Join-Path $ModelDir "iurexa-qwen3-1.7b-instruct-legal-it.imatrix"
$QuantModelPath = Join-Path $ModelDir "iurexa-qwen3-1.7b-instruct-IQ4_XS.gguf"
$BaseModelUrl = $env:IRONMIND_IUREXA_BASE_MODEL_URL
if (-not $BaseModelUrl) {
    $BaseModelUrl = "https://huggingface.co/lm-kit/qwen-3-1.7b-instruct-gguf/resolve/main/Qwen3-1.7B-F16.gguf"
}
$ZipPath = Join-Path $env:TEMP "ironmind-main.zip"
$ExtractDir = Join-Path $env:TEMP ("ironmind-" + [guid]::NewGuid().ToString("N"))

function Ensure-Node {
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) { return }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw "Node.js 18+ is required. Install Node.js, then run this installer again."
    }

    Write-Host "Installing Node.js LTS with winget..."
    winget install OpenJS.NodeJS.LTS -e --silent --accept-package-agreements --accept-source-agreements
}

function Add-ToUserPath($PathToAdd) {
    $current = [Environment]::GetEnvironmentVariable("Path", "User")
    $parts = @()
    if ($current) {
        $parts = $current -split ";" | Where-Object { $_ -ne "" }
    }
    if ($parts -contains $PathToAdd) { return }

    $newPath = (($parts + $PathToAdd) -join ";")
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = $env:Path + ";" + $PathToAdd
}

Ensure-Node

function Ensure-IkLlamaRuntime {
    if ($SkipRuntimeBuild) { return }

    $server = Join-Path $RuntimeBin "llama-server.exe"
    $worker = Join-Path $RuntimeBin "llama-cli.exe"
    $quantize = Join-Path $RuntimeBin "llama-quantize.exe"
    $imatrix = Join-Path $RuntimeBin "llama-imatrix.exe"
    if ((Test-Path $server) -and (Test-Path $worker) -and (Test-Path $quantize) -and (Test-Path $imatrix)) {
        return
    }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Warning "Git was not found. Skipping ik_llama.cpp runtime build."
        return
    }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Warning "CMake was not found. Skipping ik_llama.cpp runtime build."
        return
    }

    New-Item -ItemType Directory -Path (Split-Path $RuntimeDir) -Force | Out-Null
    if (-not (Test-Path $RuntimeDir)) {
        Write-Host "Cloning ik_llama.cpp..."
        git clone https://github.com/ikawrakow/ik_llama.cpp $RuntimeDir
    }

    Push-Location $RuntimeDir
    try {
        git fetch --depth 1 origin $IkLlamaCommit
        git checkout $IkLlamaCommit
        cmake -B build -DGGML_NATIVE=ON -DGGML_IQK_FA_ALL_QUANTS=ON
        cmake --build build --config Release --target llama-server llama-cli llama-quantize llama-imatrix llama-bench
    } finally {
        Pop-Location
    }
}

function Ensure-IurexaModel {
    if ($SkipModelBuild) { return }
    if (Test-Path $QuantModelPath) { return }

    $quantize = Join-Path $RuntimeBin "llama-quantize.exe"
    $imatrix = Join-Path $RuntimeBin "llama-imatrix.exe"
    if (-not ((Test-Path $quantize) -and (Test-Path $imatrix))) {
        Write-Warning "ik_llama quantization tools are missing. Skipping model quantization."
        return
    }

    New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null
    if (-not (Test-Path $BaseModelPath)) {
        Write-Host "Downloading Iurexa base model F16..."
        Invoke-WebRequest -Uri $BaseModelUrl -OutFile $BaseModelPath
    }

    $calibration = Join-Path $InstallDir "calibration\iurexa-legal-it.txt"
    if (-not (Test-Path $ImatrixPath)) {
        Write-Host "Generating Iurexa Italian legal importance matrix..."
        & $imatrix -m $BaseModelPath -f $calibration -o $ImatrixPath -t 6 -c 4096 -b 128 -ngl 0
    }

    Write-Host "Quantizing Iurexa IQ4_XS..."
    & $quantize --imatrix $ImatrixPath $BaseModelPath $QuantModelPath IQ4_XS 6
}

function Ensure-IronMindEmbeddedRuntime {
    if ($SkipRuntimeBuild) { return }
    if (Test-Path $EmbeddedRunner) { return }

    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Warning "Git was not found. Skipping embedded ik_llama.cpp runtime build."
        return
    }
    if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
        Write-Warning "CMake was not found. Skipping embedded ik_llama.cpp runtime build."
        return
    }

    try {
        $embeddedCMake = Join-Path $EmbeddedRuntimeDir "CMakeLists.txt"
        if (-not (Test-Path $embeddedCMake)) {
            if (Test-Path $EmbeddedRuntimeDir) {
                Remove-Item -LiteralPath $EmbeddedRuntimeDir -Recurse -Force
            }
            New-Item -ItemType Directory -Path (Split-Path $EmbeddedRuntimeDir) -Force | Out-Null
            Write-Host "Cloning pinned ik_llama.cpp for embedded Iurexa runtime..."
            git clone https://github.com/ikawrakow/ik_llama.cpp $EmbeddedRuntimeDir
        }

        Push-Location $EmbeddedRuntimeDir
        try {
            git fetch --depth 1 origin $IkLlamaCommit
            git checkout $IkLlamaCommit
        } finally {
            Pop-Location
        }

        Push-Location $InstallDir
        try {
            Write-Host "Building Iurexa embedded CPU runtime..."
            cmake -S . -B build-ik -DIRONMIND_WITH_IK_LLAMA=ON
            cmake --build build-ik --config Release --target ironmind-ik-native
        } finally {
            Pop-Location
        }
    } catch {
        Write-Warning "Embedded ik_llama.cpp runtime build failed: $($_.Exception.Message)"
    }
}

if (Test-Path $ZipPath) {
    Remove-Item -LiteralPath $ZipPath -Force
}
if (Test-Path $ExtractDir) {
    Remove-Item -LiteralPath $ExtractDir -Recurse -Force
}

Write-Host "Downloading IronMind..."
Invoke-WebRequest -Uri "https://github.com/$Repo/archive/refs/heads/$Branch.zip" -OutFile $ZipPath
Expand-Archive -Path $ZipPath -DestinationPath $ExtractDir -Force

$SourceDir = Get-ChildItem -LiteralPath $ExtractDir -Directory | Select-Object -First 1
if (-not $SourceDir) {
    throw "Downloaded archive did not contain the IronMind source directory."
}

if (Test-Path $InstallDir) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Path $InstallDir | Out-Null
Copy-Item -Path (Join-Path $SourceDir.FullName "*") -Destination $InstallDir -Recurse -Force

Ensure-IkLlamaRuntime
Ensure-IurexaModel
Ensure-IronMindEmbeddedRuntime

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
$CmdPath = Join-Path $BinDir "ironmind.cmd"
$Cmd = @"
@echo off
node "$InstallDir\bin\ironmind.mjs" %*
"@
Set-Content -LiteralPath $CmdPath -Value $Cmd -Encoding ASCII
Add-ToUserPath $BinDir

if (-not (Test-Path $ConfigPath)) {
    $serverPath = Join-Path $RuntimeBin "llama-server.exe"
    $workerPath = Join-Path $RuntimeBin "llama-cli.exe"
    $backend = "ik_embedded"
    if (-not ((Test-Path $EmbeddedRunner) -and (Test-Path $QuantModelPath))) {
        $backend = "ik_worker"
    }
    if (($backend -eq "ik_worker") -and -not ((Test-Path $workerPath) -and (Test-Path $QuantModelPath))) {
        $backend = "auto"
    }
    $Config = @"
{
  "model": "iurexa",
  "context": 40960,
  "kvDiskDir": "$($UserRoot.Replace('\', '\\'))\\kvcache",
  "kvDiskSpaceMb": 16384,
  "documentStoreDir": "$($UserRoot.Replace('\', '\\'))\\documents",
  "backend": "$backend",
  "ikLlamaServer": "$($serverPath.Replace('\', '\\'))",
  "ikLlamaWorker": "$($workerPath.Replace('\', '\\'))",
  "ikEmbeddedRunner": "$($EmbeddedRunner.Replace('\', '\\'))",
  "ikLlamaModel": "$($QuantModelPath.Replace('\', '\\'))",
  "cpuOnly": true,
  "cpuProfile": "low-latency",
  "cpuThreads": 6,
  "cpuBatch": 128,
  "cpuInteractiveCtx": 4096,
  "cpuMaxTokens": 256
}
"@
    Set-Content -LiteralPath $ConfigPath -Value $Config -Encoding UTF8
}

Remove-Item -LiteralPath $ZipPath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Iurexa installed."
Write-Host "Run: ironmind"
Write-Host "Default model: iurexa"
Write-Host "Runtime: ik_embedded when the linked CPU runtime is available, otherwise ik_worker/auto"
Write-Host "Chatbot: http://127.0.0.1:4141"
