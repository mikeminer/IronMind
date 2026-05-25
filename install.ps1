$ErrorActionPreference = "Stop"

$Repo = "mikeminer/IronMind"
$Branch = "main"
$InstallDir = Join-Path $env:LOCALAPPDATA "IronMind"
$UserRoot = Join-Path $env:USERPROFILE ".ironmind"
$BinDir = Join-Path $UserRoot "bin"
$ConfigPath = Join-Path $UserRoot "ironmind.json"
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

if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama was not found. Install it from https://ollama.com/download, then run: ollama pull qwen3-coder:30b"
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

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
$CmdPath = Join-Path $BinDir "ironmind.cmd"
$Cmd = @"
@echo off
node "$InstallDir\bin\ironmind.mjs" %*
"@
Set-Content -LiteralPath $CmdPath -Value $Cmd -Encoding ASCII
Add-ToUserPath $BinDir

if (-not (Test-Path $ConfigPath)) {
    $Config = @"
{
  "model": "qwen3-coder:30b",
  "context": 32768,
  "ollamaUrl": "http://127.0.0.1:11434"
}
"@
    Set-Content -LiteralPath $ConfigPath -Value $Config -Encoding UTF8
}

Remove-Item -LiteralPath $ZipPath -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $ExtractDir -Recurse -Force -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "IronMind installed."
Write-Host "Run: ironmind"
Write-Host "Default model: qwen3-coder:30b"
Write-Host "If needed: ollama pull qwen3-coder:30b"
Write-Host "Chatbot: http://127.0.0.1:4141"
