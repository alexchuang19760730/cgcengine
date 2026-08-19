# build.ps1 — 编译 cgc_repack + expert_streaming (Windows)
# 用法:
#   .\build.ps1              # 只编译 repack (BF16 直通)
#   .\build.ps1 -WithGgml    # 编译 repack + IQ3_M (链接 llama.cpp ggml)
#   .\build.ps1 -TestOnly    # 只编译 expert_streamer 测试

param(
    [switch]$WithGgml,
    [switch]$TestOnly,
    [string]$BuildDir = "build"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildPath = Join-Path $ScriptDir $BuildDir

# 找 clang 或 gcc
function Find-Compiler {
    $clang = Get-Command clang -ErrorAction SilentlyContinue
    if ($clang) { return "clang" }
    $clangxx = Get-Command clang++ -ErrorAction SilentlyContinue
    if ($clangxx) { return "clang++" }
    $gcc = Get-Command gcc -ErrorAction SilentlyContinue
    if ($gcc) { return "gcc" }
    $cl = Get-Command cl -ErrorAction SilentlyContinue
    if ($cl) { return "cl" }
    return $null
}

$cc = Find-Compiler
if (-not $cc) {
    Write-Host "ERROR: No C/C++ compiler found." -ForegroundColor Red
    Write-Host "Install one of:"
    Write-Host "  winget install LLVM.LLVM"
    Write-Host "  winget install BrechtSanders.WinLibs.POSIX.UCRT"
    exit 1
}
Write-Host "Using compiler: $cc" -ForegroundColor Green

# CMake 配置
New-Item -ItemType Directory -Force -Path $BuildPath | Out-Null

$cmakeArgs = @(
    "-S", $ScriptDir,
    "-B", $BuildPath,
    "-G", "Ninja"
)

if ($WithGgml) {
    $cmakeArgs += "-DCGC_LINK_GGML=ON"
    Write-Host "Enabling IQ3_M quantization (linking llama.cpp ggml)" -ForegroundColor Cyan
} else {
    $cmakeArgs += "-DCGC_LINK_GGML=OFF"
    Write-Host "BF16 passthrough mode (no IQ3_M)" -ForegroundColor Yellow
}

if ($TestOnly) {
    $cmakeArgs += "-DCGC_BUILD_TESTS=ON"
}

Write-Host "`n=== CMake Configure ===" -ForegroundColor Cyan
& cmake @cmakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configure failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Build ===" -ForegroundColor Cyan
$targets = @("cgc_repack_exe")
if ($TestOnly -or -not $WithGgml) {
    $targets += @("test_expert_streamer")
}

& cmake --build $BuildPath --target @targets -- -j4
if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Build Complete ===" -ForegroundColor Green
Write-Host "Output:"
Get-ChildItem (Join-Path $BuildPath "*.exe") | ForEach-Object {
    Write-Host "  $($_.FullName)" -ForegroundColor White
    Write-Host "    Size: $([math]::Round($_.Length/1MB,1)) MB"
}
