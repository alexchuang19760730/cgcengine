# auto_git_push.ps1 - Auto commit and push agent_harness changes
# Usage: powershell -ExecutionPolicy Bypass -File auto_git_push.ps1

$ErrorActionPreference = "Continue"

# Config
$RepoDir    = "D:\alex\flashkv0516\cgcengine_full"
$WatchDir   = "agent_harness"
$RemoteName = "cgc0907"
$BranchName = "fusionroutemot"
$LogDir     = Join-Path $RepoDir "agent_harness\scripts\logs"
$LogFile    = Join-Path $LogDir ("auto_git_push_" + (Get-Date -Format "yyyyMMdd") + ".log")

# Init log dir
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

function Log {
    param([string]$Msg, [string]$Lvl = "INFO")
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [$Lvl] $Msg"
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
    Write-Host $line
}

Log "========== Auto Git Push Start =========="
Log "Repo: $RepoDir"
Log "Watch: $WatchDir"
Log "Remote: $RemoteName/$BranchName"

# Check repo
if (-not (Test-Path $RepoDir)) {
    Log "Repo dir not found: $RepoDir" "ERROR"
    exit 1
}
Set-Location $RepoDir

# Check git
try {
    $gv = & git --version 2>&1
    Log "Git: $gv"
} catch {
    Log "Git not available: $_" "ERROR"
    exit 1
}

# Check changes
Log "Checking changes in $WatchDir/ ..."
$status = & git status --short -- $WatchDir 2>&1
$changes = @()
if ($status) {
    $changes = $status | Where-Object { $_.Trim() -ne "" }
}

if ($changes.Count -eq 0) {
    Log "No changes found. Skip."
    Log "========== Done (no changes) =========="
    exit 0
}

Log "Found $($changes.Count) change(s):"
foreach ($c in $changes) {
    Log "  $c"
}

# Git add
Log "Running: git add $WatchDir/"
$addOut = & git add $WatchDir 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "git add failed: $addOut" "ERROR"
    exit 1
}
Log "git add OK."

# Git commit
$ts = Get-Date -Format "yyyy-MM-dd HH:mm"
$msg = "auto(agent_harness): periodic backup $ts"
Log "Running: git commit -m `"$msg`""
$commitOut = & git commit -m $msg 2>&1

if ($LASTEXITCODE -ne 0) {
    $combined = ($commitOut | Out-String)
    if ($combined -match "nothing to commit" -or $combined -match "no changes added") {
        Log "Nothing to commit after add."
        Log "========== Done (nothing to commit) =========="
        exit 0
    }
    Log "git commit failed: $combined" "ERROR"
    exit 1
}
Log "git commit OK."
Log ($commitOut | Out-String)

# Get commit hash
$hash = & git rev-parse --short HEAD 2>&1
Log "Commit: $hash"

# Git push
Log "Running: git push $RemoteName $BranchName"
$pushOut = & git push $RemoteName $BranchName 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "git push failed: $pushOut" "ERROR"
    Log "Possible causes: network, credentials, branch protection" "WARN"
    exit 1
}
Log "git push OK."
Log ($pushOut | Out-String)

Log "SUCCESS: $hash -> $RemoteName/$BranchName"
Log "========== Done (success) =========="
exit 0
