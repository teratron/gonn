# ═══════════════════════════════════════════════════════════════════════════════
# AGENTS INIT (WINDOWS)
# Universal agent environment initializer.
#
# Usage:
#   setup_windows.ps1                     # default: claude
#   setup_windows.ps1 -Agents qwen
#   setup_windows.ps1 -Agents claude,qwen
#   setup_windows.ps1 -Agents all
# ═══════════════════════════════════════════════════════════════════════════════

param(
    [string]$Agents = "claude"
)

# ───────────────────────────────────────────────────────────────────────────────
# 1. Agent Registry — loaded from agents.json
# ───────────────────────────────────────────────────────────────────────────────

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RegistryFile = Join-Path (Split-Path -Parent $ScriptDir) "agents.json"

if (-not (Test-Path $RegistryFile)) {
    Write-Error "Registry not found: $RegistryFile"
    exit 1
}

$json = Get-Content $RegistryFile -Raw | ConvertFrom-Json

# Convert PSCustomObject → hashtable for easy lookup
$REGISTRY = @{}
foreach ($prop in $json.PSObject.Properties) {
    $REGISTRY[$prop.Name] = @{
        dir       = $prop.Value.dir
        workflows = $prop.Value.workflows
        skills    = $prop.Value.skills
        rules    = $prop.Value.rules
        files    = if ($prop.Value.files) { @($prop.Value.files) } else { @() }
    }
}

$ALL_AGENTS = $REGISTRY.Keys | Sort-Object

# ───────────────────────────────────────────────────────────────────────────────
# 2. Parse target agents
# ───────────────────────────────────────────────────────────────────────────────

if ($Agents -eq "all") {
    $targets = $ALL_AGENTS
} else {
    $targets = $Agents -split "[,\s]+" | Where-Object { $_ -ne "" }
}

foreach ($t in $targets) {
    if (-not $REGISTRY.ContainsKey($t)) {
        Write-Warning "Unknown agent: '$t'. Supported: $($ALL_AGENTS -join ', ')"
        exit 1
    }
}

Write-Host ">>> Initializing Windows Agent Environment for: $($targets -join ', ')" -ForegroundColor Cyan

# ───────────────────────────────────────────────────────────────────────────────
# 3. Helpers
# ───────────────────────────────────────────────────────────────────────────────

function Remove-Existing($path) {
    if (Test-Path $path) {
        Write-Host "  Removing: $path" -ForegroundColor Yellow
        $item = Get-Item -LiteralPath $path -Force
        if ($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) {
            if ($item.PSIsContainer) { cmd /c "rmdir ""$path""" }
            else                     { cmd /c "del ""$path""" }
        } else {
            Remove-Item -Recurse -Force -LiteralPath $path
        }
    }
}

function New-Junction($link, $target) {
    Remove-Existing $link
    New-Item -ItemType Junction -Path $link -Target $target -Force | Out-Null
}

function New-Hardlink($link, $target) {
    Remove-Existing $link
    New-Item -ItemType HardLink -Path $link -Target $target -Force | Out-Null
}

# ───────────────────────────────────────────────────────────────────────────────
# 4. Collect paths for git index cleanup
# ───────────────────────────────────────────────────────────────────────────────

$gitUntrack = @()

foreach ($agent in $targets) {
    $cfg = $REGISTRY[$agent]
    $dir = $cfg.dir

    if ($cfg.workflows) { $gitUntrack += "$dir/$($cfg.workflows)" }
    if ($cfg.skills)    { $gitUntrack += "$dir/$($cfg.skills)" }
    if ($cfg.rules)    { $gitUntrack += "$dir/$($cfg.rules)" }

    foreach ($f in $cfg.files) { $gitUntrack += $f }
}

Write-Host "`nSynchronizing git index (pre-link)..." -ForegroundColor Cyan
if ($gitUntrack.Count -gt 0) {
    git rm -r --cached --ignore-unmatch $gitUntrack 2>$null
}

# ───────────────────────────────────────────────────────────────────────────────
# 5. Create junctions and hardlinks per agent
# ───────────────────────────────────────────────────────────────────────────────

foreach ($agent in $targets) {
    $cfg = $REGISTRY[$agent]
    $dir = $cfg.dir

    Write-Host "`n[+] Agent: $agent  →  $dir" -ForegroundColor Magenta

    # Ensure config directory exists
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    # Junctions
    if ($cfg.workflows) { New-Junction "$dir\$($cfg.workflows)" ".agents\workflows" }
    if ($cfg.skills)    { New-Junction "$dir\$($cfg.skills)"   ".agents\skills" }
    if ($cfg.rules)    { New-Junction "$dir\$($cfg.rules)"    ".agents\rules" }

    # Hardlinks: instruction files → AGENTS.md
    foreach ($f in $cfg.files) {
        New-Hardlink $f "AGENTS.md"
    }
}

# ───────────────────────────────────────────────────────────────────────────────
# 6. Verification
# ───────────────────────────────────────────────────────────────────────────────

Write-Host "`n>>> Verification (junctions):" -ForegroundColor Green
foreach ($agent in $targets) {
    $dir = $REGISTRY[$agent].dir
    $w = if ($REGISTRY[$agent].workflows) { """$dir\$($REGISTRY[$agent].workflows)""" } else { "" }
    $s = if ($REGISTRY[$agent].skills)    { """$dir\$($REGISTRY[$agent].skills)""" }   else { "" }
    $r = if ($REGISTRY[$agent].rules)     { """$dir\$($REGISTRY[$agent].rules)""" }    else { "" }
    if ($w -or $s -or $r) { cmd /c "dir $w $s $r /AL 2>nul" }
}

Write-Host "`n>>> Hardlink Integrity Check (AGENTS.md):" -ForegroundColor Cyan
cmd /c "fsutil hardlink list AGENTS.md"

Write-Host "`n>>> Done." -ForegroundColor Green
