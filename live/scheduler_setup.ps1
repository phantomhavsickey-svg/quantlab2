# ============================================================
# QuantLab2 daily task registration
# Registers a Windows Scheduled Task: run live\daily_task.py
# every trading day at 16:00 (after quantlab's 15:30 data update,
# so the factor panel is fresh).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File live\scheduler_setup.ps1
# Inspect / remove:
#   Get-ScheduledTask -TaskName "QuantLab2DailyTask"
#   Unregister-ScheduledTask -TaskName "QuantLab2DailyTask" -Confirm:$false
# ============================================================

$ErrorActionPreference = "Stop"

$TaskName = "QuantLab2DailyTask"
$ProjectDir = Split-Path -Parent $PSScriptRoot   # parent of live/ = project root
$Python = (Get-Command python).Source
$Script = "live\daily_task.py"
$LogDir = Join-Path $ProjectDir "logs"
$LogFile = Join-Path $LogDir "scheduled_task.log"

# Ensure log directory exists
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Delete old task before re-registering
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

# Project path has no spaces, no extra quoting needed
$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument $Script `
    -WorkingDirectory $ProjectDir

# Daily at 16:00
$Trigger = New-ScheduledTaskTrigger -Daily -At 16:00

# Current user, interactive logon
$Principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Principal $Principal `
    -Settings $Settings `
    -Description "QuantLab2 Transformer daily: month-end rebalance / daily mark-to-market" | Out-Null

Write-Host "Scheduled task registered: $TaskName (daily 16:00, python $Script)"
Write-Host "Project dir: $ProjectDir"
Write-Host "Manual test: python live\daily_task.py --dry-run"
