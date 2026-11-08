# Operator utility: LAN-only access for the AI-SOC dashboard.
# RUN AS ADMINISTRATOR. Marks the given network interface (default: WiFi)
# as Private and allows inbound TCP 8501 on Private profiles only, from the
# local subnet. Idempotent. Writes results/allow_lan_access.log.
param([string]$InterfaceAlias = "WiFi")

$ErrorActionPreference = "Stop"
$log = Join-Path $PSScriptRoot "..\results\allow_lan_access.log"
"=== $(Get-Date -Format o) ===" | Out-File $log -Encoding utf8
try {
    $profiles = Get-NetConnectionProfile |
        Where-Object { $_.InterfaceAlias -eq $InterfaceAlias }
    if (-not $profiles) { throw "No connection profile for interface '$InterfaceAlias'." }
    foreach ($p in $profiles) {
        "before: $($p.Name) [$($p.InterfaceAlias)] = $($p.NetworkCategory)" |
            Out-File $log -Append -Encoding utf8
        if ($p.NetworkCategory -ne "Private") {
            Set-NetConnectionProfile -InterfaceIndex $p.InterfaceIndex `
                -NetworkCategory Private
        }
    }
    foreach ($p in (Get-NetConnectionProfile |
                    Where-Object { $_.InterfaceAlias -eq $InterfaceAlias })) {
        "after: $($p.Name) [$($p.InterfaceAlias)] = $($p.NetworkCategory)" |
            Out-File $log -Append -Encoding utf8
    }
    if (-not (Get-NetFirewallRule -DisplayName "AI-SOC Streamlit (LAN)" `
              -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -DisplayName "AI-SOC Streamlit (LAN)" `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8501 `
            -Profile Private -RemoteAddress LocalSubnet | Out-Null
    }
    Get-NetFirewallRule -DisplayName "AI-SOC Streamlit (LAN)" |
        Select-Object DisplayName, Enabled, Profile, Action |
        Format-List | Out-File $log -Append -Encoding utf8
    "OK" | Out-File $log -Append -Encoding utf8
} catch {
    "ERROR: $_" | Out-File $log -Append -Encoding utf8
    exit 1
}