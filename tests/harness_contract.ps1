$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent $PSScriptRoot
$runner = Join-Path $repoRoot 'scripts\poc.ps1'
$probe = & $runner probe | ConvertFrom-Json

if ($probe.memory_max -ne 4294967296) {
    throw "expected 4 GiB memory cap, got $($probe.memory_max)"
}
if ($probe.swap_max -ne 0) {
    throw "expected swap to be disabled, got $($probe.swap_max)"
}
if ($probe.cpu_quota -ne 200000 -or $probe.cpu_period -ne 100000) {
    throw "expected 2 CPU quota, got $($probe.cpu_quota)/$($probe.cpu_period)"
}
if ($probe.pids_max -ne 512) {
    throw "expected PID ceiling 512, got $($probe.pids_max)"
}
if ($probe.corpus_read_only -ne $true) {
    throw 'expected /corpus to be mounted read-only'
}

'harness contract passed'
