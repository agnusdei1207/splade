# Experiment runner: same image and mounts as poc.ps1, but with tunable CPU/memory
# so throughput and RAM cost can be measured outside the fixed POC cap.
param(
    [int]$Cpus = 2,
    [string]$Memory = '4g',
    # Extra packages resolved for this run only, e.g. -With onnxruntime
    [string[]]$With = @(),
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$Script,
    [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$corpusRoot = if ($env:SPLADE_CORPUS_ROOT) {
    (Resolve-Path $env:SPLADE_CORPUS_ROOT).Path
} else {
    (Resolve-Path 'C:\workspace\pentesting').Path
}
$cacheRoot = Join-Path $repoRoot '.cache'
New-Item -ItemType Directory -Force -Path $cacheRoot | Out-Null
$corpusGitSha = (& git -C $corpusRoot rev-parse HEAD).Trim()

$dockerArgs = @(
    'run', '--rm',
    '--memory', $Memory, '--memory-swap', $Memory,
    '--cpus', "$Cpus", '--pids-limit', '512',
    '--mount', "type=bind,source=$repoRoot,target=/workspace",
    '--mount', "type=bind,source=$corpusRoot,target=/corpus,readonly",
    '--mount', "type=bind,source=$cacheRoot,target=/cache",
    '--env', 'HF_HOME=/cache/huggingface',
    '--env', 'UV_CACHE_DIR=/cache/uv',
    '--env', 'UV_PROJECT_ENVIRONMENT=/cache/venv',
    '--env', 'MPLCONFIGDIR=/cache/matplotlib',
    '--env', 'PYTHONPATH=/workspace/src',
    '--env', "SPLADE_CORPUS_GIT_SHA=$corpusGitSha",
    '--env', "SPLADE_EXP_CPUS=$Cpus",
    '--workdir', '/workspace',
    'ghcr.io/astral-sh/uv:python3.12-bookworm-slim'
)

$uvArgs = @('uv', 'run', '--no-sync')
foreach ($package in $With) { $uvArgs += @('--with', $package) }
$uvArgs += @('python', $Script)

& docker @dockerArgs @uvArgs @Arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
