param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('probe', 'lock', 'sync', 'pytest', 'evaluate', 'report', 'python')]
    [string]$Action,

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
if ($LASTEXITCODE -ne 0 -or $corpusGitSha -notmatch '^[0-9a-f]{40}$') {
    throw "failed to resolve corpus Git SHA from $corpusRoot"
}

$dockerArgs = @(
    'run', '--rm',
    '--memory', '4g', '--memory-swap', '4g',
    '--cpus', '2', '--pids-limit', '512',
    '--mount', "type=bind,source=$repoRoot,target=/workspace",
    '--mount', "type=bind,source=$corpusRoot,target=/corpus,readonly",
    '--mount', "type=bind,source=$cacheRoot,target=/cache",
    '--env', 'HF_HOME=/cache/huggingface',
    '--env', 'UV_CACHE_DIR=/cache/uv',
    '--env', 'UV_PROJECT_ENVIRONMENT=/cache/venv',
    '--env', 'MPLCONFIGDIR=/cache/matplotlib',
    '--env', 'PYTHONPATH=/workspace/src',
    '--env', "SPLADE_CORPUS_GIT_SHA=$corpusGitSha",
    '--workdir', '/workspace',
    'ghcr.io/astral-sh/uv:python3.12-bookworm-slim'
)

$containerCommand = switch ($Action) {
    'probe' { @('python', 'scripts/container_probe.py') }
    'lock' { @('uv', 'lock') }
    'sync' { @('uv', 'sync', '--locked') }
    'pytest' { @('uv', 'run', '--no-sync', 'pytest') + $Arguments }
    'evaluate' { @('uv', 'run', '--no-sync', 'python', '-m', 'splade_poc.evaluate') + $Arguments }
    'report' { @('uv', 'run', '--no-sync', 'python', '-m', 'splade_poc.report') + $Arguments }
    'python' { @('uv', 'run', '--no-sync', 'python') + $Arguments }
}

& docker @dockerArgs @containerCommand
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
