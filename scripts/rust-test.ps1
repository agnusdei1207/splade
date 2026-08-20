param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet('lock', 'format', 'fmt', 'test', 'clippy', 'bench')]
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
    '--env', 'CARGO_HOME=/cache/cargo-home',
    '--env', 'CARGO_TARGET_DIR=/cache/cargo-target',
    '--env', 'CARGO_BUILD_JOBS=2',
    '--env', "SPLADE_CORPUS_GIT_SHA=$corpusGitSha",
    '--workdir', '/workspace',
    'rust:1.96-bookworm'
)

$cargoCommand = @('bash', '/workspace/scripts/rust-container.sh', $Action) + $Arguments

& docker @dockerArgs @cargoCommand
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
