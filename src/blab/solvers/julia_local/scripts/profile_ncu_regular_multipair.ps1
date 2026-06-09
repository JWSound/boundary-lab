$ErrorActionPreference = "Stop"

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
$Julia = "C:\Users\John\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe"
$Benchmark = Join-Path $ScriptDir "benchmark_cuda.jl"
$Mesh = Join-Path $ScriptDir "..\test_meshes\sample_detailed.msh"
$Results = Join-Path $ScriptDir "..\results"
$Report = Join-Path $Results "ncu_regular_multipair_16x8"
$Json = Join-Path $Results "benchmark_cuda_ncu_regular_multipair_16x8.json"

New-Item -ItemType Directory -Force -Path $Results | Out-Null

& ncu `
    --target-processes all `
    --section LaunchStats `
    --section Occupancy `
    --section SpeedOfLight `
    --section SchedulerStats `
    --section MemoryWorkloadAnalysis `
    --kernel-name "regex:_cuda_regular_quadrature_.*_kernel_" `
    --launch-skip 6 `
    --launch-count 2 `
    --force-overwrite `
    --export $Report `
    -- "$Julia" "$Benchmark" `
        --mesh "$Mesh" `
        --warmups 3 `
        --repetitions 1 `
        --regular-assembly-mode multipair `
        --json "$Json"

Write-Host "Nsight Compute report: $Report.ncu-rep"
Write-Host "Benchmark JSON: $Json"
