$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path 'build' | Out-Null
    $compileArgs = @('-interaction=nonstopmode', '-halt-on-error', '-file-line-error', '-output-directory=build', 'SPRiF_TNNLS.tex')
    & pdflatex @compileArgs
    if ($LASTEXITCODE -ne 0) { throw 'Initial pdfLaTeX pass failed.' }
    & bibtex 'build/SPRiF_TNNLS'
    if ($LASTEXITCODE -ne 0) { throw 'BibTeX failed.' }
    foreach ($pass in 1..2) {
        & pdflatex @compileArgs
        if ($LASTEXITCODE -ne 0) { throw "Final pdfLaTeX pass $pass failed." }
    }
    Write-Output (Join-Path $PSScriptRoot 'build/SPRiF_TNNLS.pdf')
}
finally {
    Pop-Location
}
