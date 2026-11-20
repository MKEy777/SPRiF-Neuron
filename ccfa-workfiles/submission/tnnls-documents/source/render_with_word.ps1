param(
    [ValidateSet('Title_Page_TNNLS', 'Cover_Letter_TNNLS', 'Cover_Letter_TNNLS_Submission')]
    [string[]]$DocumentNames = @('Title_Page_TNNLS', 'Cover_Letter_TNNLS')
)

$ErrorActionPreference = 'Stop'
$submissionRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$submissionManuscript = Join-Path $submissionRoot 'TNNLS-Manuscript'
$submissionBuild = Join-Path (Split-Path -Parent $PSScriptRoot) 'build'
$word = $null
$ownsWord = $false
try {
    $word = New-Object -ComObject Word.Application
    if ($word.Documents.Count -ne 0) {
        throw 'The Word automation instance already contains documents; leaving it unchanged.'
    }
    $ownsWord = $true
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    foreach ($submissionItem in @(
        @{ File = 'Title_Page_TNNLS'; Folder = 'title-page' },
        @{ File = 'Cover_Letter_TNNLS'; Folder = 'cover-letter' },
        @{ File = 'Cover_Letter_TNNLS_Submission'; Folder = 'cover-letter' }
    )) {
        if ($DocumentNames -notcontains $submissionItem.File) { continue }
        $submissionOutput = Join-Path $submissionBuild $submissionItem.Folder
        New-Item -ItemType Directory -Path $submissionOutput -Force | Out-Null
        $submissionInput = Join-Path $submissionManuscript ($submissionItem.File + '.docx')
        $submissionPdf = Join-Path $submissionOutput ($submissionItem.File + '.pdf')
        $submissionDoc = $null
        try {
            $submissionDoc = $word.Documents.Open($submissionInput, $false, $true)
            $submissionDoc.Repaginate()
            $submissionPages = $submissionDoc.ComputeStatistics(2)
            $submissionDoc.ExportAsFixedFormat($submissionPdf, 17)
            Write-Output ($submissionItem.File + ': ' + $submissionPages + ' page(s)')
        }
        finally {
            if ($null -ne $submissionDoc) {
                $submissionDoc.Close(0)
                [void][Runtime.InteropServices.Marshal]::ReleaseComObject($submissionDoc)
            }
        }
    }
}
finally {
    if ($null -ne $word) {
        if ($ownsWord) { $word.Quit(0) }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
