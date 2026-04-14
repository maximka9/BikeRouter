# Публикация каталога bike_router в https://github.com/maximka9/BikeRouter
# Запуск из корня монорепозитория Git (родитель каталога NIR), например:
#   cd R:\Python
#   .\NIR\bike_router\scripts\push_to_github_bike_router.ps1
#
# Требуется remote "bike-router" → BikeRouter.git (см. git remote -v).

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$gitRoot = Resolve-Path (Join-Path $here "..\..\..")
Set-Location $gitRoot

$prefix = "NIR/bike_router"
$branch = "_bike_router_publish_tmp"
$remote = "bike-router"

Write-Host "git root: $gitRoot"
Write-Host "subtree prefix: $prefix -> ${remote}:main"

git fetch $remote 2>$null
git subtree split --prefix=$prefix -b $branch
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git push $remote "${branch}:main" --force
if ($LASTEXITCODE -ne 0) { git branch -D $branch 2>$null; exit $LASTEXITCODE }

git branch -D $branch
Write-Host "OK: https://github.com/maximka9/BikeRouter (main)"
