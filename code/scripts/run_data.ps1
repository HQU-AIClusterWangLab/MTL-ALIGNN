. "$PSScriptRoot\common.ps1"
Invoke-TwoPython -Label "Check fixed environment" -Arguments @("data/check_environment.py")
Invoke-TwoPython -Label "Inspect c2db.db fields" -Arguments @("data/inspect_db.py")
Invoke-TwoPython -Label "Prepare metadata, labels and tabular features" -Arguments @("data/prepare_data.py")
Invoke-TwoPython -Label "Create leakage-controlled IID split" -Arguments @("data/make_splits.py")
Invoke-TwoPython -Label "Export data-analysis workbook" -Arguments @("data/analyze_data.py")
Invoke-TwoPython -Label "Build supervised ALIGNN graphs" -Arguments @("data/build_graphs.py", "--subset", "supervised")
Invoke-TwoPython -Label "Final graph/model smoke test" -Arguments @("train/smoke_test.py")
Write-Host "Data stage completed successfully."
