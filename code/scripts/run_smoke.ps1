. "$PSScriptRoot\common.ps1"
Invoke-TwoPython -Label "Check fixed environment" -Arguments @("data/check_environment.py")
Invoke-TwoPython -Label "Inspect c2db.db fields" -Arguments @("data/inspect_db.py")
Invoke-TwoPython -Label "Prepare metadata and labels" -Arguments @("data/prepare_data.py")
Invoke-TwoPython -Label "Create leakage-controlled IID split" -Arguments @("data/make_splits.py")
Invoke-TwoPython -Label "Build eight smoke-test graphs" -Arguments @("data/build_graphs.py", "--subset", "smoke")
Invoke-TwoPython -Label "ALIGNN forward/backward smoke test" -Arguments @("train/smoke_test.py")
Write-Host "Smoke test completed successfully."
