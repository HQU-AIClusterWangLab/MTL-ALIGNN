. "$PSScriptRoot\common.ps1"
Invoke-TwoPython -Label "Run unit tests" -Arguments @(
    "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"
)
