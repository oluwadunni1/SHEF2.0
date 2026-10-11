@echo off
REM Daily forecast snapshot (Windows Task Scheduler). Moves to GitHub Actions in P5.
cd /d "%~dp0.."
if not exist data\reports mkdir data\reports
"%USERPROFILE%\.local\bin\uv.exe" run python scripts\snapshot_forecast.py --only-at-issue >> data\reports\snapshot_forecast.log 2>&1
