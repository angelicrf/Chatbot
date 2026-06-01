@echo off
setlocal
set SCRIPT_DIR=%~dp0
python "%SCRIPT_DIR%build.py" %*
if %ERRORLEVEL% neq 0 (
  py "%SCRIPT_DIR%build.py" %*
)
endlocal
