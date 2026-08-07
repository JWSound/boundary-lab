@echo off
setlocal EnableExtensions DisableDelayedExpansion

cls
echo ============================================================
echo Boundary Lab launcher
echo ============================================================
echo.

for %%I in ("%~dp0.") do set "BASE_DIR=%%~fI"
set "PROJECT_DIR="

if exist "%BASE_DIR%\pyproject.toml" if exist "%BASE_DIR%\src\blab" goto PROJECT_HERE
if exist "%BASE_DIR%\boundary-lab\pyproject.toml" if exist "%BASE_DIR%\boundary-lab\src\blab" set "PROJECT_DIR=%BASE_DIR%\boundary-lab"
if defined PROJECT_DIR goto PROJECT_FOUND

echo ERROR: Could not find the Boundary Lab project folder.
echo Place this launcher inside the repository or directly above it.
echo.
pause
exit /b 1

:PROJECT_HERE
set "PROJECT_DIR=%BASE_DIR%"

:PROJECT_FOUND
cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not enter the Boundary Lab project folder:
    echo   %PROJECT_DIR%
    echo.
    pause
    exit /b 1
)

echo Using Boundary Lab project folder:
echo   %PROJECT_DIR%
echo.

set "BLAB_EXE=%PROJECT_DIR%\.venv\Scripts\blab.exe"
set "CONFIG_FILE=%PROJECT_DIR%\boundary_lab_gpu_choice.txt"

if exist "%BLAB_EXE%" goto INSTALL_FOUND
echo ERROR: Boundary Lab is not installed in this checkout.
echo Run 01_install_update_boundary-lab.bat first.
echo.
pause
exit /b 1

:INSTALL_FOUND
where nvidia-smi >nul 2>&1
if errorlevel 1 goto START_BLAB

if /i "%~1"=="/choose" goto CHOOSE_GPU
if not exist "%CONFIG_FILE%" goto CHOOSE_GPU

call :LOAD_SAVED_GPU
if errorlevel 2 goto CHOOSE_GPU
if errorlevel 1 goto START_BLAB
goto START_BLAB

:CHOOSE_GPU
echo Available NVIDIA GPUs:
echo ------------------------------------------------------------
nvidia-smi --query-gpu=index,name,uuid --format=csv,noheader
echo ------------------------------------------------------------
echo.
echo Enter a GPU index from the list, or press Enter for automatic selection.
set "GPU_CHOICE="
set /p "GPU_CHOICE=GPU index: "

if defined GPU_CHOICE goto RESOLVE_GPU
> "%CONFIG_FILE%" echo auto
echo.
echo GPU selection: automatic
echo Saved choice for the next launch.
echo.
goto START_BLAB

:RESOLVE_GPU
call :FIND_GPU_BY_INDEX
if defined SELECTED_UUID goto SAVE_SELECTED_GPU
echo.
echo ERROR: "%GPU_CHOICE%" is not an available GPU index.
echo.
pause
exit /b 1

:SAVE_SELECTED_GPU
set "CUDA_VISIBLE_DEVICES=%SELECTED_UUID%"
> "%CONFIG_FILE%" echo %SELECTED_UUID%
echo.
echo GPU selection: %SELECTED_NAME%
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo Saved choice for the next launch.
echo.

:START_BLAB
echo Starting Boundary Lab...
echo.
"%BLAB_EXE%" gui
set "BLAB_EXIT=%ERRORLEVEL%"

echo.
if not "%BLAB_EXIT%"=="0" (
    echo ERROR: Boundary Lab exited with status %BLAB_EXIT%.
    echo.
    pause
)
exit /b %BLAB_EXIT%

:LOAD_SAVED_GPU
set "SAVED_GPU="
set /p "SAVED_GPU=" < "%CONFIG_FILE%"
setlocal EnableDelayedExpansion
if /i "!SAVED_GPU!"=="auto" (
    echo GPU selection: automatic, saved
    echo.
    endlocal
    exit /b 1
)

set "FOUND_GPU="
for /f "tokens=1,2 delims=," %%A in ('nvidia-smi --query-gpu^=index^,uuid --format^=csv^,noheader 2^>nul') do (
    set "CANDIDATE_UUID=%%B"
    for /f "tokens=*" %%U in ("!CANDIDATE_UUID!") do set "CANDIDATE_UUID=%%U"
    if /i "!CANDIDATE_UUID!"=="!SAVED_GPU!" set "FOUND_GPU=1"
)

if not defined FOUND_GPU (
    echo The saved GPU is no longer available. Please choose again.
    echo.
    endlocal
    exit /b 2
)

endlocal & set "CUDA_VISIBLE_DEVICES=%SAVED_GPU%"
echo GPU selection: saved GPU
echo CUDA_VISIBLE_DEVICES=%CUDA_VISIBLE_DEVICES%
echo.
exit /b 0

:FIND_GPU_BY_INDEX
set "SELECTED_UUID="
set "SELECTED_NAME="
setlocal EnableDelayedExpansion
set "MATCHED_NAME="
set "MATCHED_UUID="
for /f "tokens=1,2,* delims=," %%A in ('nvidia-smi --query-gpu^=index^,name^,uuid --format^=csv^,noheader 2^>nul') do (
    set "CANDIDATE_INDEX=%%A"
    set "CANDIDATE_NAME=%%B"
    set "CANDIDATE_UUID=%%C"
    for /f "tokens=*" %%I in ("!CANDIDATE_INDEX!") do set "CANDIDATE_INDEX=%%I"
    for /f "tokens=*" %%N in ("!CANDIDATE_NAME!") do set "CANDIDATE_NAME=%%N"
    for /f "tokens=*" %%U in ("!CANDIDATE_UUID!") do set "CANDIDATE_UUID=%%U"
    if "!CANDIDATE_INDEX!"=="!GPU_CHOICE!" (
        set "MATCHED_NAME=!CANDIDATE_NAME!"
        set "MATCHED_UUID=!CANDIDATE_UUID!"
    )
)
endlocal & set "SELECTED_NAME=%MATCHED_NAME%" & set "SELECTED_UUID=%MATCHED_UUID%"
exit /b 0
