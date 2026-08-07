@echo off
setlocal EnableExtensions DisableDelayedExpansion

if /i "%~1"=="/internal" goto INTERNAL_START

for %%I in ("%~dp0.") do set "ORIGINAL_BASE_DIR=%%~fI"
set "TEMP_SCRIPT=%TEMP%\boundary_lab_install_update_%RANDOM%_%RANDOM%.bat"
copy /y "%~f0" "%TEMP_SCRIPT%" >nul
if errorlevel 1 (
    echo ERROR: Could not create a temporary updater copy:
    echo   "%TEMP_SCRIPT%"
    echo.
    pause
    exit /b 1
)

rem Parse this entire block before the temporary updater can replace this file.
setlocal EnableDelayedExpansion
(
    call "!TEMP_SCRIPT!" /internal "!ORIGINAL_BASE_DIR!"
    set "RUN_EXIT=!ERRORLEVEL!"
    del /q "!TEMP_SCRIPT!" >nul 2>&1
    exit /b !RUN_EXIT!
)

:INTERNAL_START
cls
echo ============================================================
echo Boundary Lab install / update / repair
echo ============================================================
echo.

for %%I in ("%~2") do set "BASE_DIR=%%~fI"
set "PROJECT_DIR="
set "REPO_URL=https://github.com/JWSound/boundary-lab.git"
set "TARGET_BRANCH=main"
set "HAS_WINGET=0"
set "OPTIONAL_SOLVER_WARNING=0"

where winget >nul 2>&1
if not errorlevel 1 set "HAS_WINGET=1"

rem Prefer the checkout containing this script. Standalone copies install into
rem a boundary-lab child directory instead.
if exist "%BASE_DIR%\pyproject.toml" if exist "%BASE_DIR%\src\blab" goto PROJECT_HERE
set "PROJECT_DIR=%BASE_DIR%\boundary-lab"
goto PROJECT_LOCATED

:PROJECT_HERE
set "PROJECT_DIR=%BASE_DIR%"

:PROJECT_LOCATED
echo Boundary Lab project folder:
echo   %PROJECT_DIR%
echo.

if exist "%PROJECT_DIR%\.git" goto EXISTING_GIT_CHECKOUT
if exist "%PROJECT_DIR%\pyproject.toml" if exist "%PROJECT_DIR%\src\blab" goto SOURCE_ARCHIVE
if exist "%PROJECT_DIR%" goto PROJECT_COLLISION

echo No checkout was found. Boundary Lab will be cloned from the latest
echo %TARGET_BRANCH% branch.
echo.
call :ENSURE_GIT
if errorlevel 2 goto RERUN_REQUIRED
if errorlevel 1 goto FAILED

git clone --branch "%TARGET_BRANCH%" --single-branch "%REPO_URL%" "%PROJECT_DIR%"
if errorlevel 1 (
    echo.
    echo ERROR: Boundary Lab could not be cloned.
    goto FAILED
)
goto PROJECT_READY

:EXISTING_GIT_CHECKOUT
echo Existing Git checkout detected.
call :ASK_YN "Pull the latest origin/%TARGET_BRANCH% update"
if /i not "%ANSWER%"=="Y" goto PROJECT_READY
call :UPDATE_MAIN
if errorlevel 2 goto RERUN_REQUIRED
if errorlevel 1 goto FAILED
goto PROJECT_READY

:SOURCE_ARCHIVE
echo This is a source checkout without Git metadata.
echo Update checks are unavailable, but installation and repair can continue.
echo.
goto PROJECT_READY

:PROJECT_COLLISION
echo ERROR: The target folder exists but is not a Boundary Lab checkout:
echo   %PROJECT_DIR%
echo.
echo Move the folder or place this script inside a Boundary Lab checkout.
goto FAILED

:PROJECT_READY
if not exist "%PROJECT_DIR%\pyproject.toml" (
    echo ERROR: The Boundary Lab checkout is incomplete. pyproject.toml is missing.
    goto FAILED
)

cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not enter the Boundary Lab project folder.
    goto FAILED
)

call :ENSURE_PYTHON
if errorlevel 2 goto RERUN_REQUIRED
if errorlevel 1 goto FAILED

echo Using Python:
echo   %PYTHON_CMD%
%PYTHON_CMD% --version
echo.

set "VENV_PY=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "BLAB_EXE=%PROJECT_DIR%\.venv\Scripts\blab.exe"

if exist "%VENV_PY%" goto VALIDATE_VENV
echo Creating Python virtual environment...
%PYTHON_CMD% -m venv "%PROJECT_DIR%\.venv"
if errorlevel 1 (
    echo ERROR: Failed to create the Python virtual environment.
    goto FAILED
)
goto INSTALL_PYTHON_PACKAGES

:VALIDATE_VENV
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: The existing .venv uses Python older than 3.11.
    echo Remove "%PROJECT_DIR%\.venv" and run this script again.
    goto FAILED
)
echo Existing Python virtual environment found.

:INSTALL_PYTHON_PACKAGES
echo.
echo Updating pip and installing / repairing Boundary Lab...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip could not be updated.
    goto FAILED
)

"%VENV_PY%" -m pip install -e ".[gui]"
if errorlevel 1 (
    echo ERROR: Boundary Lab's Python packages could not be installed.
    goto FAILED
)

"%VENV_PY%" -m pip check
if errorlevel 1 (
    echo ERROR: The Python environment contains incompatible packages.
    goto FAILED
)

if not exist "%BLAB_EXE%" (
    echo ERROR: The installation did not create "%BLAB_EXE%".
    goto FAILED
)

"%BLAB_EXE%" --help >nul 2>&1
if errorlevel 1 (
    echo ERROR: The Boundary Lab command-line entry point did not start correctly.
    goto FAILED
)

echo.
echo Python application setup completed successfully.
echo.
call :ASK_YN "Install or update the optional Julia-based BEAT Engine solvers"
if /i not "%ANSWER%"=="Y" goto OPTIONAL_SOLVERS_DONE

call :ENSURE_JULIA
if errorlevel 2 goto RERUN_REQUIRED
if errorlevel 1 goto OPTIONAL_SOLVER_FAILED

echo.
echo Installing / updating the BEAT Engine CPU environment...
julia --project="%PROJECT_DIR%\src\blab\solvers\julia_local" -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
if errorlevel 1 (
    echo ERROR: The BEAT Engine CPU environment could not be prepared.
    goto OPTIONAL_SOLVER_FAILED
)

echo.
where nvidia-smi >nul 2>&1
if errorlevel 1 goto CUDA_NOT_DETECTED
echo An NVIDIA driver and GPU were detected.
call :ASK_YN "Install or update NVIDIA CUDA solver support"
goto CUDA_DECISION

:CUDA_NOT_DETECTED
echo No NVIDIA driver was detected. CUDA support is not required for CPU solving.
call :ASK_YN "Install the CUDA environment anyway"

:CUDA_DECISION
if /i not "%ANSWER%"=="Y" goto OPTIONAL_SOLVERS_DONE

echo.
echo Installing / updating the BEAT Engine CUDA environment...
julia --project="%PROJECT_DIR%\src\blab\solvers\julia_cuda" -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
if errorlevel 1 (
    echo ERROR: The BEAT Engine CUDA environment could not be prepared.
    goto OPTIONAL_SOLVER_FAILED
)

where nvidia-smi >nul 2>&1
if errorlevel 1 goto OPTIONAL_SOLVERS_DONE

echo.
echo Verifying CUDA access from Julia...
julia --project="%PROJECT_DIR%\src\blab\solvers\julia_cuda" -e "using CUDA; CUDA.functional() || error(\"CUDA is not functional\"); CUDA.versioninfo()"
if not errorlevel 1 goto OPTIONAL_SOLVERS_DONE

echo.
echo WARNING: CUDA packages were installed, but the CUDA runtime check failed.
echo Boundary Lab and the CPU solvers can still be used.
echo.
set "OPTIONAL_SOLVER_WARNING=1"
pause
goto OPTIONAL_SOLVERS_DONE

:OPTIONAL_SOLVER_FAILED
set "OPTIONAL_SOLVER_WARNING=1"
echo.
echo WARNING: Optional Julia solver setup did not complete.
echo The Boundary Lab application and Bempp CPU solver are installed and usable.
echo Run this script again when you are ready to retry the optional setup.
echo.

:OPTIONAL_SOLVERS_DONE
set "LAUNCHER=%PROJECT_DIR%\02_start_boundary_lab.bat"
if not exist "%LAUNCHER%" (
    echo ERROR: The Boundary Lab launcher is missing:
    echo   %LAUNCHER%
    goto FAILED
)

echo.
echo ============================================================
if "%OPTIONAL_SOLVER_WARNING%"=="1" (
    echo Boundary Lab is ready; optional solver setup needs attention.
) else (
    echo Boundary Lab installation / update completed successfully.
)
echo ============================================================
echo.
echo Start Boundary Lab with:
echo   %LAUNCHER%
echo.
pause
exit /b 0

:RERUN_REQUIRED
echo.
echo A prerequisite was installed. Close this window and run this script again
echo so Windows can refresh the available commands.
echo.
pause
exit /b 0

:FAILED
echo.
echo Boundary Lab installation / update did not complete.
echo Review the error above, then run this script again.
echo.
pause
exit /b 1

:ASK_YN
set "ANSWER=N"
set "RESPONSE="
setlocal EnableDelayedExpansion
set "RESULT=N"
set /p "RESPONSE=%~1? [y/N]: "
if /i "!RESPONSE!"=="Y" set "RESULT=Y"
if /i "!RESPONSE!"=="YES" set "RESULT=Y"
endlocal & set "ANSWER=%RESULT%"
echo.
exit /b 0

:ENSURE_GIT
where git >nul 2>&1
if not errorlevel 1 exit /b 0
echo Git is required to download or update Boundary Lab.
if not "%HAS_WINGET%"=="1" (
    echo Install Git for Windows, then run this script again.
    exit /b 1
)
call :ASK_YN "Install Git for Windows with winget"
if /i not "%ANSWER%"=="Y" exit /b 1
winget install --id Git.Git --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo ERROR: winget could not install Git.
    exit /b 1
)
exit /b 2

:ENSURE_PYTHON
set "PYTHON_CMD="
py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=py -3"
if defined PYTHON_CMD exit /b 0

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python"
if defined PYTHON_CMD exit /b 0

python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_CMD=python3"
if defined PYTHON_CMD exit /b 0

echo Python 3.11 or newer was not found.
if not "%HAS_WINGET%"=="1" (
    echo Install Python 3.11 or newer, then run this script again.
    exit /b 1
)
call :ASK_YN "Install Python 3.11 with winget"
if /i not "%ANSWER%"=="Y" exit /b 1
winget install --id Python.Python.3.11 --exact --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo ERROR: winget could not install Python.
    exit /b 1
)
exit /b 2

:ENSURE_JULIA
julia --version >nul 2>&1
if not errorlevel 1 (
    julia --version
    exit /b 0
)
echo Julia was not found.
if not "%HAS_WINGET%"=="1" (
    echo Install Julia and make it available on PATH, then run this script again.
    exit /b 1
)
call :ASK_YN "Install Julia with winget"
if /i not "%ANSWER%"=="Y" exit /b 1
winget install --id 9NJNWW8PVKMN --exact --source msstore --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo ERROR: winget could not install Julia.
    exit /b 1
)
exit /b 2

:UPDATE_MAIN
call :ENSURE_GIT
if errorlevel 2 exit /b 2
if errorlevel 1 exit /b 1

pushd "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Could not enter the Git checkout.
    exit /b 1
)

set "CURRENT_BRANCH="
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if /i not "%CURRENT_BRANCH%"=="%TARGET_BRANCH%" goto UPDATE_WRONG_BRANCH

git diff --quiet
if errorlevel 1 goto UPDATE_DIRTY
git diff --cached --quiet
if errorlevel 1 goto UPDATE_DIRTY

echo Pulling the latest origin/%TARGET_BRANCH%...
git pull --ff-only origin "%TARGET_BRANCH%"
if errorlevel 1 goto UPDATE_FAILED
popd
echo.
exit /b 0

:UPDATE_WRONG_BRANCH
echo ERROR: Automatic updates are only supported on the %TARGET_BRANCH% branch.
echo Current branch: %CURRENT_BRANCH%
echo Switch branches manually or decline the update to repair this checkout.
popd
exit /b 1

:UPDATE_DIRTY
echo ERROR: Tracked files contain local changes. The update was not attempted.
echo Commit or restore those changes, or decline the update to repair in place.
popd
exit /b 1

:UPDATE_FAILED
echo ERROR: Git could not fast-forward to origin/%TARGET_BRANCH%.
popd
exit /b 1
