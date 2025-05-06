@echo off
echo Face and Person Tracker
echo ======================
echo.

:: Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python 3.6+ and try again.
    pause
    exit /b 1
)

:: Run the setup script first to check environment
python setup.py

:: If the user chose to run the application directly from setup.py, we're done
if %ERRORLEVEL% equ 0 (
    exit /b 0
)

echo.
echo You can now run the application with custom options:
echo python face_tracker_app.py --config custom_config.yml
echo.

pause 