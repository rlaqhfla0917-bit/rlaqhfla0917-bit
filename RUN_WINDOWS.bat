@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/4] Python virtual environment
py -3 -m venv .venv
if errorlevel 1 goto :failed
call .venv\Scripts\activate.bat

echo [2/4] Installing dependencies
python -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [3/4] Downloading official UCI SECOM data and training
python scripts/download_data.py
if errorlevel 1 goto :failed
python train.py
if errorlevel 1 goto :failed

echo [4/4] Opening Streamlit dashboard
python -m streamlit run app.py
exit /b 0

:failed
echo.
echo An earlier step failed. See errors above.
echo You may also open the synthetic preview using: python -m streamlit run app.py
pause
exit /b 1
