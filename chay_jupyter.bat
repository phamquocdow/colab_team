@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [1/2] Tao moi truong ao .venv ...
    py -3.11 -m venv .venv || python -m venv .venv
    if errorlevel 1 (
        echo Khong tao duoc .venv - hay cai Python 3.11 tu python.org roi chay lai.
        pause
        exit /b 1
    )
    echo [2/2] Cai thu vien trong requirements.txt ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Cai thu vien that bai - xem loi o tren.
        pause
        exit /b 1
    )
)

echo Mo JupyterLab - chay lan luot notebook 1 -^> 5 ...
".venv\Scripts\python.exe" -m jupyter lab 1_setup_moi_truong.ipynb
