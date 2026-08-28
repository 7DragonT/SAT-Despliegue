@echo off
setlocal

echo ==========================================
echo Sistema de Alerta Temprana - SAT
echo ==========================================
echo.

echo [1/4] Creando entorno virtual...
if not exist ".venv" (
    python -m venv .venv
) else (
    echo El entorno virtual ya existe.
)

echo.
echo [2/4] Activando entorno virtual...
call .venv\Scripts\activate.bat

echo.
echo [3/4] Instalando dependencias...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo [4/4] Iniciando API FastAPI...
start "SAT - API FastAPI" cmd /k "call .venv\Scripts\activate.bat && uvicorn app.main:app --host 127.0.0.1 --port 8000"

timeout /t 3 /nobreak >nul

echo.
echo Iniciando aplicacion Streamlit...
set API_URL=http://127.0.0.1:8000

streamlit run app_streamlit.py

pause
