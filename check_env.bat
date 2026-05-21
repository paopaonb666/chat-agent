@echo off
chcp 936 >nul
setlocal enabledelayedexpansion

echo ============================================
echo   Chat Agent - �������ű�
echo ============================================
echo.

set PASS=0
set FAIL=0

rem ---------- 1. PostgreSQL ----------
echo [1/6] PostgreSQL
where pg_isready >nul 2>&1
if !errorlevel! equ 0 (
    pg_isready -h localhost -p 5432 >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] PostgreSQL is running on localhost:5432
        set /a PASS+=1
    ) else (
        echo   [FAIL] PostgreSQL port 5432 not responding
        set /a FAIL+=1
    )
) else (
    netstat -an | findstr ":5432 " >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] PostgreSQL port 5432 is open
        set /a PASS+=1
    ) else (
        echo   [FAIL] PostgreSQL not detected (pg_isready not found, port 5432 not listening)
        set /a FAIL+=1
    )
)
echo.

rem ---------- 2. Milvus ----------
echo [2/6] Milvus
curl -s --connect-timeout 3 http://localhost:19530 >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Milvus is running on localhost:19530
    set /a PASS+=1
) else (
    echo   [FAIL] Milvus port 19530 not responding
    set /a FAIL+=1
)
echo.

rem ---------- 3. Ollama ----------
echo [3/6] Ollama
curl -s --connect-timeout 3 http://localhost:11434 >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Ollama is running on localhost:11434

    set MODELS_MISSING=0
    for %%m in ("qwen2.5:0.5b" "qwen3-embedding:0.6b" "pdurugyan/qwen3-reranker-0.6b-q8_0") do (
        curl -s http://localhost:11434/api/tags 2>nul | findstr /C:"%%~m" >nul 2>&1
        if !errorlevel! equ 0 (
            echo     [OK] Model %%~m is available
        ) else (
            echo     [MISS] Model %%~m not pulled (run: ollama pull %%~m)
            set /a MODELS_MISSING+=1
        )
    )
    if !MODELS_MISSING! gtr 0 (
        set /a FAIL+=1
    ) else (
        set /a PASS+=1
    )
) else (
    echo   [FAIL] Ollama port 11434 not responding
    set /a FAIL+=1
)
echo.

rem ---------- 4. Backend (FastAPI) ----------
echo [4/6] Backend API
curl -s --connect-timeout 3 http://localhost:8000/docs >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Backend is running on localhost:8000
    set /a PASS+=1
) else (
    echo   [FAIL] Backend port 8000 not responding (run: run.bat in backend/)
    set /a FAIL+=1
)
echo.

rem ---------- 5. Frontend (Vite) ----------
echo [5/6] Frontend
curl -s --connect-timeout 3 http://localhost:5173 >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Frontend is running on localhost:5173
    set /a PASS+=1
) else (
    echo   [FAIL] Frontend port 5173 not responding (run: npm run start in frontend/)
    set /a FAIL+=1
)
echo.

rem ---------- 6. Proxy (optional) ----------
echo [6/6] DuckDuckGo Proxy (optional)
curl -s --connect-timeout 2 http://127.0.0.1:7890 >nul 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Proxy detected on 127.0.0.1:7890
) else (
    echo   [SKIP] No proxy on 127.0.0.1:7890 (optional, needed for DuckDuckGo in China)
)
set /a PASS+=1
echo.

rem ---------- Summary ----------
echo ============================================
if !FAIL! equ 0 (
    echo   All services are running! (PASS/6)
    echo.
    echo   Frontend: http://localhost:5173
    echo   Backend:  http://localhost:8000/docs
) else (
    echo   %FAIL% service(s) have issues
    echo   %PASS% service(s) OK
)
echo ============================================
echo.
pause
