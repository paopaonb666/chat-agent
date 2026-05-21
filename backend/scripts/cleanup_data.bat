@echo off
chcp 936 >nul
echo ============================================
echo   Chat Agent - ������������
echo ============================================
echo.

cd /d "%~dp0.."

if not exist ".env" (
    echo [ERROR] .env �ļ������ڣ������� backend/ Ŀ¼�´���
    pause
    exit /b 1
)

echo �������� PostgreSQL + Milvus ��������...
echo.

:confirm
echo �˲�����ɾ�����жԻ�����Ϣ���ļ����ݣ����ɻָ���
set /p input="���� YES ȷ��: "
if /i not "%input%"=="YES" (
    echo ��ȡ��
    pause
    exit /b 0
)

venv\Scripts\python scripts\cleanup_data.py --force
if errorlevel 1 (
    echo [ERROR] ����ʧ�ܣ��������̨���
) else (
    echo ������ɣ�
)

pause
