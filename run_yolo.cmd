@echo off
chcp 65001 >nul
title YOLOv10 Detect
set "BASE=D:\YOLO"
set "PY=%BASE%\env\Scripts\python.exe"
set "INBOX=%BASE%\work\inbox"
set "OUT=%BASE%\work\out"
if not exist "%INBOX%" mkdir "%INBOX%"
if "%~1"=="" goto run
echo Copying dropped files to inbox ...
:copyone
copy /y "%~1" "%INBOX%\" >nul
shift
if not "%~1"=="" goto copyone
:run
echo.
echo   inbox : %INBOX%
echo   out   : %OUT%
echo.
"%PY%" "%BASE%\scripts\annotate_mine.py"
echo.
start "" "%OUT%"
echo Done.
pause
