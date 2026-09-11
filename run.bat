@echo off
echo Starting Sign Language Interpreter...
cd /d "%~dp0Code"
"..\\venv\\Scripts\\python.exe" final.py
pause
