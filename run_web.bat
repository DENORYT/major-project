@echo off
echo Starting Sign Language Interpreter Web Server...
cd /d "%~dp0"
".\\venv\\Scripts\\python.exe" Code/app.py
pause
