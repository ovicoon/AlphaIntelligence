
D:
cd /d "D:\Users\ohver\python\AlphaIntelligence\"

set /p VERSION=Enter version (ex: 1.0.0): 

if "%VERSION%"=="" (
    echo [ERROR] Version not entered.
    pause
    exit /b
)

python -m nuitka ^
--onefile ^
--lto=yes ^
--company-name="Zero Dragon" ^
--product-name="AlphaIntelligence" ^
--file-version=%VERSION% ^
--product-version=%VERSION% ^
--copyright="© 2026 Zero Dragon" ^
--output-filename=AlphaIntelligence.exe ^
"%CD%\ProtoType\alpha_intelligence.py

echo.
echo ============================
echo Build Finished!
echo ============================
pause