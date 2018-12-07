:: print some notes
@echo off
echo User:   Tom
echo Script: database_tagging
echo Note1:  if failed - run as administrator or check paths!
echo ...

:: 1. call anaconda prompt
echo 1. call anaconda prompt
call "C:\ProgramData\Anaconda3\Scripts\activate.bat"

:: 2. activate opencv library (according to computer OS)
:: 3. set project directory (according to computer OS)
echo 2. activate opencv library
echo 3. set project directory
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
if "%version%" == "10.0" (
call activate "opencv-env"
cd "C:\Users\user\Technion\ME - Biomedical Engineering\Project\Code_Git\biomed_final_project\Files\Code\3_database_tagging"
) else (
call activate "opencv-envpython=3.6"
cd "C:\Users\אורן\Documents\GitHub\BioMed_Final_Project\Files\Code\3_database_tagging")

:: 4. get inputs from user

:: 5. set relevant paths

:: 6. run scripts
echo 6. run script
echo ...
python database_tagging.py
pause