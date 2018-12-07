setlocal enabledelayedexpansion

:: print some notes
@echo off
echo User:   Tom
echo Script: generate_database
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
cd "C:\Users\user\Technion\ME - Biomedical Engineering\Project\Code_Git\biomed_final_project\Files\Code\4_training_classifiers"
) else (
call activate "opencv-envpython=3.6"
cd "C:\Users\אורן\Documents\GitHub\BioMed_Final_Project\Files")

:: 4. run scripts
python generate_database.py --user TOM --exclude normal palsy 3 6 

pause
