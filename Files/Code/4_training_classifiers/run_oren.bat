:: print some notes
@echo off
echo User:   Oren
echo Script: generate_database
echo Note1:  if failed - run as administrator or check paths!
echo ...

:: 1. call anaconda prompt
echo 1. call anaconda prompt
call "D:\Softwares\Anaconda3\Scripts\activate.bat"

:: 2. activate opencv library (according to computer OS)
:: 3. set project directory (according to computer OS)
echo 2. activate opencv library
echo 3. set project directory
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
if "%version%" == "10.0" (
call activate "opencv-env"
cd "C:\Users\orens\Documents\GitHub\BioMed_Final_Project\Files\Code\4_training_classifiers"
) else (
call activate "opencv-envpython=3.6"
cd "C:\Users\אורן\Documents\GitHub\BioMed_Final_Project\Files\Code\4_training_classifiers")

:: 4. get inputs from user

:: 5. set relevant paths

:: 6. run scripts
echo 6. run script
echo ...
python generate_database.py --user OREN
pause