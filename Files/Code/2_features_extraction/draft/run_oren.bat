:: print some notes
@echo off
echo User:   Oren
echo Script: feature_extraction (jupyter notebook)
echo Note1:  if failed - run as administrator or check paths!
echo ...

:: 1. call anaconda prompt
echo 1. call anaconda prompt
call "D:\Softwares\Anaconda3\Scripts\activate.bat"

:: 2. set project directory (according to computer OS)
echo 2. set project directory
for /f "tokens=4-5 delims=. " %%i in ('ver') do set VERSION=%%i.%%j
if "%version%" == "10.0" (
cd "C:\Users\orens\Documents\GitHub\BioMed_Final_Project\Files\Code\2_features_extractions\draft"
) else (
cd "C:\Users\אורן\Documents\GitHub\BioMed_Final_Project\Files\Code\2_features_extractions\draft")

:: 3. start jupyter notebook
echo 3. start jupyter notebook
start jupyter notebook
