setlocal enabledelayedexpansion

:: print some notes
@echo off
echo User:   Tom
echo Script: feature_extraction
echo Note1:  if failed - run as administrator or check paths!
echo Note2:  press ESC on keyboard to break video processing
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
cd "C:\Users\user\Technion\ME - Biomedical Engineering\Project\Code_Git\biomed_final_project\Files\Code\2_features_extraction"
) else (
call activate "opencv-envpython=3.6"
cd "C:\Users\אורן\Documents\GitHub\BioMed_Final_Project\Files")

:: 4. get inputs from user
echo 4. get inputs from user
set /p "input_kind=...Choose input kind? ENTER [v]ideo/[w]ebcam = "
set /p "run_kind=...run kind? ENTER []/[_full] = "
if %input_kind%==v (
	set /p "video_name=...Enter video name? ENTER normal/palsy_xx = "
	for /f "delims=" %%a in ('dir /s /b ..\..\Database\!video_name!*') do set "input_path=%%~nxa"
	:: 5. set relevant paths
	echo 5. set relevant paths
	:: 6. run scripts
	echo 6. run script
	echo ...
	python feature_extraction_video%run_kind%.py --input !input_path! --tag 1 --face_frame 25
)
if %input_kind%==w (
	:: 5. run scripts
	set /p "eye=...Choose palsy eye? ENTER [l]eft/[r]ight/[n]one = "
	echo 5. run script
	echo ...
	python feature_extraction_webcam%run_kind%.py --palsy_eye !eye! --tag 1 --face_frame 800
)


pause
