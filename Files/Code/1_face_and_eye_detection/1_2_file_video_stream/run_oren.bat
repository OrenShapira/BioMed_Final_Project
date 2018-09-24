:: print some notes
@echo off
echo User:   Oren
echo Script: file_video_stream_read_x (x = fast/slow)
echo Note1:  if failed - run as administrator or check paths!
echo Note2:  press ESC on keyboard to break video processing
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
cd "C:\Users\orens\Documents\GitHub\BioMed_Final_Project\Files"
) else (
call activate "opencv-envpython=3.6"
cd "C:\Users\אורן\Documents\GitHub\BioMed_Final_Project\Files")

:: 4. get inputs from user
echo 4. get inputs from user
set /p script_kind=...enter kind of script to run (format fast/slow only)? ENTER fast/slow =  
set /p input_index=...enter input index to run (format video_xx.mp4 only)? ENTER xx = 

:: 5. set relevant paths
echo 5. set relevant paths
set script_path="Code\1_face_and_eye_detection\1_2_file_video_stream\file_video_stream_read_%script_kind%.py"
set input_path="Code\1_face_and_eye_detection\1_2_file_video_stream\files\input\video_%input_index%.mp4"
set output_path="Code\1_face_and_eye_detection\1_2_file_video_stream\files\output\video_%input_index%_%script_kind%_out.mp4"

:: 6. run scripts
echo 6. run script
echo ...
python %script_path% --video %input_path% --output %output_path% --output_fps 25
pause
