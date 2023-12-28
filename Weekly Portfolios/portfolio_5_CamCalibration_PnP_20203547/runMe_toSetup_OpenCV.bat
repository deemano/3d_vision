@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

:: Set the code page to UTF-8
chcp 65001 > nul

:: Get the current directory of the batch script, including the trailing backslash
SET "scriptPath=%~dp0"

:: Create a libs directory if it doesn't exist
IF NOT EXIST "%scriptPath%libs\" (
    MKDIR "%scriptPath%libs"
)

:: Define the URL to download OpenCV from
SET "opencvUrl=https://github.com/opencv/opencv/releases/download/4.8.0/opencv-4.8.0-windows.exe"
:: Define the URL to download Eigen3 from
SET "eigenUrl=https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"

:: Define the filenames for the downloaded files
SET "opencvFileName=opencv_installer.exe"
SET "eigenFileName=eigen.zip"

:: Inform the user of the expected download size for OpenCV
echo.
echo. 
echo.
echo.
echo. 
echo.
echo.
echo  Downloading OpenCV, please wait. Expected file size: ~ 176000000 (176 MB).
echo  ^Copyright: Denis Manolescu - Liverpool Hope University, 2024

:: Download OpenCV using PowerShell
echo.
PowerShell -Command "Invoke-WebRequest -Uri '%opencvUrl%' -OutFile '%scriptPath%libs\%opencvFileName%' -UseBasicParsing"

:: Check if the OpenCV download was successful before attempting to execute the installer
IF EXIST "%scriptPath%libs\%opencvFileName%" (
    ECHO.
    ECHO  An Admin extraction request will pop-up, please confirm it and approve the extraction.
    :: Execute the installer
    START /wait "" "%scriptPath%libs\%opencvFileName%" /VERYSILENT /DIR="%scriptPath%libs\opencv"
    ECHO  Now extracting OpenCV ...

    :: Delete the installer if no longer needed
    ECHO  Cleaning up ...
    DEL "%scriptPath%libs\%opencvFileName%"
) ELSE (
    ECHO  OpenCV download failed, installer not found.
    PAUSE
)

:: Inform the user that the script will now download Eigen3
echo  Now downloading Eigen3 library...

:: Download Eigen3 using PowerShell
PowerShell -Command "Invoke-WebRequest -Uri '%eigenUrl%' -OutFile '%scriptPath%libs\%eigenFileName%' -UseBasicParsing"

:: Check if the Eigen3 download was successful before attempting to extract
IF EXIST "%scriptPath%libs\%eigenFileName%" (
    :: Extract the zip file
    ECHO  Now extracting Eigen Lib ...
    PowerShell -Command "Expand-Archive -LiteralPath '%scriptPath%libs\%eigenFileName%' -DestinationPath '%scriptPath%libs\eigen' -Force"

    :: Delete the zip file if no longer needed
    DEL "%scriptPath%libs\%eigenFileName%"
) ELSE (
    ECHO  Eigen3 download failed, zip file not found.
    PAUSE
)

echo  Setup completed. Visual Studio script should now have all dependencies in place.
echo.  
echo  This message will self-destruct 10 seconds...
timeout /t 10 /nobreak
:: End of script
ENDLOCAL
