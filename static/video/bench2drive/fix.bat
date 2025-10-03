@echo off
setlocal enabledelayedexpansion

REM 输入视频目录
set INPUT_DIR=D:\recogdrive\static\video\bench2drive

REM 输出视频目录
set OUTPUT_DIR=%INPUT_DIR%\converted

REM 如果输出目录不存在则创建
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM 循环处理目录下所有 mp4 文件
for %%f in ("%INPUT_DIR%\*.mp4") do (
    echo Processing %%~nxf ...
    ffmpeg -i "%%f" -c:v libx264 -preset fast -crf 23 -c:a aac -movflags +faststart "%OUTPUT_DIR%\%%~nxf"
)

echo.
echo All videos processed successfully!
pause
