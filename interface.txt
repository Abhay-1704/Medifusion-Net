@echo off

REM Set full path to conda
CALL "D:\Python\Scripts\activate.bat" base
CALL conda activate lungs-ai

REM Change to your project directory
cd /d "C:\Users\Abhay\Desktop\Lungs disease AI and ML\Custom Model"

REM Run Streamlit (also with full path if needed)
CALL streamlit run Interface.py

pause