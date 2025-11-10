@echo off
:: ==============================================================
:: Project: Dog's Breed Identification
:: Author: Mobin Yousefi
:: GitHub: https://github.com/mobinyousefi-cs
:: Description: Automates environment setup and model training.
:: ==============================================================


REM Activate virtual environment
if exist .venv\Scripts\activate (
call .venv\Scripts\activate
) else (
echo Virtual environment not found. Creating one...
python -m venv .venv
call .venv\Scripts\activate
)


REM Install dependencies
pip install -e .[dev]


REM Train model
python -m dogs_breed_identification.train --config configs/training_config.yaml


echo Training complete! Press any key to exit.
pause >nul