@echo off
setlocal

:: Base folder
set BASE=pneumonia-detection-ai

:: Create directories
mkdir %BASE%\data\raw
mkdir %BASE%\data\processed
mkdir %BASE%\.github\workflows
mkdir %BASE%\configs
mkdir %BASE%\deployment
mkdir %BASE%\docs
mkdir %BASE%\models
mkdir %BASE%\notebooks
mkdir %BASE%\output
mkdir %BASE%\scripts
mkdir %BASE%\src
mkdir %BASE%\tests

:: Create files
type nul > %BASE%\.gitignore
type nul > %BASE%\CODE_OF_CONDUCT.md
type nul > %BASE%\CONTRIBUTING.md
type nul > %BASE%\LICENSE
type nul > %BASE%\README.md
type nul > %BASE%\requirements.txt

type nul > %BASE%\.github\workflows\ci.yml
type nul > %BASE%\configs\config.yaml
type nul > %BASE%\data\raw\.gitkeep
type nul > %BASE%\data\processed\.gitkeep
type nul > %BASE%\deployment\app.py
type nul > %BASE%\deployment\Dockerfile
type nul > %BASE%\deployment\.dockerignore
type nul > %BASE%\docs\architecture.md
type nul > %BASE%\docs\data_pipeline.md
type nul > %BASE%\models\.gitkeep
type nul > %BASE%\notebooks\01_data_exploration.ipynb
type nul > %BASE%\notebooks\02_model_prototyping.ipynb
type nul > %BASE%\notebooks\03_results_visualization.ipynb
type nul > %BASE%\output\.gitkeep
type nul > %BASE%\scripts\download_data.sh
type nul > %BASE%\scripts\run_training.sh
type nul > %BASE%\src\__init__.py
type nul > %BASE%\src\data_loader.py
type nul > %BASE%\src\model.py
type nul > %BASE%\src\predict.py
type nul > %BASE%\src\train.py
type nul > %BASE%\src\utils.py
type nul > %BASE%\tests\__init__.py
type nul > %BASE%\tests\test_data_loader.py
type nul > %BASE%\tests\test_model.py

echo ✅ Project structure created successfully!
endlocal
pause
