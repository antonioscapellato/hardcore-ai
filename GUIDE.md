# Project Setup Guide

## 1. Setting up the Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Installing Dependencies
After activating the virtual environment, install the required packages:
```bash
pip install -r requirements.txt
```

## 3. Running the Project
To run the project, make sure you're in the virtual environment (you should see `(venv)` in your terminal prompt), then:

```bash
# For Python scripts
python your_script.py

# For Jupyter notebooks
jupyter notebook
```

## 4. Development Workflow
1. Always activate the virtual environment before starting work:
   ```bash
   source venv/bin/activate
   ```

2. When installing new packages, update requirements.txt:
   ```bash
   pip freeze > requirements.txt
   ```

3. To deactivate the virtual environment when done:
   ```bash
   deactivate
   ```

## 5. Common Issues
- If you see "command not found" errors, make sure the virtual environment is activated
- If package imports fail, verify that you've installed all requirements
- If you get permission errors, ensure you're using the virtual environment and not the system Python

## 6. Best Practices
- Keep your virtual environment activated while working
- Regularly update requirements.txt when adding new dependencies
- Use version control (git) to track changes
- Create a .gitignore file to exclude venv/ and other unnecessary files