# 🪟 Data Transformer - Windows Setup Guide

## Quick Start (2 Steps)

### Step 1: Clone Repository
Open **Command Prompt** or **PowerShell** and navigate to your folder:
```bash
cd C:\Users\kiflem\Documents\data-transformer
```

Then clone the repository:
```bash
git clone https://github.com/meronmkifle/Data-Transformer---No-Code-Data-Transformation-with-R-Tidyverse-Export.git .
```

The `.` at the end clones into the current (empty) folder.

### Step 2: Run Setup
Double-click `setup_windows.bat` OR run from command line:
```bash
setup_windows.bat
```

That's it! The app will be ready to run.

---

## Detailed Instructions

### Prerequisites
- Windows 10 or 11
- Python 3.9+ ([Download here](https://www.python.org/downloads/))
- Git ([Download here](https://git-scm.com/)) - Optional but recommended
- 2GB RAM minimum
- ~500MB disk space

### Installing Python on Windows

1. **Download Python**
   - Go to https://www.python.org/downloads/
   - Click "Download Python 3.11" (or latest 3.x)
   - Choose the "Windows installer"

2. **Install Python**
   - Run the installer
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH"
   - Click "Install Now"
   - Wait for completion

3. **Verify Installation**
   - Open Command Prompt (Win+R, type `cmd`)
   - Type: `python --version`
   - Should see: `Python 3.x.x`

### Installing Git on Windows (Optional)

1. Go to https://git-scm.com/
2. Click "Download for Windows"
3. Run installer, accept defaults
4. Restart your computer
5. Verify: Open Command Prompt, type `git --version`

---

## Step-by-Step Setup

### 1. Open Command Prompt
- Press **Win+R**
- Type `cmd`
- Click OK

### 2. Navigate to Folder
```bash
cd C:\Users\kiflem\Documents\data-transformer
```

(Replace `kiflem` with your username if different)

### 3. Clone Repository
```bash
git clone https://github.com/meronmkifle/Data-Transformer---No-Code-Data-Transformation-with-R-Tidyverse-Export.git .
```

You should see files downloading. When done:
```
Cloning into '.'...
remote: Enumerating objects...
...
```

### 4. Run Setup Script
```bash
setup_windows.bat
```

Or double-click `setup_windows.bat` in File Explorer.

**This script will:**
- ✓ Check Python installation
- ✓ Create virtual environment
- ✓ Activate virtual environment
- ✓ Upgrade pip
- ✓ Install dependencies (streamlit, pandas, openpyxl)

### 5. Activate Virtual Environment (Future Sessions)
Once setup is done, every time you open Command Prompt:
```bash
cd C:\Users\kiflem\Documents\data-transformer
venv\Scripts\activate.bat
```

(You should see `(venv)` at the start of the prompt)

### 6. Start the App
```bash
streamlit run data_transformer_app.py
```

**You should see:**
```
Collecting usage statistics...
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
```

### 7. Open in Browser
Click the link or open: http://localhost:8501

---

## Common Windows Issues & Fixes

### "Python is not recognized"
**Problem**: `'python' is not recognized as an internal or external command`

**Solution**:
1. Reinstall Python
2. ⚠️ **CHECK "Add Python to PATH"** during installation
3. Restart Command Prompt
4. Try again

### "venv\Scripts\activate.bat is not recognized"
**Problem**: Command not found when trying to activate

**Solution**:
1. Make sure you're in the correct folder:
   ```bash
   cd C:\Users\kiflem\Documents\data-transformer
   ```
2. Check venv folder exists:
   ```bash
   dir venv
   ```
3. If folder doesn't exist, virtual environment wasn't created. Run:
   ```bash
   python -m venv venv
   ```

### "pip: command not found"
**Problem**: Pip not available

**Solution**:
```bash
python -m pip install --upgrade pip
```

### "Address already in use: port 8501"
**Problem**: Another app using port 8501

**Solution 1**: Find and stop the other process
```bash
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

**Solution 2**: Use different port
```bash
streamlit run data_transformer_app.py --server.port=8502
```

### "ModuleNotFoundError: No module named 'streamlit'"
**Problem**: Dependencies not installed

**Solution**:
1. Activate virtual environment:
   ```bash
   venv\Scripts\activate.bat
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### "Permission denied" or "Access is denied"
**Problem**: Windows security blocking something

**Solution**:
1. Run Command Prompt as Administrator:
   - Right-click Command Prompt
   - Click "Run as administrator"
2. Navigate to folder and try again

---

## Running the App

### First Run
```bash
REM Navigate to folder
cd C:\Users\kiflem\Documents\data-transformer

REM Activate virtual environment
venv\Scripts\activate.bat

REM Start the app
streamlit run data_transformer_app.py
```

### Subsequent Runs
Just need to activate and run:
```bash
cd C:\Users\kiflem\Documents\data-transformer
venv\Scripts\activate.bat
streamlit run data_transformer_app.py
```

### To Stop the App
- Press **Ctrl+C** in Command Prompt
- Or close the browser tab (won't fully stop)

### To Deactivate Virtual Environment
```bash
deactivate
```

---

## Using the App

### 1. Upload Data
- Sidebar → "📁 Data Source"
- Click "Upload CSV or Excel"
- Select file
- Wait for "✅ Loaded" message

### 2. Try a Transformation
- Select "Filter" from dropdown
- Enter: `year == 2024`
- Click "Add Filter"
- See result in right panel

### 3. Export R Code
- Scroll to "💻 R/Tidyverse Code Export"
- Click "📥 Download .R"
- Code saved to Downloads folder

### 4. Save Pipeline
- Scroll to "💾 Save/Load Pipelines"
- Click "💾 Save Pipeline"
- Download JSON file

---

## Testing with Sample Data

Sample data is included: `sample_sales_data.csv`

**Test workflow:**
1. Start the app
2. Upload `sample_sales_data.csv`
3. Filter: `year == 2024`
4. Select: `date, salesperson, amount, region`
5. Mutate: `profit = amount * 0.15`
6. Group by: `region, salesperson`
7. Summarize: `sum(amount)`, `mean(profit)`
8. Download R code

---

## Git Workflow (If You Have Git)

### First Time Setup
```bash
cd C:\Users\kiflem\Documents\data-transformer
git config user.name "Your Name"
git config user.email "your.email@example.com"
```

### After Making Changes
```bash
git add .
git commit -m "Add: description of changes"
git push origin main
```

### Update from GitHub
```bash
git pull origin main
```

---

## Virtual Environment Explained

### What is it?
A virtual environment isolates Python packages for this project. You can have different versions for different projects without conflicts.

### What's in the venv folder?
- Python interpreter
- All installed packages
- Activation scripts

### When to use
- **Always** when running the app
- It's automatic once activated

### How to delete
If you mess up and want to start over:
```bash
rmdir /s venv
```
Then run `setup_windows.bat` again.

---

## File Structure After Setup

```
C:\Users\kiflem\Documents\data-transformer\
├── venv\                              (Virtual environment - created by setup)
│   ├── Scripts\
│   │   └── activate.bat               (Activate venv)
│   └── Lib\
│       └── site-packages\             (Installed packages)
├── .github\
│   └── workflows\
│       └── ci.yml
├── .streamlit\
│   └── config.toml
├── data_transformer_app.py            (Main app)
├── requirements.txt
├── sample_sales_data.csv
├── README.md
├── QUICK_REFERENCE.md
├── ARCHITECTURE.md
├── TROUBLESHOOTING.md
├── setup_windows.bat                  (Run this first time)
├── setup.sh
├── Dockerfile
└── docker-compose.yml
```

---

## Useful Commands

| Task | Command |
|------|---------|
| Activate venv | `venv\Scripts\activate.bat` |
| Deactivate venv | `deactivate` |
| List packages | `pip list` |
| Install package | `pip install package_name` |
| Stop app | `Ctrl+C` |
| Check Python version | `python --version` |
| Check pip version | `pip --version` |
| Navigate folder | `cd C:\path\to\folder` |
| List files | `dir` |
| Clear screen | `cls` |

---

## Updating the Project

If you make changes on GitHub, update locally:
```bash
cd C:\Users\kiflem\Documents\data-transformer
git pull origin main
```

If dependencies were updated:
```bash
pip install -r requirements.txt
```

---

## Next Steps

1. ✅ Clone repository
2. ✅ Run setup_windows.bat
3. ✅ Start app: `streamlit run data_transformer_app.py`
4. ✅ Upload sample_sales_data.csv
5. ✅ Try a transformation
6. ✅ Export R code
7. ✅ Read README.md for more features

---

## Getting Help

### If something breaks:
1. **Stop the app**: Ctrl+C
2. **Check TROUBLESHOOTING.md** in the project folder
3. **Look for your error** in the list
4. **Follow the solution**
5. **Try again**

### Common issues:
- Python not installed → Install Python with PATH
- Virtual environment not working → Run setup_windows.bat again
- Port in use → Use `--server.port=8502`
- Dependencies missing → `pip install -r requirements.txt`

### Still stuck?
- Open GitHub Issue with error message
- Include Python version: `python --version`
- Include error from Command Prompt

---

## Pro Tips

1. **Pin to Quick Access**: Right-click folder → Pin to Quick Access
2. **Create shortcut**: Right-click setup_windows.bat → Create shortcut
3. **Edit in VS Code**: Open folder in VS Code for better editing
4. **Use terminal in VS Code**: Built-in terminal handles activation
5. **Keep venv folder**: Don't delete, use for all future runs

---

**You're all set! Run setup_windows.bat and enjoy! 🚀**

For detailed usage: See **README.md**  
For quick help: See **QUICK_REFERENCE.md**  
For problems: See **TROUBLESHOOTING.md**
