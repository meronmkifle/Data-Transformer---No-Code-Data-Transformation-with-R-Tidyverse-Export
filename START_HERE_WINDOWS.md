# 🚀 Your Windows Setup - Step by Step

Hi Meron! Follow these exact steps to get Data Transformer running on your Windows PC.

---

## Prerequisites Check

Before starting, verify you have:

1. **Python 3.9+** installed
   ```bash
   python --version
   ```
   
   If not installed:
   - Go to https://www.python.org/downloads/
   - Download Python 3.11
   - Run installer
   - ⚠️ **IMPORTANT**: Check "Add Python to PATH"
   - Restart your computer

2. **Git** installed (recommended)
   ```bash
   git --version
   ```
   
   If not installed:
   - Go to https://git-scm.com/
   - Download Windows installer
   - Run installer, accept defaults

---

## Step 1: Open Command Prompt

Press **Win+R**, type `cmd`, click OK

You should see:
```
C:\Users\kiflem>
```

---

## Step 2: Navigate to Your Folder

```bash
cd C:\Users\kiflem\Documents\data-transformer
```

Press Enter.

You should see:
```
C:\Users\kiflem\Documents\data-transformer>
```

(Note: The folder is empty right now)

---

## Step 3: Clone the Repository

```bash
git clone https://github.com/meronmkifle/Data-Transformer---No-Code-Data-Transformation-with-R-Tidyverse-Export.git .
```

Important: The `.` at the end clones into the current empty folder.

You should see output like:
```
Cloning into '.'...
remote: Enumerating objects... 100% (45/45), done.
remote: Counting objects... 100% (45/45), done.
...
```

This takes 10-30 seconds. Wait for it to finish.

When done, you'll see:
```
C:\Users\kiflem\Documents\data-transformer>
```

---

## Step 4: Verify Files Were Downloaded

List the files:
```bash
dir
```

You should see files like:
- `data_transformer_app.py`
- `requirements.txt`
- `sample_sales_data.csv`
- `README.md`
- `setup_windows.bat`
- Folders like `.github`, `.streamlit`, `venv` (if exists)

If you don't see these files, something went wrong with the clone. Contact me.

---

## Step 5: Run the Setup Script

```bash
setup_windows.bat
```

**What this does:**
- Creates a virtual environment (isolated Python)
- Installs streamlit, pandas, openpyxl
- Gets everything ready to run

**You'll see output like:**
```
============================================
  Data Transformer - Windows Setup
============================================

✓ Python found: Python 3.11.7
Creating virtual environment...
✓ Virtual environment created
Activating virtual environment...
✓ Virtual environment activated
Upgrading pip...
✓ Pip upgraded
Installing dependencies...
  - streamlit
  - pandas
  - openpyxl
...
✓ Setup Complete!
```

When it's done, the Command Prompt will stay open. **Press any key to close it.**

---

## Step 6: Start the App

Open **Command Prompt** again (Win+R, type `cmd`, click OK)

Navigate to your folder:
```bash
cd C:\Users\kiflem\Documents\data-transformer
```

Activate the virtual environment:
```bash
venv\Scripts\activate.bat
```

You should see `(venv)` at the start of the line:
```
(venv) C:\Users\kiflem\Documents\data-transformer>
```

Start the app:
```bash
streamlit run data_transformer_app.py
```

You should see:
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

---

## Step 7: Open in Browser

Go to: **http://localhost:8501**

You should see the Data Transformer app with:
- 📁 Data Source (sidebar)
- 🔧 Transformations (left panel)
- 👁️ Data Preview (right panel)
- 💻 R/Tidyverse Code Export (bottom)

---

## Step 8: Test with Sample Data

1. Click **"Upload CSV or Excel"** in the sidebar
2. Navigate to: `C:\Users\kiflem\Documents\data-transformer\sample_sales_data.csv`
3. Select it and open
4. Wait for **"✅ Loaded: 37 rows × 8 cols"** message

---

## Step 9: Try Your First Transformation

1. In left panel, select **"Filter"** from dropdown
2. Enter condition: `year == 2024`
3. Click **"Add Filter"**
4. Watch the right panel update to show only 2024 data

---

## Step 10: Export R Code

1. Scroll down to **"💻 R/Tidyverse Code Export"**
2. Click **"📥 Download .R"**
3. File saves to your Downloads folder
4. Open in notepad to see the R code

---

## Future Sessions: Quick Start

Every time you want to use the app:

1. Open Command Prompt (Win+R, type `cmd`)
2. Navigate to folder:
   ```bash
   cd C:\Users\kiflem\Documents\data-transformer
   ```
3. Activate virtual environment:
   ```bash
   venv\Scripts\activate.bat
   ```
4. Start the app:
   ```bash
   streamlit run data_transformer_app.py
   ```
5. Open browser: http://localhost:8501

---

## Stopping the App

Press **Ctrl+C** in Command Prompt.

To exit Command Prompt: Type `exit` and press Enter.

---

## Important Files

| File | What It Does |
|------|------|
| `data_transformer_app.py` | The app itself |
| `requirements.txt` | List of packages to install |
| `sample_sales_data.csv` | Example dataset for testing |
| `setup_windows.bat` | Setup script (only needed once) |
| `README.md` | Full documentation |
| `QUICK_REFERENCE.md` | Quick lookup while using |
| `WINDOWS_SETUP.md` | Detailed Windows guide |
| `TROUBLESHOOTING.md` | Fix problems |

---

## Common Issues

### "Python is not recognized"
**Fix**: Reinstall Python and CHECK "Add Python to PATH"

### "venv\Scripts\activate.bat is not recognized"
**Fix**: Make sure you're in the right folder:
```bash
cd C:\Users\kiflem\Documents\data-transformer
```

### "ModuleNotFoundError: No module named 'streamlit'"
**Fix**: Make sure venv is activated (see `(venv)` at start of line)

### "Address already in use: port 8501"
**Fix**: Use different port:
```bash
streamlit run data_transformer_app.py --server.port=8502
```
Then go to: http://localhost:8502

### App won't load data
**Fix**: 
1. Use sample_sales_data.csv first to test
2. Make sure file is CSV or Excel (.xlsx)
3. First row must be column headers

---

## Folder Structure After Setup

```
C:\Users\kiflem\Documents\data-transformer\
├── venv\                              ← Virtual environment (created by setup)
├── .github\
├── .streamlit\
├── data_transformer_app.py            ← The app
├── requirements.txt
├── sample_sales_data.csv              ← Test data
├── setup_windows.bat                  ← Setup script
├── README.md                          ← Documentation
├── QUICK_REFERENCE.md
├── WINDOWS_SETUP.md                   ← Windows guide
├── TROUBLESHOOTING.md
└── ... other files
```

---

## Pro Tips

1. **Create shortcut to folder**
   - Right-click `C:\Users\kiflem\Documents\data-transformer`
   - Create shortcut on Desktop
   - Pin to Quick Access

2. **Use VS Code for editing**
   - Open folder in VS Code
   - Built-in terminal saves you typing cd

3. **Keep venv folder**
   - Don't delete it
   - Reuse for all future runs

4. **Upgrade later if needed**
   ```bash
   pip install --upgrade streamlit pandas
   ```

---

## Next: Learn How to Use

Once the app is running:

1. Read **README.md** - Features and examples
2. Try **QUICK_REFERENCE.md** - Commands and syntax
3. Upload your own data and test

---

## Video Summary (What You're Doing)

1. ✅ Clone repo from GitHub
2. ✅ Run setup_windows.bat (installs packages)
3. ✅ Activate virtual environment
4. ✅ Run `streamlit run data_transformer_app.py`
5. ✅ Browser opens to http://localhost:8501
6. ✅ Upload CSV/Excel
7. ✅ Click through transformations
8. ✅ Download R code
9. ✅ Use in RStudio

---

## Questions?

1. **Check WINDOWS_SETUP.md** - More detailed version of this guide
2. **Check TROUBLESHOOTING.md** - Common problems & solutions
3. **Check QUICK_REFERENCE.md** - Command reference
4. **Read README.md** - Full feature documentation

---

## You're Ready! 🚀

Start from **Step 1** and let me know if you get stuck on any step!

**Let's go!**
