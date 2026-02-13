# ⚡ Quick Reference Guide

## Installation (Pick One)

### Bash Auto-Setup (Linux/Mac)
```bash
bash setup.sh && source venv/bin/activate && streamlit run data_transformer_app.py
```

### Manual Setup
```bash
pip install -r requirements.txt
streamlit run data_transformer_app.py
```

### Docker
```bash
docker-compose up
```

---

## Common Tasks

### Upload Data
1. Sidebar → "📁 Data Source"
2. Click "Upload CSV or Excel"
3. Select file
4. Wait for "✅ Loaded" message

### Add Transformation
1. Select transformation type from dropdown
2. Configure options
3. Click "Add [Type]"
4. See changes in right panel preview

### Export R Code
1. Scroll to "💻 R/Tidyverse Code Export"
2. Click "📥 Download .R" to save
3. Or "📋 Copy" to copy code

### Save Pipeline
1. Scroll to "💾 Save/Load Pipelines"
2. Click "💾 Save Pipeline"
3. Enter pipeline name
4. Click "📥 Download Pipeline JSON"

### Load Pipeline
1. Scroll to "💾 Save/Load Pipelines"
2. Click "Load Pipeline"
3. Select JSON file
4. Click "Open"

---

## Transformation Syntax

### Filter
```
age > 25
year == 2024
status == 'active' & amount > 100
```

### Mutate
```
profit = amount * 0.15
total = revenue - cost
ratio = numerator / denominator
```

### Group & Summarize
- Group by: Select columns
- Summarize: Choose function (sum, mean, median, min, max, count)

### Pivot Longer
- Columns to pivot: Select wide columns
- Names To: Variable name (e.g., "quarter")
- Values To: Value name (e.g., "sales")

### Pivot Wider
- Names From: Column with variable names
- Values From: Column with values

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Refresh | F5 / Cmd+R |
| Stop Streamlit | Ctrl+C |
| Clear browser cache | Ctrl+Shift+Delete |
| Hard refresh | Ctrl+Shift+R (Win/Lin) or Cmd+Shift+R (Mac) |

---

## Troubleshooting Quick Fixes

### "Module not found" Error
```bash
pip install -r requirements.txt
```

### Port 8501 in use
```bash
streamlit run data_transformer_app.py --server.port=8502
```

### Clear Streamlit cache
```bash
rm -rf ~/.streamlit/cache/
```

### Docker won't start
```bash
docker-compose down
docker system prune
docker-compose up
```

---

## Expression Examples

### Math
```python
profit = amount * 0.15
discount_price = price * (1 - discount_rate)
total = sum_column + tax_column
```

### String (requires column names without spaces)
```python
name_upper = name.str.upper()
initials = name.str[:1]
```

### Date (pandas)
```python
import pandas as pd
date_parsed = pd.to_datetime(date_column)
```

### Conditional (when supported)
```python
status = 'high' if amount > 1000 else 'low'
```

---

## File Format Support

### ✅ Supported
- CSV (comma, semicolon, tab-delimited)
- XLSX (Excel 2010+)
- XLS (Excel 2003-2007)

### ❌ Not Supported
- JSON
- Parquet
- HDF5
- Databases

### 📝 Tips
- Use ASCII filenames (no special characters)
- First row must be column headers
- No blank rows at the beginning

---

## Performance Tips

### For Large Files (>100K rows)
1. Filter data first
2. Select only needed columns
3. Avoid grouping by many columns
4. Consider loading in R directly:
   ```r
   # In RStudio - much faster
   data <- read_csv("large_file.csv") %>%
     # your transformations
   ```

### General
- Use CSV instead of Excel
- Preview first 20 rows only
- Save pipelines, not raw R code for version control

---

## Git Workflow

### Commit Pipeline
```bash
git add my_pipeline.json
git commit -m "Add sales analysis pipeline"
git push origin main
```

### Share Transformation
1. Save pipeline as JSON
2. Commit to Git
3. Share JSON file with team
4. Team members can load and modify

---

## R Code Quick Check

**Before running in RStudio:**
```r
# Install once
install.packages("tidyverse")
install.packages("readr")

# Load
library(tidyverse)

# Run your generated code
data <- read_csv("your_file.csv") %>%
  # ... transformations ...

# View
head(data)
```

---

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `KeyError: column name` | Column doesn't exist | Check exact column name (case-sensitive) |
| `TypeError in eval` | Invalid expression | Check syntax, use backticks for spaces: \`column name\` |
| `ValueError in filter` | Wrong condition syntax | Use `==` not `=`, use `&` not `and` |
| `Empty dataframe result` | Filter too strict | Preview data first, check condition |
| `Port in use` | Port 8501 occupied | Use `--server.port=8502` |
| `File too large` | >500MB file | Convert to CSV, use sample |

---

## Best Practices

1. **Start Simple** - One transformation at a time
2. **Preview Often** - Check results after each step
3. **Save Pipelines** - Save after major milestones
4. **Document Steps** - Use descriptive transformation names
5. **Test in R** - Verify generated code works
6. **Version Control** - Commit pipelines to Git
7. **Share Safely** - Use anonymized sample data

---

## Useful Links

| Resource | URL |
|----------|-----|
| Streamlit Docs | https://docs.streamlit.io |
| Pandas Docs | https://pandas.pydata.org/docs |
| R Tidyverse | https://www.tidyverse.org |
| This Project | https://github.com/yourusername/data-transformer |

---

## Getting Help

1. **Read TROUBLESHOOTING.md** - 90% of issues covered
2. **Check ARCHITECTURE.md** - Understand how it works
3. **Review sample_sales_data.csv** - Test with example
4. **Open GitHub Issue** - For bugs and features
5. **Check GitHub Discussions** - For questions

---

**Save time. Keep it simple. Transform with confidence.** ⚡
