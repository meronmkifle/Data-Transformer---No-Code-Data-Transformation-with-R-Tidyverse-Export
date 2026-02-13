# Troubleshooting & FAQ

## Installation Issues

### `ModuleNotFoundError: No module named 'streamlit'`
**Solution**: Install dependencies
```bash
pip install -r requirements.txt
```

### `pip: command not found`
**Solution**: Use `pip3` instead or check Python installation
```bash
pip3 install -r requirements.txt
python3 -c "import pip; print(pip.__version__)"
```

### `Permission denied` when running `setup.sh`
**Solution**: Make it executable
```bash
chmod +x setup.sh
bash setup.sh
```

---

## Runtime Issues

### App crashes immediately after loading
**Check**: Python version compatibility
```bash
python --version  # Should be 3.9+
```

### `Streamlit is using cached data. Clear cache to reload.`
**Solution**: Clear Streamlit cache
```bash
# Delete cache directory
rm -rf ~/.streamlit/cache/
# Or restart the app with --logger.level=debug
streamlit run data_transformer_app.py --logger.level=debug
```

### Data file uploaded but transformations don't work
**Solution**: Check file format
- Ensure CSV has headers in first row
- Excel files must have sheet data in first sheet
- No empty columns or rows at start

---

## Data Transformation Issues

### Filter returns 0 rows
**Common causes**:
- Case sensitivity: `Status == 'Active'` ≠ `Status == 'active'`
- Column name incorrect: Use exact column name from header
- Data type mismatch: `age > "25"` (string) vs `age > 25` (integer)

**Example fixes**:
```
❌ age = 25        →  ✅ age == 25
❌ Status='active' →  ✅ Status == 'active'
❌ year=2024       →  ✅ year == 2024
```

### Mutate expression gives error
**Common causes**:
- Column name with spaces: Use backticks
  ```
  ✅ `Amount USD` * 1.1
  ```
- Undefined column: Double-check spelling
- Invalid operation: Can't multiply string by number

**Example expressions**:
```python
# Simple math
profit = amount * 0.15

# Column combination
total = revenue - cost

# Conditional (when supported)
status = 'high' if amount > 1000 else 'low'

# Date parsing
import pandas as pd; pd.to_datetime(date_column)
```

### Group & Summarize gives strange column names
**Fix**: Rename columns after operation
- Use Rename transformation
- Or download R code and clean names there

---

## Code Export Issues

### Generated R code has backticks everywhere
**This is normal** - backticks protect column names with special characters
```r
# Backticks are safe, they're just defensive
select(`column with spaces`, `column-with-dashes`)
```

### R code won't run in RStudio
**Ensure you have**:
```r
# These libraries installed
install.packages(c("tidyverse", "readr"))

# Then load
library(tidyverse)
library(readr)

# Replace 'your_data.csv' with actual file path
data <- read_csv("your_data.csv") %>% ...
```

### Copy to clipboard doesn't work
**Workaround**: 
- Click inside the code text box and select all (`Ctrl+A`)
- Copy manually (`Ctrl+C`)
- Or use "Download .R" button instead

---

## Performance Issues

### App is slow with large file (>100K rows)
**Solutions**:
1. Filter data first before other operations
2. Avoid grouping by many columns
3. Consider working with sample (first 10K rows) during development
4. Download R code and run in RStudio for speed

**Example**: Load full data in R
```r
# In RStudio - much faster than Streamlit
data <- read_csv("large_file.csv") %>%
  filter(year == 2024) %>%
  # ... rest of transformations
```

### Browser gets unresponsive
**Solutions**:
- Reduce number of rows displayed (app shows first 20 by default)
- Don't preview very wide datasets (100+ columns)
- Restart Streamlit app: Stop and run `streamlit run data_transformer_app.py` again

---

## Docker Issues

### Docker build fails
**Check Docker installation**:
```bash
docker --version
docker-compose --version
```

### Port 8501 already in use
**Solution 1**: Stop other Streamlit apps
```bash
lsof -i :8501  # Find process
kill -9 <PID>  # Kill it
```

**Solution 2**: Use different port
```bash
docker-compose down
# Edit docker-compose.yml: change "8501:8501" to "8502:8501"
docker-compose up
# Access at http://localhost:8502
```

---

## File Handling

### Large Excel file won't load
**Issue**: `openpyxl` can be slow with large files
**Solution**: 
1. Convert Excel to CSV
2. Or increase Streamlit timeout:
```bash
streamlit run data_transformer_app.py --client.dataframeSerializationVersion=latest
```

### Special characters in filenames cause issues
**Solution**: Use simple ASCII filenames
```
✅ sales_2024.csv
❌ ventes_2024_données.csv  # Non-ASCII
```

---

## Pipeline Management

### Can't load saved pipeline JSON
**Check**:
- File was downloaded from this app (not manually created)
- JSON is valid (not corrupted)
- File extension is `.json`

**Debug**: Try opening JSON in text editor to verify format

### Pipeline works in one browser but not another
**Cause**: Browser may have cached old app version
**Solution**: 
- Hard refresh: `Ctrl+Shift+R` (Windows/Linux) or `Cmd+Shift+R` (Mac)
- Clear browser cache
- Open in incognito/private window

---

## General Tips

### Best Practices
- **Start simple**: Build transformations one at a time
- **Save often**: Save pipelines after each major milestone
- **Test locally**: Verify transformations work as expected
- **Version control**: Commit pipeline JSONs to git

### Debugging Workflow
1. **Isolate the problem**: Remove transformations one by one
2. **Test in R**: Copy generated code to RStudio and run
3. **Check data**: Preview at each step
4. **Validate inputs**: Ensure column names and conditions are correct

### Reporting Bugs
Include:
- Operating system and Python version
- Steps to reproduce
- Sample data (anonymized)
- Screenshot of error
- Generated R code if applicable

---

## Contact & Support

- **Issues**: Open GitHub Issue
- **Questions**: GitHub Discussions
- **Feedback**: GitHub Issues with `feedback` label

---

Still stuck? Open an issue with details and we'll help! 🚀
