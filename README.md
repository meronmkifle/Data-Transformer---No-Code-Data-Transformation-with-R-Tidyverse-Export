# 📊 Data Transformer

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://data-transformer.streamlit.app)

A **no-code data transformation platform** that converts visual transformations into reproducible R/tidyverse code. Built with Streamlit for instant feedback and collaboration.

![Data Transformer](https://via.placeholder.com/800x400?text=Data+Transformer)

## ✨ Features

### 🎯 Visual Transformations
- **Filter** – Query-based row filtering (`age > 25`, `status == 'active'`)
- **Select Columns** – Keep/drop columns easily
- **Rename** – Bulk rename with inline editing  
- **Mutate** – Create/modify columns with expressions
- **Group & Summarize** – Aggregate data with multiple functions
- **Pivot** – Reshape from long to wide and vice versa
- **Sort** – Ascending/descending by any column
- **Distinct** – Remove duplicates by specific columns

### 👁️ Live Preview & Feedback
- Real-time data preview after each transformation
- Data shape, types, and null counts
- Column info panel showing data types
- Transformation stack with reordering and deletion

### 💻 R/Tidyverse Code Generation
- Auto-generates clean, production-ready R code
- Copy to clipboard or download as `.R` file
- Share reproducible workflows with colleagues
- Perfect for version control (git-friendly)

### 💾 Pipeline Management
- Save entire transformation sequences as JSON
- Load and reuse pipelines  
- Version control transformation logic
- Share pipeline files across teams

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

**Option 1: Automated (Linux/Mac)**
```bash
git clone https://github.com/yourusername/data-transformer.git
cd data-transformer
bash setup.sh
source venv/bin/activate
streamlit run data_transformer_app.py
```

**Option 2: Manual**
```bash
git clone https://github.com/yourusername/data-transformer.git
cd data-transformer
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run data_transformer_app.py
```

**Option 3: Docker**
```bash
docker-compose up
# Open http://localhost:8501
```

The app opens at **http://localhost:8501**

---

## 📖 Usage Guide

### Step 1: Upload Data
Upload a CSV or Excel file using the sidebar file uploader. The app will:
- Display shape (rows × columns)
- Show column data types
- Flag missing values
- Confirm successful load

**Supported formats**: CSV, XLSX, XLS

### Step 2: Build Transformations
Select transformation type and configure:

**Filter Example:**
```
Condition: year == 2024 & amount > 500
```

**Mutate Example:**
```
New Column: profit
Expression: amount * 0.15
```

**Group & Summarize Example:**
```
Group by: region, salesperson
Summarize:
  - Column: amount, Function: sum
  - Column: units, Function: mean
```

### Step 3: Manage Transformations
- View the transformation stack with descriptions
- Reorder steps using ↑ button
- Remove steps with ❌ button
- See final schema before export

### Step 4: Export & Share
Download R code, copy to clipboard, or save the transformation pipeline as JSON for version control.

### Step 5: Use in R
Paste generated code into RStudio:
```r
library(tidyverse)
library(readr)

data <- read_csv("your_data.csv") %>%
  filter(year == 2024) %>%
  select(date, salesperson, amount, region) %>%
  group_by(region, salesperson) %>%
  summarize(
    total_amount = sum(amount),
    avg_units = mean(units),
    .groups = 'drop'
  ) %>%
  arrange(desc(total_amount))

head(data)
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [README.md](README.md) | This file - quick start guide |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical design and implementation details |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Common issues and solutions |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute to the project |

---

## 🔄 Transformation Operations Reference

### Filter
Remove rows based on conditions.
```
Syntax: column > value, column == 'text', column >= 10 & other < 5
```

### Select Columns
Keep only specific columns.
```
Select: date, amount, region
Result: Dataset with only these 3 columns
```

### Rename
Change column names.
```
Old Name: amount_usd
New Name: revenue
```

### Mutate
Create new columns or modify existing ones.
```
Column: profit
Expression: amount * 0.15
```

### Group & Summarize
Aggregate data by grouping columns.
```
Group By: region, salesperson
Sum: amount
Mean: discount
```

### Pivot Longer
Convert wide data to long format.
```
Columns: Q1, Q2, Q3, Q4
Names To: quarter
Values To: sales
```

### Pivot Wider
Convert long data to wide format.
```
Names From: month
Values From: revenue
```

### Sort
Order rows by a column.
```
Column: amount
Direction: Descending
```

### Distinct
Remove duplicate rows.
```
By Columns: customer_id (or empty for all duplicates)
Keep All: Yes (keep all columns)
```

---

## 💡 Example Workflows

### Sales Analysis Pipeline
1. Filter: `year == 2024`
2. Select: `date, salesperson, region, amount`
3. Mutate: `profit = amount * 0.15`
4. Group By: `region, salesperson`
5. Summarize: `sum(amount)`, `mean(profit)`
6. Sort: Descending by `sum(amount)`

**Generated R Code:**
```r
library(tidyverse)

data <- read_csv("sales_data.csv") %>%
  filter(year == 2024) %>%
  select(`date`, `salesperson`, `region`, `amount`) %>%
  mutate(profit = amount * 0.15) %>%
  group_by(`region`, `salesperson`) %>%
  summarize(
    `sum_amount` = sum(`amount`),
    `mean_profit` = mean(`profit`),
    .groups = 'drop'
  ) %>%
  arrange(desc(`sum_amount`))
```

---

## 🏗️ Architecture

```
Streamlit UI
    ↓
Session State (original_data, data, transformations)
    ↓
Data Processing Layer (Pandas)
    ↓
Code Generation (R/Tidyverse)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed technical documentation.

---

## 🐛 Troubleshooting

**Can't install dependencies?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Port 8501 already in use?**
```bash
streamlit run data_transformer_app.py --server.port=8502
```

**Need help?**
Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) or open a GitHub Issue.

---

## 📊 Supported Data Types

| Type | Filter | Select | Mutate | Summarize |
|------|--------|--------|--------|-----------|
| Integer | ✅ | ✅ | ✅ | ✅ |
| Float | ✅ | ✅ | ✅ | ✅ |
| String | ✅ | ✅ | ✅ | ⚠️ |
| DateTime | ✅ | ✅ | ✅ | ⚠️ |
| Boolean | ✅ | ✅ | ✅ | ⚠️ |

---

## 🗺️ Roadmap

### v1.1 (Planned)
- [ ] Join operations (inner, left, right, full)
- [ ] String manipulation (regex, substring, case)
- [ ] Date/time operations (parsing, formatting)

### v1.2 (Planned)
- [ ] Window functions (lag, lead, rank)
- [ ] Data quality checks (missing values, outliers)
- [ ] Python code generation alongside R

### v2.0 (Vision)
- [ ] Database connectivity (SQL Server, PostgreSQL)
- [ ] Undo/redo with full history
- [ ] Cloud pipeline sharing
- [ ] Collaborative editing
- [ ] Custom transformation functions

See [CONTRIBUTING.md](CONTRIBUTING.md) to help build these features!

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Ways to contribute:**
- 🐛 Report bugs
- 💡 Suggest features
- 📝 Improve documentation
- 🔧 Submit pull requests
- 🧪 Add tests

---

## 👥 Authors

- Created for data analysts, researchers, and data scientists who want reproducible, shareable, no-code data workflows.

---

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) - Amazing framework for data apps
- [Pandas](https://pandas.pydata.org/) - Powerful data manipulation
- [R Tidyverse](https://www.tidyverse.org/) - Inspiring data transformation approach

---

## 📞 Support

- **Documentation**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/data-transformer/issues)
- **Questions**: [GitHub Discussions](https://github.com/yourusername/data-transformer/discussions)

---

**Built with ❤️ for data lovers everywhere.**
