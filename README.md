# Data Transformer Studio Pro

A comprehensive, interactive data transformation and visualization tool built with Streamlit, covering the complete **R for Data Science (2e)** workflow. Transform raw data into publication-ready insights with no code required, while generating production-grade R/tidyverse code at every step.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)
![Plotly](https://img.shields.io/badge/plotly-6.x-purple)

## Features

### Transform (dplyr verbs — Ch. 3)
- **filter()** - Keep rows matching conditions (categorical multi-select or numeric range/comparison, with negate)
- **select()** - Choose and organize columns (keep or drop mode)
- **rename()** - Single column rename
- **relocate()** - Move column to first, last, or after a specified column
- **arrange()** - Sort by multiple columns ascending/descending
- **distinct()** - Deduplicate with row count feedback
- **slice()** - Head, tail, sample, min, max variants
- **mutate()** - Create new columns with:
  - Arithmetic operations (col ± col, col ± const)
  - Statistical transformations (log, sqrt, abs, exp, scale 0-1, z-score)
  - `lag()` / `lead()` — time series shifts
  - `rank()` — 5 ranking variants
  - Custom Python/R expression
- **case_when()** - Multi-condition recoding into a new column
- **if_else()** - Binary conditional column creation
- **across()** - Apply function across multiple columns (8 functions: scale, z-score, log, sqrt, round, replace_na, abs, cumsum)
- **group_by() & summarize()** - Aggregate with up to 5 simultaneous summary columns (mean, sum, median, min, max, sd, n, n_distinct)
- **count()** - Frequency table with sort
- **add_count()** - Append count without collapsing rows

### Tidy & Join (Ch. 5 & 19)
- **pivot_longer()** - Wide → long with configurable names_to / values_to
- **pivot_wider()** - Long → wide with automatic multi-index handling
- **separate()** - Split one column into many with separator preview
- **unite()** - Merge columns with custom separator
- **6 join types** — left, right, inner, full, semi, anti (upload second dataset in sidebar)

### Strings, Dates & Factors (Ch. 14–17)

**14 string operations (stringr):**
- Case: `str_to_upper`, `str_to_lower`, `str_to_title`
- Whitespace: `str_trim` (left/right/both), `str_squish`
- Replace/Remove: `str_replace_all`, `str_remove`
- Detect: `str_detect` with filter and negate option
- Extract: `str_extract` with regex preview
- Length/position: `str_length`, `str_sub`
- Pad: `str_pad` with width and side
- Match: `str_starts`, `str_ends`

**Date/time (lubridate):**
- Parse: ymd, mdy, dmy, ymd_hms, auto-detect
- Extract 8 components: year, month, day, wday, hour, minute, quarter, week
- Arithmetic: days since today, difference between two date columns, add/subtract days

**Factor operations (forcats):**
`fct_infreq`, `fct_rev`, `fct_reorder` (by numeric median), `fct_lump_n`, `fct_recode` (dropdown of existing levels), `fct_collapse`, `fct_explicit_na`

### Clean (Ch. 18)
- **Missing Value Handling**:
  - Drop rows with NA (by column)
  - Fill forward (`ffill`) or backward (`bfill`)
  - Replace with custom value
  - Fill with column mean or median (column-targeted)
- **Type Conversion** - numeric, character, integer, factor, logical
- **Duplicate Detection** - by column subset, with preview of duplicate rows
- **Outlier Treatment (IQR)**:
  - Remove outliers
  - Cap / Winsorize to IQR bounds
  - Flag as boolean column
  - Interactive boxplot with Q1/Q3/IQR metrics

### Visualize (Ch. 1 & 9 — ggplot2)

**5 sub-tabs:**

1. **Quick Plot** — Bar, Line, Scatter, Histogram (with box marginal), Box, Violin, Density, Heatmap (count); saves to R pipeline
2. **Distribution** — Histogram + Box marginal, ECDF, Q-Q Plot; mean, median, std, skewness stats
3. **Relationship** — Scatter, Scatter Matrix, Scatter + Regression (OLS or LOWESS fallback), Bubble chart
4. **Facets** — `facet_wrap` and `facet_grid` for Scatter, Bar, Box
5. **Time Series** — Line chart with date aggregation (Day/Week/Month/Quarter/Year) and interactive range selector

### Plot Editor — Publication-Ready
Full control over every aesthetic parameter:
- **Plot types**: Bar, Line, Scatter, Box, Violin, Histogram, Density
- **Labels**: title, subtitle, X/Y axis, caption (all editable)
- **Aesthetics**: font size, height, point/bar size, opacity, base colour, colour palette
- **14 publication-ready themes**:
  - ggplot2 built-in: `theme_minimal`, `theme_bw`, `theme_classic`, `theme_linedraw`, `theme_light`, `theme_void`
  - hrbrthemes: `theme_ipsum`, `theme_ft_rc`
  - ggthemes: `theme_economist`, `theme_wsj`, `theme_fivethirtyeight`, `theme_tufte`, `theme_solarized`
  - ggpubr: `theme_pubr`
- **Advanced options**: log scale X/Y, reference lines, LOWESS trend lines, coord_flip
- Generates complete, copy-paste-ready ggplot2 R code
- "Save to R Pipeline" persists chart code into the pipeline tracker

### R Pipeline
- Every operation tracked as a numbered step with category and description
- 4 output modes: Complete Pipeline, Wrangling Only, Visualization Only, Separate Blocks by Category
- Uses modern native `|>` pipe operator throughout
- Auto-detects required libraries (stringr, lubridate, forcats)
- One-click download as `.R` file
- Clear pipeline option

### Export
- **CSV** — transformed dataset
- **TSV** — tab-separated
- **Excel (.xlsx)** — requires openpyxl
- **JSON** — records-oriented
- **Complete R Script** — full wrangling + viz pipeline
- **Wrangling-only R Script**
- **Visualization-only R Script**
- **ggplot2 Theme Reference** — downloadable `.R` cheat sheet with all 14 themes + colour scale examples
- Column types summary panel

---

## Usage

### Basic Workflow

1. **Upload Data** — CSV, Excel, or TSV; automatic type detection and instant EDA
2. **EDA** — glimpse, summary stats, distributions, correlation heatmap, missing value patterns
3. **Transform** — filter, select, mutate, group, summarize, case_when, across
4. **Tidy/Join** — reshape wide↔long, join a second dataset
5. **Strings/Dates/Factors** — clean text, parse dates, recode factor levels
6. **Clean** — missing values, type conversion, outlier treatment
7. **Visualize** — quick plots, distributions, relationships, facets, time series
8. **Plot Editor** — fine-tune aesthetics, get publication-ready ggplot2 code
9. **R Pipeline** — review all steps, choose output mode, download `.R` script
10. **Export** — CSV / Excel / JSON + R scripts

### Example: Sales Data Analysis

```
1. Upload sales_data.csv (100,000 rows)
2. Filter by year >= 2023
3. Select: date, region, product, amount
4. Mutate: profit = amount * 0.15
5. case_when: profit_band = Low / Mid / High
6. Group by region, sum amount
7. Visualize as Bar Chart
8. Export R code and CSV
```

Generated R code:
```r
# R Analysis Pipeline — Data Transformer Studio Pro
# Generated: 2025-02-16 14:30
# R4DS 2e (Wickham, Cetinkaya-Rundel, Grolemund)

library(tidyverse)

data <- read_csv("sales_data.csv")

data_clean <- data |>
  # -- Transform --
  filter(year >= 2023) |>
  select(date, region, product, amount) |>
  mutate(profit = amount * 0.15) |>
  mutate(profit_band = case_when(
    profit < 100 ~ "Low",
    profit < 500 ~ "Mid",
    .default = "High"
  )) |>
  group_by(region) |>
  summarize(total_amount = sum(amount, na.rm = TRUE), .groups = "drop")

# -- Visualization --
# Bar chart: Total Sales by Region
ggplot(data_clean, aes(x = region, y = total_amount)) +
  geom_col(alpha = 0.8, fill = "#4a6cf7") +
  labs(
    title = "Total Sales by Region",
    x = "Region",
    y = "Total Amount"
  ) +
  theme_minimal()
```

---

## Requirements

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.0.0
openpyxl>=3.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

> **Note:** `trendline="ols"` in the Relationship scatter requires `statsmodels`. If not installed, the app automatically falls back to `lowess`.

---

## Project Structure

```
Data-Transformer-Studio/
├── data_transformer_app.py     # Main Streamlit application (~2,090 lines)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── sample_sales_data.csv       # Example dataset
```

---

## Architecture

### Core Components

1. **Data Import Module** — CSV/Excel/TSV handling, automatic type detection, EDA on load
2. **Transformation Engine** — dplyr verb implementations with 15-step undo (data snapshots) and full reset
3. **Tidy Module** — pivot, separate/unite, 6 join types
4. **String/Date/Factor Module** — 14 stringr ops, lubridate parsing + arithmetic, 7 forcats ops
5. **Clean Module** — missing values (5 strategies), outlier IQR treatment, type conversion
6. **Visualization Layer** — Plotly 6.x-compatible charts across 5 sub-tabs
7. **Plot Editor** — parameterised publication chart builder with live ggplot2 code generation
8. **R Pipeline Tracker** — per-step R code accumulation with multi-mode script generation
9. **Export Module** — 4 data formats + 3 R script variants + theme reference

### Session State & Stability
- 15-step undo history via data snapshots
- Full reset to original uploaded data at any point
- Plotly colour palette guarded against numeric colour columns — prevents `NoneType.constructor` crash in Plotly 6.x
- All `marginal="kde"` replaced with `marginal="box"` for Plotly 6.x compatibility (kde marginals require statsmodels and fail silently)
- Save-to-pipeline buttons decoupled from generate buttons via `st.session_state` (Streamlit does not support nested buttons)
- `buf.seek(0)` applied before Excel `getvalue()` to ensure complete buffer reads

---

## Based on R for Data Science (2e)

| Tab | R4DS Chapters |
|---|---|
| EDA | Ch. 2, 10 |
| Transform | Ch. 3 (dplyr) |
| Tidy / Join | Ch. 5, 19 |
| Strings / Dates / Factors | Ch. 14, 15, 16, 17 |
| Clean | Ch. 18 |
| Visualize / Plot Editor | Ch. 1, 9 (ggplot2) |

---

## Data Cleaning Capabilities

- **Missing Values**: 5 strategies — drop, fill forward, fill backward, fill mean/median, custom value
- **Duplicates**: Remove by column subset with preview
- **Outliers**: IQR-based detection with 3 options (remove, cap, flag)
- **Data Types**: Automatic detection + manual conversion (numeric, character, integer, factor, logical)
- **String Cleaning**: Trim, case conversion, squish, regex operations
- **Date Handling**: Type conversion, component extraction, arithmetic

---

## Visualization Options

### Chart Types (Quick Plot)
1. **Bar** — category vs value, or count
2. **Line** — trends over time
3. **Scatter** — relationships between variables
4. **Histogram** — distribution with box marginal
5. **Box** — statistical summary by groups
6. **Violin** — distribution shape by groups
7. **Density** — probability distributions with rug marginal
8. **Heatmap (count)** — cross-tabulation of two categorical columns

### Additional Charts
- **ECDF** — empirical cumulative distribution
- **Q-Q Plot** — normality assessment
- **Scatter Matrix** — pairwise relationships across numeric columns
- **Scatter + Regression** — OLS (with statsmodels) or LOWESS
- **Bubble** — three-variable scatter with size encoding
- **Faceted** — any geom split by categorical variable (facet_wrap / facet_grid)
- **Time Series** — line with date aggregation and interactive range selector

### Customization
- Axis labels, titles, subtitle, caption (all editable)
- Marker size, colour, opacity
- Font size control
- 14 publication-ready themes
- Color-by categorical or continuous variable
- Legend on/off
- Log scale X/Y, coord_flip, reference lines, trend lines

---

## R Code Generation

All operations generate proper R/tidyverse code using the modern `|>` pipe:

```r
library(tidyverse)
library(stringr)
library(lubridate)
library(forcats)

data <- read_csv("data.csv")

data_clean <- data |>
  filter(...) |>
  select(...) |>
  mutate(...) |>
  group_by(...) |>
  summarize(...) |>
  pivot_longer(...)

# Visualization
ggplot(data_clean, aes(x = ..., y = ..., color = ...)) +
  geom_point(alpha = 0.8) +
  labs(title = "...", subtitle = "...", x = "...", y = "...") +
  theme_minimal()
```

---

## Performance

- **File Size**: Up to 100MB+ depending on system RAM
- **Rows**: Handles millions of rows
- **Real-time Updates**: Interactive filters and transformations
- **Memory Efficient**: Streamlit session state management and 15-step snapshot pruning

---

## Tips & Tricks

### For Best Results

1. **Data Preparation**
   - Upload clean CSV/Excel files
   - Check for encoding issues
   - Remove unnecessary columns early

2. **Transformations**
   - Apply filters first to reduce data size
   - Group before summarizing for performance
   - Use `across()` to transform multiple columns at once

3. **Visualization**
   - Start with Quick Plot for rapid exploration
   - Move to Plot Editor for publication-ready output
   - Use the Time Series sub-tab for date-indexed data

4. **R Code**
   - Copy generated code directly to RStudio or Quarto
   - Use "Wrangling Only" export to separate data prep from plotting
   - Download the ggplot2 Theme Reference from the Export tab

---

## Browser Compatibility

Chrome ✓ · Firefox ✓ · Safari ✓ · Edge ✓

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## Support

If you find this tool helpful, consider buying me a coffee! ☕

<a href="https://buymeacoffee.com/qhl34mcne4" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

For questions or issues: https://www.linkedin.com/in/meronmkifle/

---

## License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

---

## Acknowledgments

- Inspired by **R for Data Science (2e)** by Hadley Wickham, Mine Çetinkaya-Rundel, and Garrett Grolemund
- Built with [Streamlit](https://streamlit.io/)
- Visualizations with [Plotly](https://plotly.com/)
- Data manipulation with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Based on the tidyverse ecosystem: dplyr, ggplot2, tidyr, stringr, lubridate, forcats

## Resources

- [R for Data Science (2e)](https://r4ds.hadley.nz/)
- [dplyr Documentation](https://dplyr.tidyverse.org/)
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Made with ❤️ for data scientists and R enthusiasts**

Last updated: February 2025
