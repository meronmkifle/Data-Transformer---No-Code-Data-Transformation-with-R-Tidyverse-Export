# Data Transformer Studio

A comprehensive, interactive data transformation and visualization tool built with Streamlit, inspired by R for Data Science (2e) principles. Transform raw data into publication-ready insights with no code required, while generating production-grade R/tidyverse code.

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Streamlit](https://img.shields.io/badge/streamlit-1.28%2B-red)

## Features

### Transform (dplyr verbs)
- **filter()** - Keep rows matching conditions
- **select()** - Choose and organize columns
- **arrange()** - Sort data ascending/descending
- **mutate()** - Create new columns with:
  - Arithmetic operations (Sum, Multiply, Divide, Subtract)
  - Statistical transformations (% Change, Scale 0-1, Log, Abs Value, Round, Rank)
- **group_by() & summarize()** - Aggregate data with 10+ functions:
  - sum, mean, median, count, min, max, std, var, first, last
- **pivot_wider() / pivot_longer()** - Reshape data between wide and long formats

### Wrangle (stringr & lubridate)
- **String Operations**:
  - Case conversion (uppercase, lowercase, title case)
  - Whitespace trimming
  - Text replacement and removal
  - Regex pattern extraction
  - Column splitting by separator
  
- **Date/Time Operations**:
  - Extract components (year, month, day, quarter, week)
  - Day of week extraction
  - Automatic date type conversion

### Clean (Data Quality)
- **Missing Value Handling**:
  - Drop rows or columns with NA
  - Fill forward/backward
  - Fill with custom value or mean
  
- **Duplicate Management**: Remove duplicate rows with distinct()
  
- **Outlier Detection**: IQR method with 3 actions:
  - Remove outliers
  - Cap outliers to bounds
  - Flag outliers as boolean column

### Visualize (ggplot2)
11 interactive chart types with Grammar of Graphics:
- Bar Chart
- Line Chart
- Scatter Plot
- Histogram
- Box Plot
- Violin Plot
- Pie Chart
- Area Chart
- Density Plot
- Heatmap (Correlation)
- Faceted Plots

### Plot Editor
Full customization of every visualization:
- Axis labels and title (editable)
- Marker size, color, and opacity
- Font sizes and legend control
- 5 professional themes
- Color-by column mapping
- Real-time R code generation

### Export
Generate publication-ready code:
- **R Script**: Complete dplyr, ggplot2, stringr, lubridate code
- **CSV Data**: Transformed dataset
- **All R code is proper, runnable tidyverse syntax**

## Usage

### Basic Workflow

1. **Upload Data**
   - CSV or Excel files
   - Automatic data type detection
   - Quick statistics and missing value summary

2. **Transform** (Tab 1)
   - Filter rows by values or ranges
   - Select columns to keep
   - Create computed columns
   - Sort and arrange data
   - Group and summarize

3. **String/Date Operations** (Tab 2)
   - Manipulate text data
   - Extract date components
   - Convert and transform strings

4. **Clean** (Tab 3)
   - Handle missing values
   - Remove duplicates
   - Detect and handle outliers

5. **Visualize** (Tab 4)
   - Quick plots with Plotly
   - Interactive exploration
   - 11 chart types

6. **Plot Editor** (Tab 5)
   - Customize every aspect
   - Edit labels and titles
   - Control colors and fonts
   - View live R code

7. **Export** (Tab 6)
   - Download R script
   - Download transformed CSV
   - Copy ready-to-run code

### Example: Sales Data Analysis

```
1. Upload sales_data.csv (100,000 rows)
2. Filter by year == 2024
3. Select: date, region, product, amount
4. Create: profit = amount * 0.15
5. Group by region, sum amount
6. Visualize as Bar Chart
7. Export R code and CSV
```

Generated R code:
```r
library(tidyverse)

data <- read_csv('sales_data.csv') %>%
  filter(year == 2024) %>%
  select(date, region, product, amount) %>%
  mutate(profit = amount * 0.15) %>%
  group_by(region) %>%
  summarize(total_amount = sum(amount), .groups = 'drop')

# Visualization
ggplot(data, aes(x = region, y = total_amount)) +
  geom_bar(stat = 'identity', fill = '#1f4788') +
  labs(title = 'Total Sales by Region', x = 'Region', y = 'Amount') +
  theme_minimal()
```

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

## Project Structure

```
Data-Transformer-Studio/
├── data_transformer_app.py     # Main Streamlit application
├── requirements.txt             # Python dependencies
├── README.md                   # This file
└── sample_sales_data.csv       # Example dataset
```

## Architecture

### Core Components

1. **Data Import Module**
   - CSV/Excel file handling
   - Automatic type detection
   - EDA statistics

2. **Transformation Engine**
   - dplyr verb implementations
   - Safe data operations
   - Error handling

3. **Visualization Layer**
   - Plotly interactive charts
   - ggplot2-style customization
   - Theme management

4. **R Code Generator**
   - Proper tidyverse syntax
   - Grammar of graphics formatting
   - Function library integration (stringr, lubridate)

5. **Export Module**
   - R script generation
   - CSV/JSON export
   - Code formatting

## Based on R for Data Science (2e)

This tool implements best practices from Hadley Wickham's "R for Data Science" (2e):

- **Tidy Data Principles**: Variables in columns, observations in rows
- **dplyr Verbs**: filter, select, arrange, mutate, group_by, summarize, pivot
- **Grammar of Graphics**: ggplot2 layering and aesthetic mapping
- **Functional Programming**: Pipes and composition
- **Reproducible Workflows**: Generate complete R code

Perfect for learning R data science workflows!

## Data Cleaning Capabilities

- **Missing Values**: 5 strategies (drop, fill forward/backward, mean, custom)
- **Duplicates**: Remove exact duplicates
- **Outliers**: IQR-based detection with 3 options (remove, cap, flag)
- **Data Types**: Automatic and manual conversion
- **String Cleaning**: Trim, case conversion, regex operations
- **Date Handling**: Type conversion, component extraction

## Visualization Options

### Chart Types
1. **Bar** - Category vs Value
2. **Line** - Trends over time
3. **Scatter** - Relationships between variables
4. **Histogram** - Distribution of numeric data
5. **Box** - Statistical summary by groups
6. **Violin** - Distribution shape by groups
7. **Pie** - Proportions and percentages
8. **Area** - Cumulative trends
9. **Density** - Probability distributions
10. **Heatmap** - Correlation matrices
11. **Faceted** - Same plot by groups

### Customization
- Axis labels and titles (editable)
- Marker size, color, opacity
- Font sizing
- 5 professional themes
- Color-by categorical variable
- Legend control

## R Code Generation

All operations generate proper R/tidyverse code:

```r
library(tidyverse)
library(ggplot2)
library(stringr)
library(lubridate)

# Data transformation
data <- read_csv('data.csv') %>%
  filter(...) %>%
  select(...) %>%
  mutate(...) %>%
  group_by(...) %>%
  summarize(...) %>%
  pivot_longer(...)

# Visualization
ggplot(data, aes(x = ..., y = ...)) +
  geom_point() +
  labs(title = "...", x = "...", y = "...") +
  theme_minimal()
```

## Performance

- **File Size**: Up to 100MB+ depending on system RAM
- **Rows**: Handles millions of rows
- **Real-time Updates**: Interactive filters and transformations
- **Memory Efficient**: Streamlit caching and optimization

## Tips & Tricks

### For Best Results

1. **Data Preparation**
   - Upload clean CSV/Excel files
   - Check for encoding issues
   - Remove unnecessary columns first

2. **Transformations**
   - Apply filters early to reduce data size
   - Group before summarizing for performance
   - Use pivot operations for reshaping

3. **Visualization**
   - Start with quick plots (Tab 4)
   - Customize in Plot Editor (Tab 5)
   - Export R code for fine-tuning

4. **R Code**
   - Copy generated code to RStudio
   - Add more transformations as needed
   - Use for reproducible reports

## Browser Compatibility

- Chrome/Chromium ✓
- Firefox ✓
- Safari ✓
- Edge ✓

## Performance Tips

- Filter data early (reduces memory usage)
- Use summarize to reduce row count
- Close browser tabs to free RAM
- For 1M+ rows, consider sampling first

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Inspired by **R for Data Science (2e)** by Hadley Wickham, Mine Çetinkaya-Rundel, and Garrett Grolemund
- Built with [Streamlit](https://streamlit.io/)
- Visualizations with [Plotly](https://plotly.com/)
- Data manipulation with [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/)
- Based on tidyverse ecosystem (dplyr, ggplot2, stringr, lubridate, tidyr)

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/Data-Transformer-Studio/issues)
- Email: your.email@example.com

## Resources

- [R for Data Science (2e)](https://r4ds.hadley.nz/)
- [dplyr Documentation](https://dplyr.tidyverse.org/)
- [ggplot2 Documentation](https://ggplot2.tidyverse.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

**Made with ❤️ for data scientists and R enthusiasts**

Last updated: February 2025
