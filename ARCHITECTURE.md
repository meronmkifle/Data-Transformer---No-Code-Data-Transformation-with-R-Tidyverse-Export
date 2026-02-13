# Architecture & Design

## Overview

Data Transformer is a **visual data transformation editor** that generates reproducible R/tidyverse code. The architecture separates concerns between UI, data processing, and code generation.

```
┌─────────────────────────────────────────────┐
│         Streamlit UI Layer                   │
│  (Transformations, Preview, Code Export)    │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│       Session State Management              │
│  (original_data, transformations, data)     │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│      Data Processing Layer (Pandas)         │
│  (apply_transformations, validation)        │
└────────────────────┬────────────────────────┘
                     │
┌────────────────────▼────────────────────────┐
│     Code Generation Layer (R/Tidyverse)     │
│  (generate_r_code, template rendering)      │
└─────────────────────────────────────────────┘
```

## Core Components

### 1. Session State (`st.session_state`)
Persistent state across Streamlit reruns:
- `original_data`: Immutable reference data (loaded file)
- `data`: Current data after transformations
- `transformations`: List of transformation operations
- `r_code`: Generated R code string

### 2. Transformation Object

Each transformation is a dictionary:
```python
{
    'type': 'filter|select|rename|mutate|group_summarize|pivot_longer|pivot_wider|arrange|distinct',
    # Type-specific fields:
    'condition': str,           # filter
    'columns': List[str],       # select
    'mapping': Dict[str, str],  # rename
    'expression': str,          # mutate
    'group_by': List[str],      # group_summarize
    'summaries': List[Dict],    # group_summarize
    ...
}
```

### 3. Data Processing (`apply_transformations()`)

1. Start with `original_data.copy()`
2. Iterate through `transformations` list in order
3. Apply each operation to the current state
4. Return final transformed dataframe
5. Update `st.session_state.data` and regenerate R code

### 4. Code Generation (`generate_r_code()`)

Template-based generation:
1. Start with library imports
2. Add `read_csv()` skeleton
3. For each transformation, append pipe `%>%` expression
4. Remove trailing `%>%` from last operation
5. Add viewing code

## Data Types & Operations

| Operation | Input Type | Output Type | Implementation |
|-----------|-----------|-------------|-----------------|
| Filter | DataFrame → DataFrame | Query string | `df.query()` |
| Select | DataFrame → DataFrame | Column list | `df[cols]` |
| Rename | DataFrame → DataFrame | Dict mapping | `df.rename()` |
| Mutate | DataFrame → DataFrame | Expression | `eval()` + assignment |
| Group/Summarize | DataFrame → DataFrame | Groupby + agg | `df.groupby().agg()` |
| Pivot Longer | DataFrame → DataFrame | Id vars + pivot | `df.melt()` |
| Pivot Wider | DataFrame → DataFrame | Index + columns + values | `df.pivot_table()` |
| Sort | DataFrame → DataFrame | Column + direction | `df.sort_values()` |
| Distinct | DataFrame → DataFrame | Subset + keep_all | `df.drop_duplicates()` |

## R Code Generation Examples

### Filter
Python:
```python
{'type': 'filter', 'condition': 'year == 2024'}
```
R:
```r
filter(year == 2024)
```

### Mutate
Python:
```python
{'type': 'mutate', 'new_column': 'profit', 'expression': 'amount * 0.15'}
```
R:
```r
mutate(profit = amount * 0.15)
```

### Group & Summarize
Python:
```python
{
    'type': 'group_summarize',
    'group_by': ['region', 'salesperson'],
    'summaries': [
        {'column': 'amount', 'func': 'sum', 'name': 'total_amount'},
        {'column': 'units', 'func': 'mean', 'name': 'avg_units'}
    ]
}
```
R:
```r
group_by(`region`, `salesperson`) %>%
summarize(
    `total_amount` = sum(`amount`),
    `avg_units` = mean(`units`),
    .groups = 'drop'
)
```

## Error Handling

1. **Validation**: Check inputs before adding to transformation list
2. **Execution**: Try/catch around each operation in `apply_transformations()`
3. **User Feedback**: Streamlit toast/error messages
4. **Recovery**: Reset to last valid state on error

## Performance Considerations

- **Data Size**: Suitable for <1M rows (limited by Streamlit session memory)
- **Transformation Count**: No practical limit (operations applied sequentially)
- **Preview**: Shows first 20 rows to keep UI responsive
- **Caching**: Could add `@st.cache_data` for repeated file uploads

## Security

- **Input Validation**: Pandas query syntax validated by pandas parser
- **Eval Restriction**: Mutate expressions use restricted eval (no `__builtins__`)
- **File Upload**: Limited to CSV/Excel, server-side validation
- **No SQL Injection**: No database connectivity yet

## Future Enhancements

1. **Undo/Redo**: Maintain transformation history
2. **Joins**: Multi-table operations
3. **String Functions**: Regex, substring, case operations
4. **Advanced Agg**: Window functions, rank, lag/lead
5. **SQL Generation**: Output SQL instead of R
6. **Python Export**: Generate pandas code alongside R
7. **Collaboration**: Share pipelines via URL/cloud
8. **Big Data**: Dask backend for larger datasets
9. **Custom Functions**: User-defined transformations
10. **Visual Analytics**: Built-in charts from transformations

---

For implementation details, see docstrings in `data_transformer_app.py`.
