import streamlit as st
import pandas as pd
import json
from datetime import datetime

st.set_page_config(page_title="Data Transformer", layout="wide")

st.title("📊 Simple Data Transformer")
st.write("Upload data → Transform → Generate R code")

# File upload
uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    # Load data
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.success(f"✅ Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Show original data
    st.subheader("Original Data")
    st.dataframe(df, use_container_width=True)
    
    st.divider()
    
    # Transformations
    st.subheader("🔧 Choose Transformations")
    
    transformed_df = df.copy()
    r_code_lines = []
    
    # 1. FILTER
    st.write("**1. Filter rows**")
    filter_col = st.selectbox("Filter by column", ["None"] + df.columns.tolist())
    
    if filter_col != "None":
        if df[filter_col].dtype == 'object':
            # String column - multiselect
            values = st.multiselect(f"Select {filter_col}", df[filter_col].unique())
            if values:
                transformed_df = transformed_df[transformed_df[filter_col].isin(values)]
                r_code_lines.append(f"filter({filter_col} %in% c({', '.join(repr(v) for v in values)}))")
        else:
            # Numeric column - range slider
            min_val, max_val = st.slider(
                f"Range for {filter_col}",
                float(df[filter_col].min()),
                float(df[filter_col].max()),
                (float(df[filter_col].min()), float(df[filter_col].max()))
            )
            transformed_df = transformed_df[(transformed_df[filter_col] >= min_val) & (transformed_df[filter_col] <= max_val)]
            r_code_lines.append(f"filter({filter_col} >= {min_val}, {filter_col} <= {max_val})")
    
    st.divider()
    
    # 2. SELECT COLUMNS
    st.write("**2. Select columns to keep**")
    selected_cols = st.multiselect("Choose columns", df.columns.tolist(), default=df.columns.tolist())
    if selected_cols:
        transformed_df = transformed_df[selected_cols]
        r_code_lines.append(f"select({', '.join(selected_cols)})")
    
    st.divider()
    
    # 3. SORT
    st.write("**3. Sort data**")
    sort_col = st.selectbox("Sort by column", ["None"] + transformed_df.columns.tolist())
    if sort_col != "None":
        ascending = st.radio("Order", ["Ascending", "Descending"]) == "Ascending"
        transformed_df = transformed_df.sort_values(sort_col, ascending=ascending)
        direction = "asc" if ascending else "desc"
        r_code_lines.append(f"arrange({'' if ascending else 'desc('}{ sort_col}{')' if not ascending else ''})")
    
    st.divider()
    
    # 4. GROUP & SUMMARIZE
    st.write("**4. Group and summarize**")
    group_cols = st.multiselect("Group by columns", transformed_df.columns.tolist())
    
    if group_cols:
        agg_col = st.selectbox("Aggregate column", transformed_df.columns.tolist())
        agg_func = st.selectbox("Aggregation function", ["sum", "mean", "count", "min", "max"])
        
        if agg_func == "sum":
            transformed_df = transformed_df.groupby(group_cols)[agg_col].sum().reset_index()
            transformed_df.columns = list(group_cols) + [f"total_{agg_col}"]
            agg_func_r = "sum"
        elif agg_func == "mean":
            transformed_df = transformed_df.groupby(group_cols)[agg_col].mean().reset_index()
            transformed_df.columns = list(group_cols) + [f"avg_{agg_col}"]
            agg_func_r = "mean"
        elif agg_func == "count":
            transformed_df = transformed_df.groupby(group_cols)[agg_col].count().reset_index()
            transformed_df.columns = list(group_cols) + [f"count_{agg_col}"]
            agg_func_r = "n"
        elif agg_func == "min":
            transformed_df = transformed_df.groupby(group_cols)[agg_col].min().reset_index()
            transformed_df.columns = list(group_cols) + [f"min_{agg_col}"]
            agg_func_r = "min"
        elif agg_func == "max":
            transformed_df = transformed_df.groupby(group_cols)[agg_col].max().reset_index()
            transformed_df.columns = list(group_cols) + [f"max_{agg_col}"]
            agg_func_r = "max"
        
        group_str = ", ".join(group_cols)
        r_code_lines.append(f"group_by({group_str})")
        r_code_lines.append(f"summarize(result = {agg_func_r}({agg_col}))")
    
    st.divider()
    
    # Show transformed data
    st.subheader("✨ Transformed Data")
    st.dataframe(transformed_df, use_container_width=True)
    st.write(f"**Result:** {transformed_df.shape[0]} rows × {transformed_df.shape[1]} columns")
    
    st.divider()
    
    # Generate R code
    st.subheader("💻 R/Tidyverse Code")
    
    if r_code_lines:
        r_code = f"""library(tidyverse)
library(readr)

data <- read_csv("your_data.csv") %>%
  {" %>%\n  ".join(r_code_lines)}

head(data)
"""
    else:
        r_code = """library(tidyverse)
library(readr)

data <- read_csv("your_data.csv")

head(data)
"""
    
    st.code(r_code, language="r")
    
    # Download options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="📥 Download R Script",
            data=r_code,
            file_name=f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
            mime="text/plain"
        )
    
    with col2:
        st.download_button(
            label="📥 Download Transformed CSV",
            data=transformed_df.to_csv(index=False),
            file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col3:
        if st.button("📋 Copy R Code"):
            st.success("Code copied! (Paste with Ctrl+V)")

else:
    st.info("👆 Upload a CSV or Excel file to start")
