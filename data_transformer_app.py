import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Try to import plotly, if not available use matplotlib
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Transformer Studio", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for RStudio-like appearance
st.markdown("""
<style>
    .main {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
    }
    h1, h2, h3 {
        color: #003f5c;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 5px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Data Transformer Studio")
st.markdown("*Transform • Analyze • Visualize • Code Generate*")

# ============================================
# SIDEBAR: DATA UPLOAD & BASIC INFO
# ============================================
with st.sidebar:
    st.header("📁 Data Import")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        # Load data
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"✅ Loaded: {df.shape[0]} rows")
        st.write(f"**Columns:** {df.shape[1]}")
        
        # Data types summary
        st.subheader("Data Types")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"• {dtype}: {count}")

if uploaded_file:
    # ============================================
    # MAIN AREA: 3 TABS (Transform, Visualize, Code)
    # ============================================
    
    tab1, tab2, tab3 = st.tabs(["🔧 Transform", "📊 Visualize", "💻 R Code"])
    
    # Initialize transformation
    transformed_df = df.copy()
    r_code_lines = []
    
    # ============================================
    # TAB 1: TRANSFORM
    # ============================================
    with tab1:
        st.subheader("Data Transformation Pipeline")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Step 1️⃣: Filter Rows")
            filter_col = st.selectbox("Column to filter", ["None"] + df.columns.tolist(), key="filter1")
            
            if filter_col != "None":
                if df[filter_col].dtype == 'object':
                    values = st.multiselect(f"Values in {filter_col}", df[filter_col].unique(), key="filter2")
                    if values:
                        transformed_df = transformed_df[transformed_df[filter_col].isin(values)]
                        r_code_lines.append(f"filter({filter_col} %in% c({', '.join(repr(v) for v in values)}))")
                else:
                    min_val, max_val = st.slider(
                        f"Range for {filter_col}",
                        float(df[filter_col].min()),
                        float(df[filter_col].max()),
                        (float(df[filter_col].min()), float(df[filter_col].max())),
                        key="filter3"
                    )
                    transformed_df = transformed_df[(transformed_df[filter_col] >= min_val) & (transformed_df[filter_col] <= max_val)]
                    r_code_lines.append(f"filter({filter_col} >= {min_val}, {filter_col} <= {max_val})")
        
        with col2:
            st.write("### Step 2️⃣: Select Columns")
            selected_cols = st.multiselect(
                "Keep columns",
                df.columns.tolist(),
                default=df.columns.tolist(),
                key="select1"
            )
            if selected_cols:
                transformed_df = transformed_df[selected_cols]
                r_code_lines.append(f"select({', '.join(selected_cols)})")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("### Step 3️⃣: Sort Data")
            sort_col = st.selectbox("Sort by", ["None"] + transformed_df.columns.tolist(), key="sort1")
            if sort_col != "None":
                ascending = st.radio("Order", ["Ascending ↑", "Descending ↓"], key="sort2") == "Ascending ↑"
                transformed_df = transformed_df.sort_values(sort_col, ascending=ascending)
                r_code_lines.append(f"arrange({'' if ascending else 'desc('}{ sort_col}{')' if not ascending else ''})")
        
        with col4:
            st.write("### Step 4️⃣: Group & Summarize")
            group_cols = st.multiselect("Group by", transformed_df.columns.tolist(), key="group1")
            
            if group_cols:
                agg_col = st.selectbox("Aggregate column", transformed_df.columns.tolist(), key="group2")
                agg_func = st.selectbox("Function", ["sum", "mean", "count", "min", "max"], key="group3")
                
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
        st.subheader("📋 Transformed Data Preview")
        st.dataframe(transformed_df, use_container_width=True, height=400)
        st.write(f"**Shape:** {transformed_df.shape[0]} rows × {transformed_df.shape[1]} columns")
    
    # ============================================
    # TAB 2: VISUALIZE
    # ============================================
    with tab2:
        st.subheader("Interactive Data Visualization")
        
        viz_type = st.selectbox(
            "Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart", "Heatmap"],
            key="viz1"
        )
        
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns.tolist()
        
        try:
            if viz_type == "Bar Chart":
                if categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis (Category)", categorical_cols, key="bar1")
                    with col2:
                        y_col = st.selectbox("Y-axis (Value)", numeric_cols, key="bar2")
                    
                    if HAS_PLOTLY:
                        fig = px.bar(transformed_df, x=x_col, y=y_col, title=f"{y_col} by {x_col}",
                                    color=x_col, height=500, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Plotly not available. Install with: pip install plotly")
            
            elif viz_type == "Line Chart":
                if numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        x_col = st.selectbox("X-axis", transformed_df.columns.tolist(), key="line1")
                    with col2:
                        y_col = st.selectbox("Y-axis", numeric_cols, key="line2")
                    
                    if HAS_PLOTLY:
                        fig = px.line(transformed_df, x=x_col, y=y_col, title=f"{y_col} over {x_col}",
                                     markers=True, height=500, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Scatter Plot":
                if len(numeric_cols) >= 2:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        x_col = st.selectbox("X-axis", numeric_cols, key="scatter1")
                    with col2:
                        y_col = st.selectbox("Y-axis", [c for c in numeric_cols if c != x_col], key="scatter2")
                    with col3:
                        color_col = st.selectbox("Color by", ["None"] + categorical_cols, key="scatter3")
                    
                    if HAS_PLOTLY:
                        if color_col != "None":
                            fig = px.scatter(transformed_df, x=x_col, y=y_col, color=color_col,
                                           title=f"{x_col} vs {y_col}", height=500, template="plotly_white")
                        else:
                            fig = px.scatter(transformed_df, x=x_col, y=y_col,
                                           title=f"{x_col} vs {y_col}", height=500, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Histogram":
                if numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        col = st.selectbox("Column", numeric_cols, key="hist1")
                    with col2:
                        bins = st.slider("Bins", 5, 50, 20, key="hist2")
                    
                    if HAS_PLOTLY:
                        fig = px.histogram(transformed_df, x=col, nbins=bins,
                                         title=f"Distribution of {col}", height=500, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Box Plot":
                if numeric_cols and categorical_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        y_col = st.selectbox("Y-axis (Numeric)", numeric_cols, key="box1")
                    with col2:
                        x_col = st.selectbox("X-axis (Category)", categorical_cols, key="box2")
                    
                    if HAS_PLOTLY:
                        fig = px.box(transformed_df, x=x_col, y=y_col,
                                    title=f"{y_col} by {x_col}", height=500, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Pie Chart":
                if categorical_cols and numeric_cols:
                    col1, col2 = st.columns(2)
                    with col1:
                        names_col = st.selectbox("Categories", categorical_cols, key="pie1")
                    with col2:
                        values_col = st.selectbox("Values", numeric_cols, key="pie2")
                    
                    if HAS_PLOTLY:
                        fig = px.pie(transformed_df, names=names_col, values=values_col,
                                    title=f"{values_col} by {names_col}", height=500, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Heatmap":
                if len(numeric_cols) >= 2:
                    corr_matrix = transformed_df[numeric_cols].corr()
                    
                    if HAS_PLOTLY:
                        fig = go.Figure(data=go.Heatmap(
                            z=corr_matrix.values,
                            x=corr_matrix.columns,
                            y=corr_matrix.columns,
                            colorscale='Viridis'
                        ))
                        fig.update_layout(title="Correlation Matrix", height=500,
                                        template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"Visualization error: {str(e)}")
    
    # ============================================
    # TAB 3: R CODE GENERATION
    # ============================================
    with tab3:
        st.subheader("Generated R/Tidyverse Code")
        
        if r_code_lines:
            r_code = f"""library(tidyverse)
library(readr)

# Read and transform data
data <- read_csv("your_data.csv") %>%
  {" %>%" + chr(10) + "  "}
  {(" %>%" + chr(10) + "  ").join(r_code_lines)}

# View results
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
                label="📥 R Script",
                data=r_code,
                file_name=f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="📥 CSV Data",
                data=transformed_df.to_csv(index=False),
                file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            if st.button("📋 Copy Code"):
                st.success("✅ Copied to clipboard!")

else:
    st.info("👈 **Upload a CSV or Excel file in the sidebar to begin**")
    
    st.write("""
    ### Features:
    - 🔧 **Transform** - Filter, select, sort, group & summarize
    - 📊 **Visualize** - Interactive charts with Plotly
    - 💻 **Code Generate** - Get R/tidyverse code
    - 📥 **Export** - Download transformed data
    """)
