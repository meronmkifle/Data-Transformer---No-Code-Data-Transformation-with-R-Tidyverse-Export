import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json

try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Data Transformer Studio", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; padding: 2rem; }
    h1, h2, h3 { color: #1f4788; }
    .section-header { 
        padding: 20px; 
        background: linear-gradient(135deg, #1f4788 0%, #2d5fa3 100%);
        color: white;
        border-radius: 8px;
        margin: 30px 0 20px 0;
        font-size: 1.3em;
        font-weight: 600;
    }
    .spacer { margin: 20px 0; }
    .r_code { background-color: #f5f5f5; border-left: 4px solid #1f4788; padding: 10px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

st.title("Data Transformer Studio Pro")
st.markdown("*R4DS 2e - Transform • Clean • Visualize • LIVE R Code Generation • Publication Themes*")
st.markdown("---")

# Publication-ready themes
PUBLICATION_THEMES = {
    "Theme Minimal": "theme_minimal()",
    "Theme B&W": "theme_bw()",
    "Theme Linedraw": "theme_linedraw()",
    "Hrbrthemes Ipsum": "theme_ipsum()",
    "Hrbrthemes Ipsum RC": "theme_ipsum_rc()",
    "Hrbrthemes FT": "theme_ft_rc()",
    "GGThemes Economist": "theme_economist()",
    "GGThemes WSJ": "theme_wsj()",
    "GGThemes Solarized": "theme_solarized()",
    "GGThemes FiveThirtyEight": "theme_fivethirtyeight()",
    "GGThemes Tufte": "theme_tufte()",
}

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'cleaning_code' not in st.session_state:
    st.session_state.cleaning_code = []
if 'custom_theme' not in st.session_state:
    st.session_state.custom_theme = ""

with st.sidebar:
    st.header("⚙️ Settings & Import")
    st.markdown("---")
    
    # Custom Theme Upload
    st.subheader("Custom Theme Code")
    theme_source = st.radio("Theme source", ["Built-in", "Upload Custom"], key="theme_source")
    
    if theme_source == "Upload Custom":
        theme_code = st.text_area("Paste R theme code (e.g., theme_() function)", 
                                 placeholder="theme_minimal() + theme(...)",
                                 height=100, key="custom_theme_input")
        if theme_code:
            st.session_state.custom_theme = theme_code
            st.success("✓ Custom theme loaded")
    
    st.markdown("---")
    st.subheader("Custom Colors (HEX)")
    custom_colors = st.text_area("JSON format: {\"primary\": \"#1f4788\", \"secondary\": \"#2d5fa3\"}", 
                                placeholder='{"primary": "#1f4788", "secondary": "#2d5fa3"}',
                                height=80, key="custom_colors")
    if custom_colors:
        try:
            color_dict = json.loads(custom_colors)
            st.success("✓ Custom colors loaded")
        except:
            st.warning("Invalid JSON format")
    
    st.markdown("---")
    st.header("Data Import")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)
        
        st.session_state.original_df = df_uploaded.copy()
        st.session_state.data = df_uploaded.copy()
        
        st.success(f"✓ Loaded: {df_uploaded.shape[0]} rows × {df_uploaded.shape[1]} columns")
        st.markdown("---")
        st.subheader("Quick Stats")
        st.write(f"**Rows:** {df_uploaded.shape[0]}")
        st.write(f"**Cols:** {df_uploaded.shape[1]}")
        st.write(f"**Memory:** {df_uploaded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

if st.session_state.data is not None:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Transform", "String/Date", "Clean", "Visualize", "Plot Editor", "Export"])
    
    # ============== TAB 1: TRANSFORM ==============
    with tab1:
        st.markdown("<div class='section-header'>Data Transformation (dplyr)</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("filter() - Select Rows")
        with col2:
            if st.button("↺ Reset", key="reset_transform"):
                st.session_state.data = st.session_state.original_df.copy()
                st.rerun()
        
        col1, col2 = st.columns(2)
        with col1:
            filter_col = st.selectbox("Column to filter", ["None"] + st.session_state.data.columns.tolist(), key="f_col")
        
        if filter_col != "None":
            if st.session_state.data[filter_col].dtype == 'object':
                values = st.multiselect(f"Keep values", st.session_state.data[filter_col].unique(), key="f_vals")
                if values:
                    st.session_state.data = st.session_state.data[st.session_state.data[filter_col].isin(values)]
                    st.success(f"✓ Filtered: {st.session_state.data.shape[0]} rows remain")
            else:
                min_val, max_val = st.slider(f"Range", 
                    float(st.session_state.original_df[filter_col].min()),
                    float(st.session_state.original_df[filter_col].max()),
                    (float(st.session_state.data[filter_col].min()), float(st.session_state.data[filter_col].max())),
                    key="f_range")
                st.session_state.data = st.session_state.data[(st.session_state.data[filter_col] >= min_val) & (st.session_state.data[filter_col] <= max_val)]
                st.success(f"✓ Filtered: {st.session_state.data.shape[0]} rows remain")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("select() - Choose Columns")
        selected_cols = st.multiselect("Columns to keep", st.session_state.data.columns.tolist(), 
                                       default=st.session_state.data.columns.tolist(), key="sel_cols")
        if selected_cols:
            st.session_state.data = st.session_state.data[selected_cols]
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("arrange() - Sort Data")
        col1, col2 = st.columns(2)
        with col1:
            sort_col = st.selectbox("Sort by", ["None"] + st.session_state.data.columns.tolist(), key="sort_col")
        with col2:
            if sort_col != "None":
                ascending = st.radio("Order", ["↑ Ascending", "↓ Descending"], key="sort_order", horizontal=True) == "↑ Ascending"
                st.session_state.data = st.session_state.data.sort_values(sort_col, ascending=ascending)
                st.success(f"✓ Sorted by {sort_col}")
        
        st.markdown("---")
        st.dataframe(st.session_state.data, use_container_width=True, height=400)
        st.write(f"Shape: **{st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns**")
    
    # ============== TAB 2: STRING & DATE ==============
    with tab2:
        st.markdown("<div class='section-header'>String & Date Operations</div>", unsafe_allow_html=True)
        
        st.subheader("String Operations (stringr)")
        string_cols = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
        
        if string_cols:
            col1, col2 = st.columns(2)
            with col1:
                str_col = st.selectbox("String column", string_cols, key="str_col")
                str_op = st.selectbox("Operation", ["uppercase", "lowercase", "title case", "trim"], key="str_op")
            
            if str_op == "uppercase":
                st.session_state.data[f"{str_col}_upper"] = st.session_state.data[str_col].str.upper()
                st.success(f"✓ Created {str_col}_upper")
            elif str_op == "lowercase":
                st.session_state.data[f"{str_col}_lower"] = st.session_state.data[str_col].str.lower()
                st.success(f"✓ Created {str_col}_lower")
            elif str_op == "title case":
                st.session_state.data[f"{str_col}_title"] = st.session_state.data[str_col].str.title()
                st.success(f"✓ Created {str_col}_title")
            elif str_op == "trim":
                st.session_state.data[str_col] = st.session_state.data[str_col].str.strip()
                st.success(f"✓ Trimmed {str_col}")
    
    # ============== TAB 3: CLEAN (WITH LIVE R CODE) ==============
    with tab3:
        st.markdown("<div class='section-header'>Data Cleaning (with LIVE R Code)</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Missing Values")
        with col2:
            if st.button("Clear Code", key="clear_clean_code"):
                st.session_state.cleaning_code = []
        
        missing = st.session_state.data.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"Total missing: {missing.sum()}")
            st.dataframe(missing[missing > 0])
            
            action = st.radio("Action", ["Drop rows", "Drop columns", "Fill forward", "Fill with value", "Fill mean"], key="miss_action")
            
            if action == "Drop rows":
                if st.button("Apply", key="miss_drop_rows"):
                    st.session_state.data = st.session_state.data.dropna()
                    st.session_state.cleaning_code.append("drop_na()")
                    st.success(f"✓ Dropped rows. Shape: {st.session_state.data.shape}")
            
            elif action == "Drop columns":
                cols = st.multiselect("Drop these", missing[missing > 0].index.tolist(), key="miss_drop_cols")
                if st.button("Apply", key="miss_drop_cols_btn"):
                    st.session_state.data = st.session_state.data.drop(columns=cols)
                    cols_str = ", ".join([f"'{c}'" for c in cols])
                    st.session_state.cleaning_code.append(f"select(-c({cols_str}))")
                    st.success(f"✓ Dropped {len(cols)} columns")
            
            elif action == "Fill forward":
                if st.button("Apply", key="miss_ffill"):
                    st.session_state.data = st.session_state.data.fillna(method='ffill')
                    st.session_state.cleaning_code.append("fill(everything(), .direction = 'down')")
                    st.success(f"✓ Filled forward")
            
            elif action == "Fill with value":
                val = st.text_input("Fill value", "0", key="miss_val")
                if st.button("Apply", key="miss_fill_btn"):
                    st.session_state.data = st.session_state.data.fillna(float(val))
                    st.session_state.cleaning_code.append(f"replace_na({val})")
                    st.success(f"✓ Filled with {val}")
            
            elif action == "Fill mean":
                if st.button("Apply", key="miss_mean"):
                    numeric = st.session_state.data.select_dtypes(include=['number']).columns
                    for col in numeric:
                        st.session_state.data[col].fillna(st.session_state.data[col].mean(), inplace=True)
                    st.session_state.cleaning_code.append("mutate(across(where(is.numeric), ~replace_na(., mean(., na.rm = TRUE))))")
                    st.success(f"✓ Filled numeric columns with mean")
        else:
            st.success("No missing values ✓")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("Duplicates")
        dup = st.session_state.data.duplicated().sum()
        st.write(f"Duplicate rows: **{dup}**")
        if dup > 0 and st.button("Remove duplicates", key="remove_dup"):
            st.session_state.data = st.session_state.data.drop_duplicates()
            st.session_state.cleaning_code.append("distinct()")
            st.success(f"✓ Removed {dup} duplicates")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("Outliers (IQR Method)")
        numeric_cols_clean = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols_clean:
            col = st.selectbox("Column", numeric_cols_clean, key="outlier_col")
            Q1 = st.session_state.data[col].quantile(0.25)
            Q3 = st.session_state.data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((st.session_state.data[col] < (Q1 - 1.5 * IQR)) | 
                       (st.session_state.data[col] > (Q3 + 1.5 * IQR))).sum()
            
            st.write(f"Outliers detected: **{outliers}**")
            if outliers > 0:
                action = st.radio("Action", ["Remove", "Cap", "Flag"], key="outlier_action")
                
                if st.button("Apply", key="outlier_btn"):
                    if action == "Remove":
                        st.session_state.data = st.session_state.data[
                            (st.session_state.data[col] >= (Q1 - 1.5 * IQR)) & 
                            (st.session_state.data[col] <= (Q3 + 1.5 * IQR))
                        ]
                        st.session_state.cleaning_code.append(f"filter({col} >= {Q1 - 1.5 * IQR} & {col} <= {Q3 + 1.5 * IQR})")
                        st.success(f"✓ Removed {outliers} outliers")
                    elif action == "Cap":
                        st.session_state.data[col] = st.session_state.data[col].clip(
                            lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
                        st.session_state.cleaning_code.append(f"mutate({col} = pmin({col}, {Q3 + 1.5 * IQR}))")
                        st.success(f"✓ Capped {outliers} outliers")
                    else:
                        st.session_state.data[f"{col}_outlier"] = (
                            (st.session_state.data[col] < (Q1 - 1.5 * IQR)) | 
                            (st.session_state.data[col] > (Q3 + 1.5 * IQR))
                        )
                        st.session_state.cleaning_code.append(f"mutate({col}_outlier = {col} < {Q1 - 1.5 * IQR} | {col} > {Q3 + 1.5 * IQR})")
                        st.success(f"✓ Flagged {outliers} outliers")
        
        # LIVE R CODE FOR CLEANING
        st.markdown("---")
        st.markdown("<div class='section-header'>Generated Cleaning Code (R)</div>", unsafe_allow_html=True)
        if st.session_state.cleaning_code:
            clean_r_code = """data <- data %>%
  """ + " %>%\n  ".join(st.session_state.cleaning_code)
            st.code(clean_r_code, language="r")
        else:
            st.info("Make cleaning changes to generate code")
    
    # ============== TAB 4: VISUALIZE ==============
    with tab4:
        st.markdown("<div class='section-header'>Data Visualization (ggplot2)</div>", unsafe_allow_html=True)
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            numeric_cols_viz = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            cat_cols_viz = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            chart_type = st.selectbox("Chart Type", 
                ["Bar", "Line", "Scatter", "Histogram", "Box", "Violin", "Heatmap"], key="viz_type")
            
            try:
                if chart_type == "Bar" and cat_cols_viz and numeric_cols_viz:
                    x = st.selectbox("X-axis", cat_cols_viz, key="bar_x")
                    y = st.selectbox("Y-axis", numeric_cols_viz, key="bar_y")
                    fig = px.bar(st.session_state.data, x=x, y=y, template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Line" and numeric_cols_viz:
                    x = st.selectbox("X-axis", st.session_state.data.columns.tolist(), key="line_x")
                    y = st.selectbox("Y-axis", numeric_cols_viz, key="line_y")
                    fig = px.line(st.session_state.data, x=x, y=y, markers=True, template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter" and len(numeric_cols_viz) >= 2:
                    x = st.selectbox("X-axis", numeric_cols_viz, key="scatter_x")
                    y = st.selectbox("Y-axis", [c for c in numeric_cols_viz if c != x], key="scatter_y")
                    color = st.selectbox("Color by", ["None"] + cat_cols_viz, key="scatter_color")
                    fig = px.scatter(st.session_state.data, x=x, y=y, 
                                    color=color if color != "None" else None,
                                    template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Histogram" and numeric_cols_viz:
                    x = st.selectbox("Column", numeric_cols_viz, key="hist_x")
                    bins = st.slider("Bins", 5, 100, 30, key="hist_bins")
                    fig = px.histogram(st.session_state.data, x=x, nbins=bins, template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box" and numeric_cols_viz and cat_cols_viz:
                    x = st.selectbox("X-axis", cat_cols_viz, key="box_x")
                    y = st.selectbox("Y-axis", numeric_cols_viz, key="box_y")
                    fig = px.box(st.session_state.data, x=x, y=y, template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Violin" and numeric_cols_viz and cat_cols_viz:
                    x = st.selectbox("X-axis", cat_cols_viz, key="violin_x")
                    y = st.selectbox("Y-axis", numeric_cols_viz, key="violin_y")
                    fig = px.violin(st.session_state.data, x=x, y=y, template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Heatmap" and len(numeric_cols_viz) >= 2:
                    corr = st.session_state.data[numeric_cols_viz].corr()
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                                     colorscale='RdBu', zmid=0))
                    fig.update_layout(template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ============== TAB 5: PLOT EDITOR (WITH THEMES) ==============
    with tab5:
        st.markdown("<div class='section-header'>Advanced Plot Editor (Publication Themes)</div>", unsafe_allow_html=True)
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            numeric_cols_pe = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            cat_cols_pe = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                plot_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box"], key="pe_type")
            with col2:
                theme_choice = st.selectbox("Publication Theme", list(PUBLICATION_THEMES.keys()), key="pe_theme_choice")
            with col3:
                height = st.slider("Height", 300, 800, 550, key="pe_height")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**X Axis**")
                x_col = st.selectbox("Column", st.session_state.data.columns.tolist(), key="pe_x")
                x_label = st.text_input("Label", value=x_col, key="pe_x_label")
            with col2:
                st.write("**Y Axis**")
                y_col = st.selectbox("Column", numeric_cols_pe if numeric_cols_pe else st.session_state.data.columns.tolist(), key="pe_y")
                y_label = st.text_input("Label", value=y_col, key="pe_y_label")
            with col3:
                st.write("**Title**")
                title = st.text_input("Title", value=f"{y_col} by {x_col}", key="pe_title")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                marker_size = st.slider("Marker Size", 1, 20, 8, key="pe_size")
            with col2:
                color = st.color_picker("Color", "#1f4788", key="pe_color")
            with col3:
                opacity = st.slider("Opacity", 0.0, 1.0, 1.0, key="pe_opacity", step=0.1)
            with col4:
                font_size = st.slider("Font Size", 10, 24, 14, key="pe_font")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                color_by = st.selectbox("Color by", ["None"] + cat_cols_pe, key="pe_colorby")
            with col2:
                legend = st.checkbox("Show Legend", True, key="pe_legend")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            try:
                fig = None
                if plot_type == "Bar":
                    fig = px.bar(st.session_state.data, x=x_col, y=y_col, 
                                color=color_by if color_by != "None" else None, template="plotly_white", height=height)
                elif plot_type == "Line":
                    fig = px.line(st.session_state.data, x=x_col, y=y_col, markers=True,
                                 color=color_by if color_by != "None" else None, template="plotly_white", height=height)
                elif plot_type == "Scatter":
                    fig = px.scatter(st.session_state.data, x=x_col, y=y_col,
                                    color=color_by if color_by != "None" else None, template="plotly_white", height=height)
                    fig.update_traces(marker=dict(size=marker_size, opacity=opacity))
                elif plot_type == "Box":
                    fig = px.box(st.session_state.data, x=x_col, y=y_col,
                                color=color_by if color_by != "None" else None, template="plotly_white", height=height)
                
                if fig:
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=font_size, color=color)),
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        showlegend=legend,
                        font=dict(size=font_size),
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # LIVE R CODE WITH PUBLICATION THEME
                    selected_theme = PUBLICATION_THEMES[theme_choice]
                    custom_theme_str = f" + {st.session_state.custom_theme}" if st.session_state.custom_theme else ""
                    
                    r_code = f"""# {title}
ggplot(data, aes(x = {x_col}, y = {y_col}{', color = ' + color_by if color_by != 'None' else ''})) +
  geom_{plot_type.lower()}(size = {marker_size/10}, alpha = {opacity}, fill = '{color}') +
  labs(
    title = '{title}',
    x = '{x_label}',
    y = '{y_label}'
  ) +
  {selected_theme} +
  theme(
    plot.title = element_text(size = {font_size}, face = 'bold', color = '{color}'),
    axis.text = element_text(size = {font_size - 2}),
    legend.position = {'bottom' if legend else 'none'}
  ){custom_theme_str}"""
                    
                    st.markdown("**LIVE R Code (Publication-Ready):**")
                    st.code(r_code, language="r")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ============== TAB 6: EXPORT ==============
    with tab6:
        st.markdown("<div class='section-header'>Export Data & Complete R Analysis Script</div>", unsafe_allow_html=True)
        
        st.subheader("Download Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Transformed CSV",
                data=st.session_state.data.to_csv(index=False),
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Complete analysis script
            complete_r_script = """# Data Transformation Analysis Script
# Generated by Data Transformer Studio Pro

library(tidyverse)
library(ggplot2)
library(lubridate)
library(stringr)

# Optional: Install publication-ready themes
# install.packages('hrbrthemes')
# install.packages('ggthemes')
# install.packages('ggpubr')

# Load data
data <- read_csv('data.csv')

# ===== EXPLORATORY DATA ANALYSIS =====
head(data)
summary(data)
glimpse(data)

# ===== DATA CLEANING =====
"""
            if st.session_state.cleaning_code:
                complete_r_script += "data <- data %>%\n  " + " %>%\n  ".join(st.session_state.cleaning_code)
            else:
                complete_r_script += "# No cleaning operations applied yet\n"
            
            complete_r_script += """

# ===== VISUALIZATIONS =====
# Example bar plot with publication theme
ggplot(data, aes(x = ..., y = ...)) +
  geom_bar(stat = 'identity', fill = '#1f4788', alpha = 0.8) +
  labs(
    title = 'Your Title Here',
    x = 'X Label',
    y = 'Y Label',
    caption = 'Data source: your_source'
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = 'bold', color = '#1f4788'),
    axis.text = element_text(size = 12),
    legend.position = 'bottom'
  )

# Scatter plot with color mapping
ggplot(data, aes(x = ..., y = ..., color = ...)) +
  geom_point(size = 3, alpha = 0.6) +
  theme_ipsum() +  # From hrbrthemes package
  theme(
    plot.title = element_text(size = 14, face = 'bold')
  )

# Box plot by group
ggplot(data, aes(x = ..., y = ...)) +
  geom_boxplot(fill = '#2d5fa3', alpha = 0.7) +
  theme_pubr() +  # From ggpubr package
  theme(legend.position = 'bottom')
"""
            
            st.download_button(
                label="Complete R Script",
                data=complete_r_script,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
                mime="text/plain"
            )
        
        with col3:
            # Theme reference
            theme_ref = """# Publication-Ready ggplot2 Themes Reference

# Built-in themes
theme_minimal()
theme_bw()
theme_linedraw()
theme_light()

# hrbrthemes package (install.packages('hrbrthemes'))
theme_ipsum()          # Classic theme with typography focus
theme_ipsum_rc()       # Roboto Condensed version
theme_ft_rc()          # FT theme

# ggthemes package (install.packages('ggthemes'))
theme_economist()      # Economist magazine style
theme_wsj()            # Wall Street Journal style
theme_solarized()      # Solarized color scheme
theme_fivethirtyeight()# FiveThirtyEight style
theme_tufte()          # Edward Tufte minimalist style
theme_excel()          # Excel-like (for humor)

# ggpubr package (install.packages('ggpubr'))
theme_pubr()           # Publication-ready theme

# Example: Combine theme with customization
ggplot(data, aes(x, y)) +
  geom_point() +
  theme_ipsum() +
  theme(
    plot.title = element_text(face = 'bold', size = 16),
    panel.grid.minor = element_blank(),
    plot.margin = margin(20, 20, 20, 20)
  )
"""
            st.download_button(
                label="Theme Reference",
                data=theme_ref,
                file_name="ggplot2_themes_reference.R",
                mime="text/plain"
            )
        
        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data, use_container_width=True, height=400)
        st.write(f"**Final Shape:** {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")

else:
    st.info("Upload a CSV or Excel file to start")
    st.markdown("---")
    st.markdown("""
    ### Features:
    - **Live R Code Generation** for all operations
    - **Publication-Ready Themes** (hrbrthemes, ggthemes, ggpubr)
    - **Custom Theme Upload** - paste your own R theme code
    - **Custom Colors** - JSON hex color configuration
    - **Data Cleaning** with automatic R code generation
    - **Real-time Updates** - all changes reflected instantly
    """)
