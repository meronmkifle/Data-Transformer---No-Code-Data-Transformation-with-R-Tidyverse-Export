import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

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
</style>
""", unsafe_allow_html=True)

st.title("Data Transformer Studio")
st.markdown("*R4DS 2e - Transform • Clean • Visualize • Export with LIVE UPDATES*")
st.markdown("---")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_df' not in st.session_state:
    st.session_state.original_df = None

with st.sidebar:
    st.header("Data Import")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df_uploaded = pd.read_csv(uploaded_file)
        else:
            df_uploaded = pd.read_excel(uploaded_file)
        
        st.session_state.original_df = df_uploaded.copy()
        st.session_state.data = df_uploaded.copy()
        
        st.success(f"Loaded: {df_uploaded.shape[0]} rows × {df_uploaded.shape[1]} columns")
        st.markdown("---")
        st.subheader("Data Summary")
        st.write(f"**Rows:** {df_uploaded.shape[0]}")
        st.write(f"**Columns:** {df_uploaded.shape[1]}")
        st.write(f"**Memory:** {df_uploaded.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("---")
        st.subheader("Missing Data")
        missing = df_uploaded.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"Total: {missing.sum()} missing values")
        else:
            st.success("No missing values")

if st.session_state.data is not None:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Transform", "String/Date", "Clean", "Visualize", "Plot Editor", "Export"])
    
    # ============== TAB 1: TRANSFORM ==============
    with tab1:
        st.markdown("<div class='section-header'>Data Transformation (dplyr)</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("filter() - Select Rows")
        with col2:
            if st.button("Reset to Original", key="reset_transform"):
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
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("mutate() - Create Columns")
        col1, col2 = st.columns(2)
        with col1:
            new_col = st.text_input("New column name", key="mut_name")
        with col2:
            operation = st.selectbox("Operation", ["Sum", "Multiply", "Divide", "Subtract", "% Change", "Scale", "Log", "Abs", "Round", "Rank"], key="mut_op")
        
        if new_col and operation:
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            
            if operation == "Sum" and len(numeric_cols) >= 2:
                c1, c2 = st.columns(2)
                with c1:
                    col1_opt = st.selectbox("Col 1", numeric_cols, key="m_c1")
                with c2:
                    col2_opt = st.selectbox("Col 2", numeric_cols, key="m_c2")
                st.session_state.data[new_col] = st.session_state.data[col1_opt] + st.session_state.data[col2_opt]
                st.success(f"✓ Created {new_col}")
            
            elif operation == "Multiply":
                col_opt = st.selectbox("Column", numeric_cols, key="m_mult_col")
                factor = st.number_input("Factor", key="m_mult_factor")
                st.session_state.data[new_col] = st.session_state.data[col_opt] * factor
                st.success(f"✓ Created {new_col}")
            
            elif operation == "Divide":
                col_opt = st.selectbox("Column", numeric_cols, key="m_div_col")
                divisor = st.number_input("Divisor", value=1, key="m_div_factor")
                if divisor != 0:
                    st.session_state.data[new_col] = st.session_state.data[col_opt] / divisor
                    st.success(f"✓ Created {new_col}")
            
            elif operation == "Scale":
                col_opt = st.selectbox("Column", numeric_cols, key="m_scale_col")
                min_v = st.session_state.data[col_opt].min()
                max_v = st.session_state.data[col_opt].max()
                st.session_state.data[new_col] = (st.session_state.data[col_opt] - min_v) / (max_v - min_v)
                st.success(f"✓ Created {new_col} (scaled 0-1)")
            
            elif operation == "Log":
                col_opt = st.selectbox("Column", numeric_cols, key="m_log_col")
                st.session_state.data[new_col] = np.log(st.session_state.data[col_opt].clip(lower=0.001))
                st.success(f"✓ Created {new_col}")
            
            elif operation == "Abs":
                col_opt = st.selectbox("Column", numeric_cols, key="m_abs_col")
                st.session_state.data[new_col] = abs(st.session_state.data[col_opt])
                st.success(f"✓ Created {new_col}")
            
            elif operation == "Round":
                col_opt = st.selectbox("Column", numeric_cols, key="m_round_col")
                decimals = st.number_input("Decimals", value=0, min_value=0, key="m_round_dec")
                st.session_state.data[new_col] = round(st.session_state.data[col_opt], decimals)
                st.success(f"✓ Created {new_col}")
            
            elif operation == "Rank":
                col_opt = st.selectbox("Column", numeric_cols, key="m_rank_col")
                st.session_state.data[new_col] = st.session_state.data[col_opt].rank()
                st.success(f"✓ Created {new_col}")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("group_by() & summarize()")
        group_cols = st.multiselect("Group by", st.session_state.data.columns.tolist(), key="grp_cols")
        
        if group_cols:
            numeric_cols = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            col1, col2, col3 = st.columns(3)
            with col1:
                agg_col = st.selectbox("Aggregate column", numeric_cols, key="agg_col")
            with col2:
                agg_func = st.selectbox("Function", ["sum", "mean", "median", "count", "min", "max", "std"], key="agg_func")
            with col3:
                result_name = st.text_input("Result name", value=f"{agg_func}_{agg_col}", key="agg_name")
            
            if st.button("Apply", key="agg_btn"):
                agg_dict = {"sum": np.sum, "mean": np.mean, "median": np.median, "count": "count", 
                           "min": np.min, "max": np.max, "std": np.std}
                st.session_state.data = st.session_state.data.groupby(group_cols)[agg_col].agg(agg_dict[agg_func]).reset_index()
                st.session_state.data.columns = list(group_cols) + [result_name]
                st.success(f"✓ Grouped and summarized")
        
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
                str_op = st.selectbox("Operation", ["uppercase", "lowercase", "title case", "trim", "remove substring", "replace text"], key="str_op")
            with col2:
                pass
            
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
            elif str_op == "remove substring":
                substring = st.text_input("Remove text", key="remove_text")
                if substring:
                    st.session_state.data[f"{str_col}_clean"] = st.session_state.data[str_col].str.replace(substring, "")
                    st.success(f"✓ Created {str_col}_clean")
            elif str_op == "replace text":
                old = st.text_input("Find", key="replace_old")
                new = st.text_input("Replace with", key="replace_new")
                if old:
                    st.session_state.data[f"{str_col}_replaced"] = st.session_state.data[str_col].str.replace(old, new)
                    st.success(f"✓ Created {str_col}_replaced")
        else:
            st.info("No string columns")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("Date/Time Operations (lubridate)")
        
        date_cols = st.session_state.data.select_dtypes(include=['datetime64']).columns.tolist()
        if date_cols:
            date_col = st.selectbox("Date column", date_cols, key="date_col")
            date_op = st.selectbox("Extract", ["year", "month", "day", "weekday", "quarter", "week"], key="date_op")
            
            if date_op == "year":
                st.session_state.data[f"{date_col}_year"] = st.session_state.data[date_col].dt.year
                st.success(f"✓ Extracted year")
            elif date_op == "month":
                st.session_state.data[f"{date_col}_month"] = st.session_state.data[date_col].dt.month
                st.success(f"✓ Extracted month")
            elif date_op == "day":
                st.session_state.data[f"{date_col}_day"] = st.session_state.data[date_col].dt.day
                st.success(f"✓ Extracted day")
            elif date_op == "weekday":
                st.session_state.data[f"{date_col}_dow"] = st.session_state.data[date_col].dt.day_name()
                st.success(f"✓ Extracted day of week")
            elif date_op == "quarter":
                st.session_state.data[f"{date_col}_quarter"] = st.session_state.data[date_col].dt.quarter
                st.success(f"✓ Extracted quarter")
            elif date_op == "week":
                st.session_state.data[f"{date_col}_week"] = st.session_state.data[date_col].dt.isocalendar().week
                st.success(f"✓ Extracted week")
    
    # ============== TAB 3: CLEAN ==============
    with tab3:
        st.markdown("<div class='section-header'>Data Cleaning</div>", unsafe_allow_html=True)
        
        st.subheader("Missing Values")
        missing = st.session_state.data.isnull().sum()
        if missing.sum() > 0:
            st.warning(f"Total missing: {missing.sum()}")
            st.dataframe(missing[missing > 0])
            
            action = st.radio("Action", ["Drop rows", "Drop columns", "Fill forward", "Fill with value", "Fill mean"], key="miss_action")
            
            if action == "Drop rows":
                if st.button("Apply", key="miss_drop_rows"):
                    st.session_state.data = st.session_state.data.dropna()
                    st.success(f"✓ Dropped rows. Shape: {st.session_state.data.shape}")
            
            elif action == "Drop columns":
                cols = st.multiselect("Drop these", missing[missing > 0].index.tolist(), key="miss_drop_cols")
                if st.button("Apply", key="miss_drop_cols_btn"):
                    st.session_state.data = st.session_state.data.drop(columns=cols)
                    st.success(f"✓ Dropped {len(cols)} columns")
            
            elif action == "Fill forward":
                if st.button("Apply", key="miss_ffill"):
                    st.session_state.data = st.session_state.data.fillna(method='ffill')
                    st.success(f"✓ Filled forward")
            
            elif action == "Fill with value":
                val = st.text_input("Fill value", "0", key="miss_val")
                if st.button("Apply", key="miss_fill_btn"):
                    st.session_state.data = st.session_state.data.fillna(float(val))
                    st.success(f"✓ Filled with {val}")
            
            elif action == "Fill mean":
                if st.button("Apply", key="miss_mean"):
                    numeric = st.session_state.data.select_dtypes(include=['number']).columns
                    for col in numeric:
                        st.session_state.data[col].fillna(st.session_state.data[col].mean(), inplace=True)
                    st.success(f"✓ Filled numeric columns with mean")
        else:
            st.success("No missing values ✓")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        st.subheader("Duplicates")
        dup = st.session_state.data.duplicated().sum()
        st.write(f"Duplicate rows: **{dup}**")
        if dup > 0 and st.button("Remove duplicates", key="remove_dup"):
            st.session_state.data = st.session_state.data.drop_duplicates()
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
                        st.success(f"✓ Removed {outliers} outliers")
                    elif action == "Cap":
                        st.session_state.data[col] = st.session_state.data[col].clip(
                            lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)
                        st.success(f"✓ Capped {outliers} outliers")
                    else:
                        st.session_state.data[f"{col}_outlier"] = (
                            (st.session_state.data[col] < (Q1 - 1.5 * IQR)) | 
                            (st.session_state.data[col] > (Q3 + 1.5 * IQR))
                        )
                        st.success(f"✓ Flagged {outliers} outliers")
    
    # ============== TAB 4: VISUALIZE ==============
    with tab4:
        st.markdown("<div class='section-header'>Data Visualization (ggplot2)</div>", unsafe_allow_html=True)
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            numeric_cols_viz = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            cat_cols_viz = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            chart_type = st.selectbox("Chart Type", 
                ["Bar", "Line", "Scatter", "Histogram", "Box", "Violin", "Density", "Heatmap", "Faceted"], key="viz_type")
            
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
                
                elif chart_type == "Density" and numeric_cols_viz:
                    x = st.selectbox("Column", numeric_cols_viz, key="density_x")
                    fig = px.histogram(st.session_state.data, x=x, nbins=50, template="plotly_white", 
                                      height=550, marginal="rug")
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Heatmap" and len(numeric_cols_viz) >= 2:
                    corr = st.session_state.data[numeric_cols_viz].corr()
                    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                                     colorscale='RdBu', zmid=0))
                    fig.update_layout(template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Faceted" and numeric_cols_viz and cat_cols_viz:
                    x = st.selectbox("X-axis", numeric_cols_viz, key="facet_x")
                    y = st.selectbox("Y-axis", numeric_cols_viz, key="facet_y")
                    facet = st.selectbox("Facet by", cat_cols_viz, key="facet_by")
                    fig = px.scatter(st.session_state.data, x=x, y=y, facet_col=facet, 
                                    template="plotly_white", height=550)
                    st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ============== TAB 5: PLOT EDITOR ==============
    with tab5:
        st.markdown("<div class='section-header'>Advanced Plot Customization</div>", unsafe_allow_html=True)
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            numeric_cols_pe = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
            cat_cols_pe = st.session_state.data.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                plot_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box"], key="pe_type")
            with col2:
                theme = st.selectbox("Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"], key="pe_theme")
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
                                color=color_by if color_by != "None" else None, template=theme, height=height)
                elif plot_type == "Line":
                    fig = px.line(st.session_state.data, x=x_col, y=y_col, markers=True,
                                 color=color_by if color_by != "None" else None, template=theme, height=height)
                elif plot_type == "Scatter":
                    fig = px.scatter(st.session_state.data, x=x_col, y=y_col,
                                    color=color_by if color_by != "None" else None, template=theme, height=height)
                    fig.update_traces(marker=dict(size=marker_size, opacity=opacity))
                elif plot_type == "Box":
                    fig = px.box(st.session_state.data, x=x_col, y=y_col,
                                color=color_by if color_by != "None" else None, template=theme, height=height)
                
                if fig:
                    fig.update_layout(
                        title=dict(text=title, font=dict(size=font_size, color="#1f4788")),
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        showlegend=legend,
                        font=dict(size=font_size),
                        hovermode='closest'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # LIVE R CODE UPDATE
                    r_code = f"""# {title}
ggplot(data, aes(x = {x_col}, y = {y_col}{', color = ' + color_by if color_by != 'None' else ''})) +
  geom_{plot_type.lower()}(size = {marker_size/10}, alpha = {opacity}, fill = '{color}') +
  labs(
    title = '{title}',
    x = '{x_label}',
    y = '{y_label}'
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = {font_size}, face = 'bold', color = '#1f4788'),
    axis.text = element_text(size = {font_size - 2}),
    legend.position = {'bottom' if legend else 'none'}
  )"""
                    
                    st.code(r_code, language="r")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ============== TAB 6: EXPORT ==============
    with tab6:
        st.markdown("<div class='section-header'>Export Data & Code</div>", unsafe_allow_html=True)
        
        st.subheader("Download Options")
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download Transformed CSV",
                data=st.session_state.data.to_csv(index=False),
                file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            # Create R template code
            r_template = """library(tidyverse)
library(ggplot2)
library(lubridate)
library(stringr)

# Load data
data <- read_csv('data.csv')

# Explore
head(data)
summary(data)
glimpse(data)

# Your transformations here
data <- data %>%
  filter(...) %>%
  select(...) %>%
  mutate(...)

# Visualize
ggplot(data, aes(x = ..., y = ...)) +
  geom_point() +
  theme_minimal()
"""
            st.download_button(
                label="Download R Template",
                data=r_template,
                file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
                mime="text/plain"
            )
        
        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data, use_container_width=True, height=400)
        st.write(f"**Final Shape:** {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} columns")

else:
    st.info("Upload a CSV or Excel file to start")
