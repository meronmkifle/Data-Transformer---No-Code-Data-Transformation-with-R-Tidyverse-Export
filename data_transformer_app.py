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
    .main { 
        background-color: #f8f9fa; 
        padding: 2rem;
    }
    .sidebar { background-color: #ffffff; }
    h1, h2, h3 { color: #1f4788; margin-top: 2rem; margin-bottom: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 5px; padding: 10px 20px; }
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
st.markdown("*Transform • Clean • Analyze • Visualize • Export*", unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.header("Data Import")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
        
        st.markdown("---")
        st.subheader("Data Summary")
        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")
        st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        st.markdown("---")
        st.subheader("Column Types")
        for dtype in df.dtypes.unique():
            count = (df.dtypes == dtype).sum()
            st.write(f"{dtype}: {count}")

if uploaded_file:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Transform", "Clean", "Visualize", "Plot Editor", "Export"])
    
    transformed_df = df.copy()
    r_code_transform = []
    r_code_viz = []
    
    # ============== TAB 1: TRANSFORM ==============
    with tab1:
        st.markdown("<div class='section-header'>Data Transformation Pipeline</div>", unsafe_allow_html=True)
        
        st.subheader("Step 1: Filter Rows")
        st.markdown("Remove rows based on column values")
        col1, col2 = st.columns(2)
        
        with col1:
            filter_col = st.selectbox("Column to filter", ["None"] + df.columns.tolist(), key="f1")
        
        if filter_col != "None":
            st.markdown("")
            if df[filter_col].dtype == 'object':
                values = st.multiselect(f"Select values in {filter_col}", df[filter_col].unique(), key="f2")
                if values:
                    transformed_df = transformed_df[transformed_df[filter_col].isin(values)]
                    val_str = "', '".join([str(v) for v in values])
                    r_code_transform.append(f"filter({filter_col} %in% c('{val_str}'))")
                    st.success(f"Filtered: Kept rows where {filter_col} in {values}")
            else:
                min_val, max_val = st.slider(
                    f"Range for {filter_col}",
                    float(df[filter_col].min()),
                    float(df[filter_col].max()),
                    (float(df[filter_col].min()), float(df[filter_col].max())),
                    key="f3"
                )
                transformed_df = transformed_df[(transformed_df[filter_col] >= min_val) & (transformed_df[filter_col] <= max_val)]
                r_code_transform.append(f"filter({filter_col} >= {min_val} & {filter_col} <= {max_val})")
                st.success(f"Filtered: {filter_col} between {min_val} and {max_val}")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Step 2: Select Columns")
        st.markdown("Choose which columns to keep")
        selected_cols = st.multiselect("Columns to keep", df.columns.tolist(), default=df.columns.tolist(), key="s1")
        if selected_cols and selected_cols != df.columns.tolist():
            transformed_df = transformed_df[selected_cols]
            cols_r = ", ".join(selected_cols)
            r_code_transform.append(f"select({cols_r})")
            st.success(f"Selected {len(selected_cols)} columns")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Step 3: Sort Data")
        st.markdown("Arrange rows by column values")
        col1, col2 = st.columns(2)
        
        with col1:
            sort_col = st.selectbox("Sort by column", ["None"] + transformed_df.columns.tolist(), key="so1")
        with col2:
            if sort_col != "None":
                ascending = st.radio("Order", ["Ascending", "Descending"], key="so2", horizontal=True) == "Ascending"
                transformed_df = transformed_df.sort_values(sort_col, ascending=ascending)
                if ascending:
                    r_code_transform.append(f"arrange({sort_col})")
                else:
                    r_code_transform.append(f"arrange(desc({sort_col}))")
                st.success(f"Sorted by {sort_col} ({('ascending' if ascending else 'descending')})")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Step 4: Create Columns (Mutate)")
        st.markdown("Create new columns based on existing data")
        mutate_option = st.radio("Create new column?", ["No", "Yes"], key="m1", horizontal=True)
        
        if mutate_option == "Yes":
            col1, col2 = st.columns(2)
            with col1:
                new_col_name = st.text_input("New column name", key="m2")
            with col2:
                operation = st.selectbox("Operation", ["Sum", "Multiply", "Divide", "Subtract", "Percentage Change", "Scale (0-1)", "Log"], key="m3")
            
            if new_col_name:
                if operation == "Sum":
                    col1_opt = st.selectbox("Column 1", transformed_df.columns, key="m5")
                    col2_opt = st.selectbox("Column 2", transformed_df.columns, key="m6")
                    transformed_df[new_col_name] = transformed_df[col1_opt] + transformed_df[col2_opt]
                    r_code_transform.append(f"mutate({new_col_name} = {col1_opt} + {col2_opt})")
                    st.success(f"Created {new_col_name} = {col1_opt} + {col2_opt}")
                
                elif operation == "Multiply":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m7")
                    factor = st.number_input("Multiply by", key="m8")
                    transformed_df[new_col_name] = transformed_df[col_opt] * factor
                    r_code_transform.append(f"mutate({new_col_name} = {col_opt} * {factor})")
                    st.success(f"Created {new_col_name} = {col_opt} × {factor}")
                
                elif operation == "Divide":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m9")
                    divisor = st.number_input("Divide by", value=1, key="m10")
                    if divisor != 0:
                        transformed_df[new_col_name] = transformed_df[col_opt] / divisor
                        r_code_transform.append(f"mutate({new_col_name} = {col_opt} / {divisor})")
                        st.success(f"Created {new_col_name} = {col_opt} ÷ {divisor}")
                
                elif operation == "Subtract":
                    col1_opt = st.selectbox("Column 1", transformed_df.columns, key="m11")
                    col2_opt = st.selectbox("Column 2", transformed_df.columns, key="m12")
                    transformed_df[new_col_name] = transformed_df[col1_opt] - transformed_df[col2_opt]
                    r_code_transform.append(f"mutate({new_col_name} = {col1_opt} - {col2_opt})")
                    st.success(f"Created {new_col_name} = {col1_opt} - {col2_opt}")
                
                elif operation == "Percentage Change":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m13")
                    transformed_df[new_col_name] = transformed_df[col_opt].pct_change() * 100
                    r_code_transform.append(f"mutate({new_col_name} = (({col_opt} - lag({col_opt})) / lag({col_opt})) * 100)")
                    st.success(f"Created {new_col_name} as % change")
                
                elif operation == "Scale (0-1)":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m14")
                    min_val = transformed_df[col_opt].min()
                    max_val = transformed_df[col_opt].max()
                    transformed_df[new_col_name] = (transformed_df[col_opt] - min_val) / (max_val - min_val)
                    r_code_transform.append(f"mutate({new_col_name} = scale({col_opt}))")
                    st.success(f"Created {new_col_name} (scaled 0-1)")
                
                elif operation == "Log":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m15")
                    transformed_df[new_col_name] = np.log(transformed_df[col_opt])
                    r_code_transform.append(f"mutate({new_col_name} = log({col_opt}))")
                    st.success(f"Created {new_col_name} (log)")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Step 5: Group & Summarize")
        st.markdown("Aggregate data by groups")
        group_cols = st.multiselect("Group by columns", transformed_df.columns.tolist(), key="g1")
        
        if group_cols:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                agg_col = st.selectbox("Column to aggregate", transformed_df.columns.tolist(), key="g2")
            with col2:
                agg_func = st.selectbox("Function", ["sum", "mean", "median", "count", "min", "max", "std"], key="g3")
            with col3:
                result_col_name = st.text_input("Result name", value=f"{agg_func}_{agg_col}", key="g4")
            
            agg_map = {
                "sum": (np.sum, "sum"),
                "mean": (np.mean, "mean"),
                "median": (np.median, "median"),
                "count": (np.count_nonzero, "n"),
                "min": (np.min, "min"),
                "max": (np.max, "max"),
                "std": (np.std, "sd"),
            }
            
            func, r_func = agg_map[agg_func]
            transformed_df = transformed_df.groupby(group_cols)[agg_col].agg(func).reset_index()
            transformed_df.columns = list(group_cols) + [result_col_name]
            
            group_str = ", ".join(group_cols)
            r_code_transform.append(f"group_by({group_str})")
            r_code_transform.append(f"summarize({result_col_name} = {r_func}({agg_col}), .groups = 'drop')")
            st.success(f"Grouped and summarized")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Step 6: Pivot Data")
        st.markdown("Reshape data (wider/longer)")
        pivot_option = st.radio("Reshape data?", ["No", "Wider", "Longer"], key="p1", horizontal=True)
        
        if pivot_option == "Wider":
            idx_cols = st.multiselect("ID columns", transformed_df.columns.tolist(), key="pw1")
            if idx_cols:
                remaining = [c for c in transformed_df.columns if c not in idx_cols]
                if len(remaining) >= 2:
                    col_col = st.selectbox("Names from", remaining, key="pw2")
                    val_col = st.selectbox("Values from", [c for c in remaining if c != col_col], key="pw3")
                    
                    if st.button("Apply Pivot Wider", key="pw_btn"):
                        transformed_df = transformed_df.pivot_table(
                            index=idx_cols,
                            columns=col_col,
                            values=val_col,
                            aggfunc='first'
                        ).reset_index()
                        id_str = ", ".join(idx_cols)
                        r_code_transform.append(f"pivot_wider(id_cols = c({id_str}), names_from = {col_col}, values_from = {val_col})")
                        st.success("Pivoted wider")
        
        elif pivot_option == "Longer":
            id_cols = st.multiselect("ID columns (keep)", transformed_df.columns.tolist(), key="pl1")
            value_cols = st.multiselect("Columns to pivot", [c for c in transformed_df.columns if c not in id_cols], key="pl2")
            
            if value_cols:
                col1, col2 = st.columns(2)
                with col1:
                    names_to = st.text_input("New name column", value="variable", key="pl3")
                with col2:
                    values_to = st.text_input("New value column", value="value", key="pl4")
                
                if st.button("Apply Pivot Longer", key="pl_btn"):
                    if id_cols:
                        transformed_df = transformed_df.melt(id_vars=id_cols, value_vars=value_cols, 
                                                             var_name=names_to, value_name=values_to)
                    else:
                        transformed_df = transformed_df.melt(value_vars=value_cols, 
                                                             var_name=names_to, value_name=values_to)
                    val_cols_str = ", ".join(value_cols)
                    r_code_transform.append(f"pivot_longer(cols = c({val_cols_str}), names_to = '{names_to}', values_to = '{values_to}')")
                    st.success("Pivoted longer")
        
        st.markdown("---")
        st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(transformed_df, use_container_width=True, height=450)
        st.write(f"**Shape:** {transformed_df.shape[0]} rows × {transformed_df.shape[1]} columns")
    
    # ============== TAB 2: CLEAN ==============
    with tab2:
        st.markdown("<div class='section-header'>Data Cleaning Tools</div>", unsafe_allow_html=True)
        
        st.subheader("Missing Values")
        missing_counts = transformed_df.isnull().sum()
        if missing_counts.sum() > 0:
            st.warning(f"Total missing: {missing_counts.sum()}")
            st.dataframe(missing_counts[missing_counts > 0])
            
            col1, col2 = st.columns(2)
            with col1:
                missing_action = st.radio("Action", ["Drop rows", "Drop columns", "Fill forward", "Fill with value"], key="c1")
            
            if missing_action == "Drop rows":
                if st.button("Drop rows with NA", key="c3"):
                    transformed_df = transformed_df.dropna()
                    r_code_transform.append("drop_na()")
                    st.success(f"Dropped rows. New shape: {transformed_df.shape}")
            
            elif missing_action == "Drop columns":
                cols_to_drop = st.multiselect("Drop columns", missing_counts[missing_counts > 0].index.tolist(), key="c4")
                if st.button("Apply", key="c5"):
                    transformed_df = transformed_df.drop(columns=cols_to_drop)
                    r_code_transform.append(f"select(-c({', '.join(cols_to_drop)}))")
                    st.success(f"Dropped {len(cols_to_drop)} columns")
            
            elif missing_action == "Fill forward":
                if st.button("Fill forward (ffill)", key="c6"):
                    transformed_df = transformed_df.fillna(method='ffill')
                    r_code_transform.append("fill(everything(), .direction = 'down')")
                    st.success("Filled forward")
            
            elif missing_action == "Fill with value":
                fill_val = st.text_input("Fill value", "0", key="c7")
                if st.button("Apply", key="c8"):
                    try:
                        transformed_df = transformed_df.fillna(float(fill_val))
                        r_code_transform.append(f"replace_na({fill_val})")
                        st.success(f"Filled with {fill_val}")
                    except:
                        st.error("Invalid")
        else:
            st.success("No missing values")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Duplicates")
        dup_count = transformed_df.duplicated().sum()
        st.write(f"Duplicate rows: {dup_count}")
        
        if dup_count > 0:
            if st.button("Remove duplicates", key="c9"):
                transformed_df = transformed_df.drop_duplicates()
                r_code_transform.append("distinct()")
                st.success(f"Removed {dup_count}")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        st.subheader("Outliers (IQR Method)")
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            col_to_check = st.selectbox("Column", numeric_cols, key="c10")
            
            Q1 = transformed_df[col_to_check].quantile(0.25)
            Q3 = transformed_df[col_to_check].quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((transformed_df[col_to_check] < (Q1 - 1.5 * IQR)) | 
                       (transformed_df[col_to_check] > (Q3 + 1.5 * IQR))).sum()
            
            st.write(f"Outliers detected: {outliers}")
            
            if outliers > 0:
                outlier_action = st.radio("Action", ["Remove", "Cap"], key="c11", horizontal=True)
                
                if st.button("Apply", key="c12"):
                    if outlier_action == "Remove":
                        transformed_df = transformed_df[
                            (transformed_df[col_to_check] >= (Q1 - 1.5 * IQR)) & 
                            (transformed_df[col_to_check] <= (Q3 + 1.5 * IQR))
                        ]
                        r_code_transform.append(f"filter({col_to_check} >= {Q1 - 1.5 * IQR}, {col_to_check} <= {Q3 + 1.5 * IQR})")
                        st.success(f"Removed {outliers}")
                    else:
                        transformed_df[col_to_check] = transformed_df[col_to_check].clip(
                            lower=Q1 - 1.5 * IQR,
                            upper=Q3 + 1.5 * IQR
                        )
                        r_code_transform.append(f"mutate({col_to_check} = pmin({col_to_check}, {Q3 + 1.5 * IQR}))")
                        st.success(f"Capped {outliers}")
    
    # ============== TAB 3: VISUALIZE ==============
    with tab3:
        st.markdown("<div class='section-header'>Quick Visualization</div>", unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Heatmap", "Area", "Violin", "Density"],
            key="v1"
        )
        
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns.tolist()
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            try:
                if viz_type == "Bar":
                    if categorical_cols and numeric_cols:
                        x = st.selectbox("X-axis", categorical_cols, key="vb1")
                        y = st.selectbox("Y-axis", numeric_cols, key="vb2")
                        fig = px.bar(transformed_df, x=x, y=y, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y})) +\n  geom_bar(stat = 'identity', fill = 'steelblue') +\n  labs(title = 'Bar Chart', x = '{x}', y = '{y}') +\n  theme_minimal()")
                
                elif viz_type == "Line":
                    if numeric_cols:
                        x = st.selectbox("X-axis", transformed_df.columns.tolist(), key="vl1")
                        y = st.selectbox("Y-axis", numeric_cols, key="vl2")
                        fig = px.line(transformed_df, x=x, y=y, markers=True, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y})) +\n  geom_line(color = 'steelblue', size = 1) +\n  geom_point(size = 2) +\n  labs(title = 'Line Plot', x = '{x}', y = '{y}') +\n  theme_minimal()")
                
                elif viz_type == "Scatter":
                    if len(numeric_cols) >= 2:
                        x = st.selectbox("X-axis", numeric_cols, key="vs1")
                        y = st.selectbox("Y-axis", [c for c in numeric_cols if c != x], key="vs2")
                        color = st.selectbox("Color by", ["None"] + categorical_cols, key="vs3")
                        
                        if color == "None":
                            fig = px.scatter(transformed_df, x=x, y=y, template="plotly_white", height=550)
                            r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y})) +\n  geom_point(size = 3, alpha = 0.6) +\n  labs(title = 'Scatter Plot', x = '{x}', y = '{y}') +\n  theme_minimal()")
                        else:
                            fig = px.scatter(transformed_df, x=x, y=y, color=color, template="plotly_white", height=550)
                            r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y}, color = {color})) +\n  geom_point(size = 3, alpha = 0.6) +\n  labs(title = 'Scatter Plot', x = '{x}', y = '{y}') +\n  theme_minimal()")
                        
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Histogram":
                    if numeric_cols:
                        x = st.selectbox("Column", numeric_cols, key="vh1")
                        bins = st.slider("Bins", 5, 100, 30, key="vh2")
                        fig = px.histogram(transformed_df, x=x, nbins=bins, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x})) +\n  geom_histogram(bins = {bins}, fill = 'steelblue', alpha = 0.7) +\n  labs(title = 'Histogram', x = '{x}', y = 'Frequency') +\n  theme_minimal()")
                
                elif viz_type == "Box":
                    if numeric_cols and categorical_cols:
                        y = st.selectbox("Y-axis", numeric_cols, key="vbx1")
                        x = st.selectbox("X-axis", categorical_cols, key="vbx2")
                        fig = px.box(transformed_df, x=x, y=y, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y})) +\n  geom_boxplot(fill = 'steelblue', alpha = 0.7) +\n  labs(title = 'Box Plot', x = '{x}', y = '{y}') +\n  theme_minimal()")
                
                elif viz_type == "Violin":
                    if numeric_cols and categorical_cols:
                        y = st.selectbox("Y-axis", numeric_cols, key="vv1")
                        x = st.selectbox("X-axis", categorical_cols, key="vv2")
                        fig = px.violin(transformed_df, x=x, y=y, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y})) +\n  geom_violin(fill = 'steelblue', alpha = 0.7) +\n  labs(title = 'Violin Plot', x = '{x}', y = '{y}') +\n  theme_minimal()")
                
                elif viz_type == "Pie":
                    if categorical_cols and numeric_cols:
                        names = st.selectbox("Categories", categorical_cols, key="vp1")
                        values = st.selectbox("Values", numeric_cols, key="vp2")
                        fig = px.pie(transformed_df, names=names, values=values, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = '', y = {values}, fill = {names})) +\n  geom_bar(stat = 'identity', width = 1) +\n  coord_polar('y') +\n  labs(title = 'Pie Chart') +\n  theme_void()")
                
                elif viz_type == "Area":
                    if numeric_cols:
                        x = st.selectbox("X-axis", transformed_df.columns.tolist(), key="va1")
                        y = st.selectbox("Y-axis", numeric_cols, key="va2")
                        fig = px.area(transformed_df, x=x, y=y, template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x}, y = {y})) +\n  geom_area(fill = 'steelblue', alpha = 0.5) +\n  labs(title = 'Area Chart', x = '{x}', y = '{y}') +\n  theme_minimal()")
                
                elif viz_type == "Density":
                    if numeric_cols:
                        x = st.selectbox("Column", numeric_cols, key="vd1")
                        fig = px.histogram(transformed_df, x=x, nbins=50, template="plotly_white", height=550, marginal="rug")
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"ggplot(data, aes(x = {x})) +\n  geom_density(fill = 'steelblue', alpha = 0.5) +\n  labs(title = 'Density Plot', x = '{x}') +\n  theme_minimal()")
                
                elif viz_type == "Heatmap":
                    if len(numeric_cols) >= 2:
                        corr = transformed_df[numeric_cols].corr()
                        fig = go.Figure(data=go.Heatmap(
                            z=corr.values,
                            x=corr.columns,
                            y=corr.columns,
                            colorscale='RdBu',
                            zmid=0
                        ))
                        fig.update_layout(template="plotly_white", height=550)
                        st.plotly_chart(fig, use_container_width=True)
                        r_code_viz.append(f"cor_matrix <- cor(select_if(data, is.numeric))\nheatmap(cor_matrix, main = 'Correlation Heatmap')")
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # ============== TAB 4: PLOT EDITOR ==============
    with tab4:
        st.markdown("<div class='section-header'>Advanced Plot Customization</div>", unsafe_allow_html=True)
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            numeric_cols = transformed_df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = transformed_df.select_dtypes(include=['object']).columns.tolist()
            
            # Top row: Chart type and theme
            col1, col2, col3 = st.columns(3)
            with col1:
                plot_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box", "Histogram"], key="pc1")
            with col2:
                template = st.selectbox("Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"], key="pc11")
            with col3:
                height = st.slider("Chart Height", 300, 800, 550, key="pc10")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Axis Configuration
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**X Axis**")
                x_col = st.selectbox("Column", transformed_df.columns.tolist(), key="pc2")
                x_label = st.text_input("Label", value=x_col, key="pc3")
            with col2:
                st.write("**Y Axis**")
                y_col = st.selectbox("Column", numeric_cols if numeric_cols else transformed_df.columns.tolist(), key="pc4")
                y_label = st.text_input("Label", value=y_col, key="pc5")
            with col3:
                st.write("**Title**")
                chart_title = st.text_input("Chart Title", value=f"{y_col} by {x_col}", key="pc6")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Styling Options
            st.write("**Chart Styling**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                marker_size = st.slider("Marker Size", 1, 20, 8, key="pc7")
            with col2:
                marker_color = st.color_picker("Marker Color", "#636EFA", key="pc8")
            with col3:
                opacity = st.slider("Opacity", 0.0, 1.0, 1.0, key="pc16", step=0.1)
            with col4:
                font_size = st.slider("Font Size", 10, 24, 14, key="pc9")
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Additional Options
            st.write("**Additional Options**")
            col1, col2 = st.columns(2)
            
            with col1:
                color_by = st.selectbox("Color by Column", ["None"] + categorical_cols, key="pc13")
                show_legend = st.checkbox("Show Legend", True, key="pc12")
            with col2:
                pass
            
            st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
            
            # Generate and display chart
            try:
                fig = None
                
                if plot_type == "Bar":
                    fig = px.bar(transformed_df, x=x_col, y=y_col, 
                                color=color_by if color_by != "None" else None,
                                template=template, height=height)
                
                elif plot_type == "Line":
                    fig = px.line(transformed_df, x=x_col, y=y_col, markers=True,
                                 color=color_by if color_by != "None" else None,
                                 template=template, height=height)
                
                elif plot_type == "Scatter":
                    fig = px.scatter(transformed_df, x=x_col, y=y_col,
                                    color=color_by if color_by != "None" else None,
                                    template=template, height=height)
                    fig.update_traces(marker=dict(size=marker_size, opacity=opacity, color=marker_color))
                
                elif plot_type == "Box":
                    fig = px.box(transformed_df, x=x_col, y=y_col,
                                color=color_by if color_by != "None" else None,
                                template=template, height=height)
                
                elif plot_type == "Histogram":
                    fig = px.histogram(transformed_df, x=x_col, nbins=30,
                                      color=color_by if color_by != "None" else None,
                                      template=template, height=height)
                    fig.update_traces(marker=dict(opacity=opacity))
                
                if fig:
                    fig.update_layout(
                        title=dict(text=chart_title, font=dict(size=font_size)),
                        xaxis_title=x_label,
                        yaxis_title=y_label,
                        showlegend=show_legend,
                        font=dict(size=font_size),
                        hovermode='closest'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Update R code with styling
                    r_viz_code_updated = f"""ggplot(data, aes(x = {x_col}, y = {y_col}{', color = ' + color_by if color_by != 'None' else ''})) +
  geom_{plot_type.lower() if plot_type != 'Histogram' else 'histogram'}({f'size = {marker_size/10}, alpha = {opacity}, fill = \"{marker_color}\"' if plot_type == 'Scatter' else ''}) +
  labs(
    title = '{chart_title}',
    x = '{x_label}',
    y = '{y_label}'
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = {font_size}),
    axis.text = element_text(size = {font_size - 2}),
    legend.position = {'bottom' if show_legend else 'none'}
  )"""
                    
                    st.write("**Generated R Code:**")
                    st.code(r_viz_code_updated, language="r")
            
            except Exception as e:
                st.error(f"Error generating chart: {str(e)}")
    
    # ============== TAB 5: EXPORT ==============
    with tab5:
        st.markdown("<div class='section-header'>Export Options</div>", unsafe_allow_html=True)
        
        st.subheader("R Script")
        st.markdown("Complete R/tidyverse & ggplot2 code")
        
        complete_r_code = """library(tidyverse)
library(ggplot2)

# Read data
data <- read_csv('your_data.csv')

"""
        
        if r_code_transform:
            complete_r_code += "# Data Transformation\ndata <- data %>%\n  " + (" %>%\n  ").join(r_code_transform) + "\n\n"
        
        complete_r_code += "# Display transformed data\nhead(data)\n\n"
        
        if r_code_viz:
            complete_r_code += "# Visualization\n"
            for viz in r_code_viz:
                complete_r_code += viz + "\n\n"
        
        st.code(complete_r_code, language="r")
        
        st.markdown("<div class='spacer'></div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.download_button(
                label="Download R Script",
                data=complete_r_code,
                file_name=f"data_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="Download Transformed CSV",
                data=transformed_df.to_csv(index=False),
                file_name=f"transformed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    st.info("Upload a CSV or Excel file to start")
    st.markdown("---")
    st.markdown("""
    ### Features:
    - **Transform** - Filter, Select, Sort, Mutate, Group & Summarize, Pivot
    - **Clean** - Missing values, duplicates, outliers, data types
    - **Visualize** - 10 chart types (Plotly interactive)
    - **Plot Editor** - Full chart customization with working controls
    - **Export** - Proper R/tidyverse/ggplot2 code + CSV
    """)
