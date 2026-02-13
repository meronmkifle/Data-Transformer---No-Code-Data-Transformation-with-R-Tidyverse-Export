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
    .main { background-color: #f8f9fa; }
    .sidebar { background-color: #ffffff; }
    h1, h2, h3 { color: #1f4788; }
    .stTabs [data-baseweb="tab-list"] { gap: 2px; }
    .stTabs [data-baseweb="tab"] { border-radius: 5px; padding: 10px; }
    .section-header { 
        padding: 15px; 
        background: linear-gradient(135deg, #1f4788 0%, #2d5fa3 100%);
        color: white;
        border-radius: 5px;
        margin: 20px 0 10px 0;
    }
    .transform-box {
        background: white;
        padding: 15px;
        border-left: 4px solid #1f4788;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("Data Transformer Studio")
st.markdown("*Transform • Analyze • Visualize • Export*")

with st.sidebar:
    st.header("Data Import")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        
        st.subheader("Data Types")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            st.write(f"{dtype}: {count}")

if uploaded_file:
    tab1, tab2, tab3, tab4 = st.tabs(["Transform", "Visualize", "Plot Editor", "R Code"])
    
    transformed_df = df.copy()
    r_code_lines = []
    
    with tab1:
        st.markdown("<div class='section-header'>Transformation Pipeline</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='transform-box'><b>Filter Rows</b></div>", unsafe_allow_html=True)
            filter_col = st.selectbox("Column", ["None"] + df.columns.tolist(), key="f1")
            
            if filter_col != "None":
                if df[filter_col].dtype == 'object':
                    values = st.multiselect(f"Values", df[filter_col].unique(), key="f2")
                    if values:
                        transformed_df = transformed_df[transformed_df[filter_col].isin(values)]
                        r_code_lines.append(f"filter({filter_col} %in% c({', '.join(repr(v) for v in values)}))")
                else:
                    min_val, max_val = st.slider(
                        "Range",
                        float(df[filter_col].min()),
                        float(df[filter_col].max()),
                        (float(df[filter_col].min()), float(df[filter_col].max())),
                        key="f3"
                    )
                    transformed_df = transformed_df[(transformed_df[filter_col] >= min_val) & (transformed_df[filter_col] <= max_val)]
                    r_code_lines.append(f"filter({filter_col} >= {min_val}, {filter_col} <= {max_val})")
        
        with col2:
            st.markdown("<div class='transform-box'><b>Select Columns</b></div>", unsafe_allow_html=True)
            selected_cols = st.multiselect("Columns", df.columns.tolist(), default=df.columns.tolist(), key="s1")
            if selected_cols:
                transformed_df = transformed_df[selected_cols]
                r_code_lines.append(f"select({', '.join(selected_cols)})")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.markdown("<div class='transform-box'><b>Rename Columns</b></div>", unsafe_allow_html=True)
            rename_option = st.radio("Rename?", ["No", "Yes"], key="r1")
            rename_map = {}
            if rename_option == "Yes":
                for col in transformed_df.columns:
                    new_name = st.text_input(f"{col} →", value=col, key=f"rn_{col}")
                    if new_name != col:
                        rename_map[col] = new_name
                
                if rename_map:
                    transformed_df = transformed_df.rename(columns=rename_map)
                    r_code_lines.append(f"rename({', '.join([f'{k} = {v}' for k, v in rename_map.items()])})")
        
        with col4:
            st.markdown("<div class='transform-box'><b>Remove Duplicates</b></div>", unsafe_allow_html=True)
            remove_dups = st.checkbox("Remove duplicates?", key="d1")
            if remove_dups:
                transformed_df = transformed_df.drop_duplicates()
                r_code_lines.append("distinct()")
        
        col5, col6 = st.columns(2)
        
        with col5:
            st.markdown("<div class='transform-box'><b>Sort Data</b></div>", unsafe_allow_html=True)
            sort_col = st.selectbox("Sort by", ["None"] + transformed_df.columns.tolist(), key="so1")
            if sort_col != "None":
                ascending = st.radio("Order", ["Ascending", "Descending"], key="so2") == "Ascending"
                transformed_df = transformed_df.sort_values(sort_col, ascending=ascending)
                r_code_lines.append(f"arrange({'' if ascending else 'desc('}{ sort_col}{')' if not ascending else ''})")
        
        with col6:
            st.markdown("<div class='transform-box'><b>Mutate (Create Column)</b></div>", unsafe_allow_html=True)
            mutate_option = st.radio("Create new column?", ["No", "Yes"], key="m1")
            if mutate_option == "Yes":
                new_col_name = st.text_input("New column name", key="m2")
                operation = st.selectbox("Operation", ["Custom", "Sum two cols", "Multiply", "Divide", "Subtract"], key="m3")
                
                if operation == "Custom":
                    formula = st.text_input("Formula (e.g., df['col1'] * 2)", key="m4")
                    if new_col_name and formula:
                        try:
                            transformed_df[new_col_name] = eval(formula, {"df": transformed_df})
                            r_code_lines.append(f"mutate({new_col_name} = {formula})")
                        except:
                            st.error("Invalid formula")
                elif operation == "Sum two cols":
                    col1_opt = st.selectbox("Column 1", transformed_df.columns, key="m5")
                    col2_opt = st.selectbox("Column 2", transformed_df.columns, key="m6")
                    if new_col_name:
                        transformed_df[new_col_name] = transformed_df[col1_opt] + transformed_df[col2_opt]
                        r_code_lines.append(f"mutate({new_col_name} = {col1_opt} + {col2_opt})")
                elif operation == "Multiply":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m7")
                    factor = st.number_input("Multiply by", key="m8")
                    if new_col_name:
                        transformed_df[new_col_name] = transformed_df[col_opt] * factor
                        r_code_lines.append(f"mutate({new_col_name} = {col_opt} * {factor})")
                elif operation == "Divide":
                    col_opt = st.selectbox("Column", transformed_df.columns, key="m9")
                    divisor = st.number_input("Divide by", value=1, key="m10")
                    if new_col_name and divisor != 0:
                        transformed_df[new_col_name] = transformed_df[col_opt] / divisor
                        r_code_lines.append(f"mutate({new_col_name} = {col_opt} / {divisor})")
                elif operation == "Subtract":
                    col1_opt = st.selectbox("Column 1", transformed_df.columns, key="m11")
                    col2_opt = st.selectbox("Column 2", transformed_df.columns, key="m12")
                    if new_col_name:
                        transformed_df[new_col_name] = transformed_df[col1_opt] - transformed_df[col2_opt]
                        r_code_lines.append(f"mutate({new_col_name} = {col1_opt} - {col2_opt})")
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.markdown("<div class='transform-box'><b>Group & Summarize</b></div>", unsafe_allow_html=True)
            group_cols = st.multiselect("Group by", transformed_df.columns.tolist(), key="g1")
            
            if group_cols:
                agg_col = st.selectbox("Aggregate column", transformed_df.columns.tolist(), key="g2")
                agg_func = st.selectbox("Function", ["sum", "mean", "median", "count", "min", "max", "std", "var"], key="g3")
                
                agg_map = {
                    "sum": (np.sum, "sum"),
                    "mean": (np.mean, "mean"),
                    "median": (np.median, "median"),
                    "count": (np.count_nonzero, "n"),
                    "min": (np.min, "min"),
                    "max": (np.max, "max"),
                    "std": (np.std, "sd"),
                    "var": (np.var, "var")
                }
                
                func, r_func = agg_map[agg_func]
                transformed_df = transformed_df.groupby(group_cols)[agg_col].agg(func).reset_index()
                transformed_df.columns = list(group_cols) + [f"{agg_func}_{agg_col}"]
                
                group_str = ", ".join(group_cols)
                r_code_lines.append(f"group_by({group_str})")
                r_code_lines.append(f"summarize({agg_func}_{agg_col} = {r_func}({agg_col}))")
        
        with col8:
            st.markdown("<div class='transform-box'><b>Pivot Data</b></div>", unsafe_allow_html=True)
            pivot_option = st.radio("Pivot?", ["No", "Wider", "Longer"], key="p1")
            
            if pivot_option == "Wider":
                idx_cols = st.multiselect("Index columns", transformed_df.columns.tolist(), key="pw1")
                col_col = st.selectbox("Column names from", transformed_df.columns.tolist(), key="pw2")
                val_col = st.selectbox("Values from", transformed_df.columns.tolist(), key="pw3")
                
                if idx_cols and col_col and val_col:
                    transformed_df = transformed_df.pivot_table(
                        index=idx_cols,
                        columns=col_col,
                        values=val_col,
                        aggfunc='first'
                    ).reset_index()
                    r_code_lines.append(f"pivot_wider(names_from = {col_col}, values_from = {val_col})")
            
            elif pivot_option == "Longer":
                id_cols = st.multiselect("ID columns (keep as is)", transformed_df.columns.tolist(), key="pl1")
                value_cols = st.multiselect("Columns to pivot", [c for c in transformed_df.columns if c not in id_cols], key="pl2")
                
                if value_cols:
                    names_to = st.text_input("Name for column names", value="variable", key="pl3")
                    values_to = st.text_input("Name for values", value="value", key="pl4")
                    
                    if id_cols:
                        transformed_df = transformed_df.melt(id_vars=id_cols, value_vars=value_cols, 
                                                             var_name=names_to, value_name=values_to)
                    else:
                        transformed_df = transformed_df.melt(value_vars=value_cols, 
                                                             var_name=names_to, value_name=values_to)
                    r_code_lines.append(f"pivot_longer(cols = {value_cols}, names_to = '{names_to}', values_to = '{values_to}')")
        
        st.divider()
        st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
        st.dataframe(transformed_df, use_container_width=True, height=400)
        st.write(f"Shape: {transformed_df.shape[0]} rows × {transformed_df.shape[1]} columns")
    
    with tab2:
        st.markdown("<div class='section-header'>Quick Visualization</div>", unsafe_allow_html=True)
        
        viz_type = st.selectbox(
            "Chart Type",
            ["Bar", "Line", "Scatter", "Histogram", "Box", "Pie", "Heatmap", "Area", "Violin"],
            key="v1"
        )
        
        numeric_cols = transformed_df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = transformed_df.select_dtypes(include=['object']).columns.tolist()
        
        if not HAS_PLOTLY:
            st.error("Plotly not installed. Add 'plotly>=5.0.0' to requirements.txt")
        else:
            try:
                if viz_type == "Bar":
                    if categorical_cols and numeric_cols:
                        x = st.selectbox("X-axis", categorical_cols, key="vb1")
                        y = st.selectbox("Y-axis", numeric_cols, key="vb2")
                        fig = px.bar(transformed_df, x=x, y=y, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Line":
                    if numeric_cols:
                        x = st.selectbox("X-axis", transformed_df.columns.tolist(), key="vl1")
                        y = st.selectbox("Y-axis", numeric_cols, key="vl2")
                        fig = px.line(transformed_df, x=x, y=y, markers=True, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Scatter":
                    if len(numeric_cols) >= 2:
                        x = st.selectbox("X-axis", numeric_cols, key="vs1")
                        y = st.selectbox("Y-axis", [c for c in numeric_cols if c != x], key="vs2")
                        color = st.selectbox("Color", ["None"] + categorical_cols, key="vs3")
                        if color == "None":
                            fig = px.scatter(transformed_df, x=x, y=y, template="plotly_white")
                        else:
                            fig = px.scatter(transformed_df, x=x, y=y, color=color, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Histogram":
                    if numeric_cols:
                        x = st.selectbox("Column", numeric_cols, key="vh1")
                        bins = st.slider("Bins", 5, 100, 30, key="vh2")
                        fig = px.histogram(transformed_df, x=x, nbins=bins, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Box":
                    if numeric_cols and categorical_cols:
                        y = st.selectbox("Y-axis", numeric_cols, key="vbx1")
                        x = st.selectbox("X-axis", categorical_cols, key="vbx2")
                        fig = px.box(transformed_df, x=x, y=y, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Pie":
                    if categorical_cols and numeric_cols:
                        names = st.selectbox("Categories", categorical_cols, key="vp1")
                        values = st.selectbox("Values", numeric_cols, key="vp2")
                        fig = px.pie(transformed_df, names=names, values=values, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Heatmap":
                    if len(numeric_cols) >= 2:
                        corr = transformed_df[numeric_cols].corr()
                        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
                        fig.update_layout(template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Area":
                    if numeric_cols:
                        x = st.selectbox("X-axis", transformed_df.columns.tolist(), key="va1")
                        y = st.selectbox("Y-axis", numeric_cols, key="va2")
                        fig = px.area(transformed_df, x=x, y=y, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
                
                elif viz_type == "Violin":
                    if numeric_cols and categorical_cols:
                        y = st.selectbox("Y-axis", numeric_cols, key="vv1")
                        x = st.selectbox("X-axis", categorical_cols, key="vv2")
                        fig = px.violin(transformed_df, x=x, y=y, template="plotly_white")
                        st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab3:
        st.markdown("<div class='section-header'>Advanced Plot Customization</div>", unsafe_allow_html=True)
        
        if not HAS_PLOTLY:
            st.error("Plotly required")
        else:
            plot_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Box"], key="pc1")
            
            numeric_cols = transformed_df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = transformed_df.select_dtypes(include=['object']).columns.tolist()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_col = st.selectbox("X-axis column", transformed_df.columns.tolist(), key="pc2")
                x_label = st.text_input("X-axis label", value=x_col, key="pc3")
            
            with col2:
                y_col = st.selectbox("Y-axis column", numeric_cols, key="pc4")
                y_label = st.text_input("Y-axis label", value=y_col, key="pc5")
            
            with col3:
                chart_title = st.text_input("Chart title", value=f"{y_col} by {x_col}", key="pc6")
            
            col4, col5, col6 = st.columns(3)
            
            with col4:
                marker_size = st.slider("Marker size", 1, 20, 8, key="pc7")
                marker_color = st.color_picker("Marker color", "#636EFA", key="pc8")
            
            with col5:
                font_size = st.slider("Font size", 10, 24, 14, key="pc9")
                height = st.slider("Chart height", 300, 800, 500, key="pc10")
            
            with col6:
                template = st.selectbox("Theme", ["plotly_white", "plotly_dark", "ggplot2", "seaborn"], key="pc11")
                show_legend = st.checkbox("Show legend", True, key="pc12")
            
            col7, col8 = st.columns(2)
            
            with col7:
                color_by = st.selectbox("Color by column", ["None"] + categorical_cols, key="pc13")
                facet_col = st.selectbox("Facet by", ["None"] + categorical_cols, key="pc14")
            
            with col8:
                line_style = st.selectbox("Line style", ["solid", "dash", "dot"], key="pc15")
                opacity = st.slider("Opacity", 0.0, 1.0, 1.0, key="pc16")
            
            try:
                if plot_type == "Bar":
                    fig = px.bar(transformed_df, x=x_col, y=y_col, color=color_by if color_by != "None" else None,
                                facet_col=facet_col if facet_col != "None" else None,
                                template=template, height=height)
                
                elif plot_type == "Line":
                    fig = px.line(transformed_df, x=x_col, y=y_col, markers=True,
                                 color=color_by if color_by != "None" else None,
                                 facet_col=facet_col if facet_col != "None" else None,
                                 template=template, height=height)
                
                elif plot_type == "Scatter":
                    fig = px.scatter(transformed_df, x=x_col, y=y_col,
                                    color=color_by if color_by != "None" else None,
                                    facet_col=facet_col if facet_col != "None" else None,
                                    template=template, height=height)
                    fig.update_traces(marker=dict(size=marker_size, color=marker_color, opacity=opacity))
                
                elif plot_type == "Box":
                    fig = px.box(transformed_df, x=x_col, y=y_col,
                                color=color_by if color_by != "None" else None,
                                facet_col=facet_col if facet_col != "None" else None,
                                template=template, height=height)
                
                fig.update_layout(
                    title=dict(text=chart_title, font=dict(size=font_size)),
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    showlegend=show_legend,
                    font=dict(size=font_size),
                    hovermode='closest'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with tab4:
        st.markdown("<div class='section-header'>Generated R Code</div>", unsafe_allow_html=True)
        
        if r_code_lines:
            r_code = f"""library(tidyverse)

data <- read_csv("your_data.csv") %>%
  {(" %>%" + chr(10) + "  ").join(r_code_lines)}

head(data)
"""
        else:
            r_code = """library(tidyverse)

data <- read_csv("your_data.csv")

head(data)
"""
        
        st.code(r_code, language="r")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="Download R Script",
                data=r_code,
                file_name=f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
                mime="text/plain"
            )
        
        with col2:
            st.download_button(
                label="Download CSV",
                data=transformed_df.to_csv(index=False),
                file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col3:
            st.download_button(
                label="Download JSON",
                data=transformed_df.to_json(orient='records'),
                file_name=f"transformed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="text/json"
            )

else:
    st.info("Upload CSV or Excel file to start")
