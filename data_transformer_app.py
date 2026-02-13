import streamlit as st
import pandas as pd
import json
import numpy as np
from datetime import datetime

# Page config
st.set_page_config(page_title="Data Transformer", layout="wide")

st.title("📊 Data Transformer")
st.markdown("Transform data visually → Export R/tidyverse code")

# Initialize session state
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'transformations' not in st.session_state:
    st.session_state.transformations = []

def apply_all_transformations(data, transformations):
    """Apply all transformations in order and return modified data"""
    if data is None:
        return None
    
    result_data = data.copy()
    
    for transform in transformations:
        try:
            t_type = transform['type']
            
            if t_type == 'filter':
                result_data = result_data.query(transform['condition'])
                st.success(f"✓ Filter: {transform['condition']}")
            
            elif t_type == 'select':
                result_data = result_data[transform['columns']]
                st.success(f"✓ Select: {len(transform['columns'])} columns")
            
            elif t_type == 'rename':
                result_data = result_data.rename(columns=transform['mapping'])
                st.success(f"✓ Renamed {len(transform['mapping'])} columns")
            
            elif t_type == 'mutate':
                safe_dict = {col: result_data[col] for col in result_data.columns}
                safe_dict['pd'] = pd
                safe_dict['np'] = np
                expr = transform['expression']
                result = eval(expr, {"__builtins__": {}}, safe_dict)
                result_data[transform['new_column']] = result
                st.success(f"✓ Mutate: {transform['new_column']} = {expr}")
            
            elif t_type == 'group_summarize':
                group_cols = transform['group_by']
                summaries = transform['summaries']
                
                agg_dict = {}
                for s in summaries:
                    col = s['column']
                    func = s['func'].lower()
                    if col not in agg_dict:
                        agg_dict[col] = []
                    agg_dict[col].append(func)
                
                agg_dict = {k: (v[0] if len(v) == 1 else v) for k, v in agg_dict.items()}
                result_data = result_data.groupby(group_cols).agg(agg_dict).reset_index()
                
                if isinstance(result_data.columns, pd.MultiIndex):
                    result_data.columns = ['_'.join(col).strip('_') for col in result_data.columns.values]
                
                st.success(f"✓ Group by: {group_cols}")
            
            elif t_type == 'pivot_longer':
                id_vars = [c for c in result_data.columns if c not in transform['cols']]
                result_data = result_data.melt(
                    id_vars=id_vars if id_vars else None,
                    value_vars=transform['cols'],
                    var_name=transform['names_to'],
                    value_name=transform['values_to']
                )
                st.success(f"✓ Pivot Longer")
            
            elif t_type == 'pivot_wider':
                index_cols = [c for c in result_data.columns if c not in [transform['names_from'], transform['values_from']]]
                result_data = result_data.pivot_table(
                    index=index_cols if index_cols else None,
                    columns=transform['names_from'],
                    values=transform['values_from'],
                    aggfunc='first'
                ).reset_index()
                st.success(f"✓ Pivot Wider")
            
            elif t_type == 'sort':
                result_data = result_data.sort_values(
                    by=transform['column'],
                    ascending=not transform.get('descending', False)
                )
                st.success(f"✓ Sort by {transform['column']}")
            
            elif t_type == 'distinct':
                if transform.get('columns'):
                    result_data = result_data.drop_duplicates(subset=transform['columns'])
                else:
                    result_data = result_data.drop_duplicates()
                st.success(f"✓ Removed duplicates")
        
        except Exception as e:
            st.error(f"❌ Error in {t_type}: {str(e)}")
            return None
    
    return result_data

def generate_r_code(transformations):
    """Generate R code from transformations"""
    if not transformations:
        return "# No transformations yet"
    
    code = [
        "library(tidyverse)",
        "library(readr)",
        "",
        "data <- read_csv('your_data.csv') %>%"
    ]
    
    for t in transformations:
        t_type = t['type']
        
        if t_type == 'filter':
            code.append(f"  filter({t['condition']}) %>%")
        elif t_type == 'select':
            cols = ", ".join(f"`{c}`" for c in t['columns'])
            code.append(f"  select({cols}) %>%")
        elif t_type == 'rename':
            renames = ", ".join(f"`{k}` = `{v}`" for k, v in t['mapping'].items())
            code.append(f"  rename({renames}) %>%")
        elif t_type == 'mutate':
            code.append(f"  mutate({t['new_column']} = {t['expression']}) %>%")
        elif t_type == 'group_summarize':
            group = ", ".join(f"`{c}`" for c in t['group_by'])
            sums = ", ".join(f"`{s['name']}` = {s['func']}(`{s['column']}`)" for s in t['summaries'])
            code.append(f"  group_by({group}) %>%")
            code.append(f"  summarize({sums}, .groups = 'drop') %>%")
        elif t_type == 'sort':
            col = t['column']
            if t.get('descending'):
                code.append(f"  arrange(desc(`{col}`)) %>%")
            else:
                code.append(f"  arrange(`{col}`) %>%")
        elif t_type == 'distinct':
            code.append(f"  distinct() %>%")
    
    # Remove last %>%
    code[-1] = code[-1].replace(" %>%", "")
    code.append("")
    code.append("head(data)")
    
    return "\n".join(code)

# ============= UPLOAD DATA =============
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            st.session_state.original_data = pd.read_csv(uploaded_file)
        else:
            st.session_state.original_data = pd.read_excel(uploaded_file)
        
        st.session_state.transformations = []
        
        rows, cols = st.session_state.original_data.shape
        st.sidebar.success(f"✅ Loaded: {rows} rows × {cols} cols")
        
        with st.sidebar.expander("📋 Columns"):
            st.dataframe(pd.DataFrame({
                'Column': st.session_state.original_data.columns,
                'Type': st.session_state.original_data.dtypes
            }), use_container_width=True)
    
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# ============= MAIN APP =============
if st.session_state.original_data is not None:
    col1, col2 = st.columns([1, 1])
    
    # LEFT: Transformations
    with col1:
        st.header("🔧 Add Transformation")
        
        t_type = st.selectbox("Type", [
            "Filter", "Select Columns", "Rename", "Mutate",
            "Group & Summarize", "Pivot Longer", "Pivot Wider", "Sort", "Distinct"
        ])
        
        if t_type == "Filter":
            condition = st.text_input("Condition (e.g., year == 2024)")
            if st.button("Add Filter"):
                if condition:
                    st.session_state.transformations.append({
                        'type': 'filter',
                        'condition': condition
                    })
                    st.rerun()
        
        elif t_type == "Select Columns":
            cols = st.multiselect("Choose columns", st.session_state.original_data.columns)
            if st.button("Add Select"):
                if cols:
                    st.session_state.transformations.append({
                        'type': 'select',
                        'columns': cols
                    })
                    st.rerun()
        
        elif t_type == "Rename":
            st.write("Rename columns:")
            mapping = {}
            for col in st.session_state.original_data.columns:
                new_name = st.text_input(f"{col} →", value=col, key=f"rename_{col}")
                if new_name != col:
                    mapping[col] = new_name
            if st.button("Add Rename"):
                if mapping:
                    st.session_state.transformations.append({
                        'type': 'rename',
                        'mapping': mapping
                    })
                    st.rerun()
        
        elif t_type == "Mutate":
            new_col = st.text_input("New column name")
            expr = st.text_input("Expression (e.g., amount * 0.15)")
            if st.button("Add Mutate"):
                if new_col and expr:
                    st.session_state.transformations.append({
                        'type': 'mutate',
                        'new_column': new_col,
                        'expression': expr
                    })
                    st.rerun()
        
        elif t_type == "Group & Summarize":
            group_by = st.multiselect("Group by", st.session_state.original_data.columns)
            n = st.number_input("Number of summaries", 1, 5, 1)
            summaries = []
            for i in range(n):
                col = st.selectbox(f"Column {i+1}", st.session_state.original_data.columns, key=f"sum_col_{i}")
                func = st.selectbox(f"Function {i+1}", ["sum", "mean", "count", "min", "max"], key=f"sum_func_{i}")
                summaries.append({'column': col, 'func': func, 'name': f'{func}_{col}'})
            
            if st.button("Add Group & Summarize"):
                if group_by and summaries:
                    st.session_state.transformations.append({
                        'type': 'group_summarize',
                        'group_by': group_by,
                        'summaries': summaries
                    })
                    st.rerun()
        
        elif t_type == "Pivot Longer":
            cols = st.multiselect("Columns to pivot", st.session_state.original_data.columns)
            names_to = st.text_input("Names to", "variable")
            values_to = st.text_input("Values to", "value")
            if st.button("Add Pivot Longer"):
                if cols:
                    st.session_state.transformations.append({
                        'type': 'pivot_longer',
                        'cols': cols,
                        'names_to': names_to,
                        'values_to': values_to
                    })
                    st.rerun()
        
        elif t_type == "Pivot Wider":
            names_from = st.selectbox("Names from", st.session_state.original_data.columns)
            values_from = st.selectbox("Values from", st.session_state.original_data.columns)
            if st.button("Add Pivot Wider"):
                st.session_state.transformations.append({
                    'type': 'pivot_wider',
                    'names_from': names_from,
                    'values_from': values_from
                })
                st.rerun()
        
        elif t_type == "Sort":
            sort_col = st.selectbox("Sort by", st.session_state.original_data.columns)
            descending = st.checkbox("Descending")
            if st.button("Add Sort"):
                st.session_state.transformations.append({
                    'type': 'sort',
                    'column': sort_col,
                    'descending': descending
                })
                st.rerun()
        
        elif t_type == "Distinct":
            if st.button("Add Distinct"):
                st.session_state.transformations.append({
                    'type': 'distinct',
                    'columns': []
                })
                st.rerun()
        
        # Show transformation stack
        if st.session_state.transformations:
            st.divider()
            st.write(f"### Steps: {len(st.session_state.transformations)}")
            for idx, t in enumerate(st.session_state.transformations):
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    if t['type'] == 'filter':
                        st.write(f"**{idx+1}. Filter:** {t['condition']}")
                    elif t['type'] == 'select':
                        st.write(f"**{idx+1}. Select:** {len(t['columns'])} columns")
                    else:
                        st.write(f"**{idx+1}. {t['type'].title()}**")
                with col_b:
                    if st.button("❌", key=f"del_{idx}"):
                        st.session_state.transformations.pop(idx)
                        st.rerun()
            
            if st.button("🔄 Clear All"):
                st.session_state.transformations = []
                st.rerun()
    
    # RIGHT: Preview
    with col2:
        st.header("👁️ Data Preview")
        
        # Apply transformations
        transformed_data = apply_all_transformations(
            st.session_state.original_data,
            st.session_state.transformations
        )
        
        if transformed_data is not None:
            st.write(f"**Shape:** {transformed_data.shape[0]} rows × {transformed_data.shape[1]} cols")
            st.dataframe(transformed_data.head(20), use_container_width=True)
        else:
            st.info("No data")
    
    # CODE EXPORT
    st.divider()
    st.header("💻 R Code Export")
    
    r_code = generate_r_code(st.session_state.transformations)
    
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.code(r_code, language="r")
    with col_b:
        st.download_button(
            "📥 Download .R",
            r_code,
            f"transform_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
            "text/plain"
        )

else:
    st.info("👈 Upload data in sidebar to start")
