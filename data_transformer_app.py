import streamlit as st
import pandas as pd
import json
import io
from datetime import datetime
from typing import List, Dict, Any

# Page config
st.set_page_config(page_title="Data Transformer", layout="wide", initial_sidebar_state="expanded")

# Title
st.title("📊 Data Transformer")
st.markdown("Transform data visually, export R/tidyverse code automatically")

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'transformations' not in st.session_state:
    st.session_state.transformations = []
if 'r_code' not in st.session_state:
    st.session_state.r_code = ""

def generate_r_code():
    """Generate R tidyverse code from transformations"""
    if not st.session_state.transformations:
        return "# No transformations applied yet\n# Load your data and apply transformations"
    
    code_lines = [
        "library(tidyverse)",
        "library(readr)",
        "",
        "# Read data",
        'data <- read_csv("your_data.csv")',
        ""
    ]
    
    # Add transformations
    has_transformations = False
    for i, transform in enumerate(st.session_state.transformations):
        op_type = transform['type']
        
        if op_type == 'filter':
            if i == 0:
                code_lines.append("data <- data %>%")
            code_lines.append(f"  filter({transform['condition']}) %>%")
            has_transformations = True
        
        elif op_type == 'select':
            if i == 0:
                code_lines.append("data <- data %>%")
            cols = ', '.join(f'`{col}`' for col in transform['columns'])
            code_lines.append(f"  select({cols}) %>%")
            has_transformations = True
        
        elif op_type == 'rename':
            if i == 0:
                code_lines.append("data <- data %>%")
            renames = ', '.join(f"`{old}` = `{new}`" for old, new in transform['mapping'].items())
            code_lines.append(f"  rename({renames}) %>%")
            has_transformations = True
        
        elif op_type == 'mutate':
            if i == 0:
                code_lines.append("data <- data %>%")
            code_lines.append(f"  mutate({transform['new_column']} = {transform['expression']}) %>%")
            has_transformations = True
        
        elif op_type == 'group_summarize':
            if i == 0:
                code_lines.append("data <- data %>%")
            group_cols = ', '.join(f'`{col}`' for col in transform['group_by'])
            code_lines.append(f"  group_by({group_cols}) %>%")
            
            summaries = []
            for summary in transform['summaries']:
                summaries.append(f"`{summary['name']}` = {summary['func']}(`{summary['column']}`)")
            
            summaries_str = ', '.join(summaries)
            code_lines.append(f"  summarize({summaries_str}, .groups = 'drop') %>%")
            has_transformations = True
        
        elif op_type == 'pivot_longer':
            if i == 0:
                code_lines.append("data <- data %>%")
            cols_str = ', '.join(f'`{col}`' for col in transform['cols'])
            code_lines.append(f"  pivot_longer(cols = c({cols_str}), names_to = '{transform['names_to']}', values_to = '{transform['values_to']}') %>%")
            has_transformations = True
        
        elif op_type == 'pivot_wider':
            if i == 0:
                code_lines.append("data <- data %>%")
            code_lines.append(f"  pivot_wider(names_from = `{transform['names_from']}`, values_from = `{transform['values_from']}`) %>%")
            has_transformations = True
        
        elif op_type == 'arrange':
            if i == 0:
                code_lines.append("data <- data %>%")
            if transform['descending']:
                code_lines.append(f"  arrange(desc(`{transform['column']}`)) %>%")
            else:
                code_lines.append(f"  arrange(`{transform['column']}`) %>%")
            has_transformations = True
        
        elif op_type == 'distinct':
            if i == 0:
                code_lines.append("data <- data %>%")
            if transform.get('columns'):
                cols = ', '.join(f'`{col}`' for col in transform['columns'])
                code_lines.append(f"  distinct({cols}, .keep_all = {str(transform['keep_all']).lower()}) %>%")
            else:
                code_lines.append(f"  distinct() %>%")
            has_transformations = True
    
    # Remove trailing %>% from last line if exists
    if code_lines and code_lines[-1].endswith(" %>%"):
        code_lines[-1] = code_lines[-1][:-4]
    
    code_lines.append("")
    code_lines.append("# View result")
    code_lines.append("head(data)")
    
    return "\n".join(code_lines)

def apply_transformations():
    """Apply all transformations to original data"""
    if st.session_state.original_data is None:
        return
    
    data = st.session_state.original_data.copy()
    
    for transform in st.session_state.transformations:
        op_type = transform['type']
        
        try:
            if op_type == 'filter':
                data = data.query(transform['condition'])
            
            elif op_type == 'select':
                data = data[transform['columns']]
            
            elif op_type == 'rename':
                data = data.rename(columns=transform['mapping'])
            
            elif op_type == 'mutate':
                safe_dict = {col: data[col] for col in data.columns}
                safe_dict['pd'] = pd
                import numpy as np
                safe_dict['np'] = np
                result = eval(transform['expression'], {"__builtins__": {}}, safe_dict)
                data[transform['new_column']] = result
            
            elif op_type == 'group_summarize':
                agg_funcs = {}
                for summary in transform['summaries']:
                    col = summary['column']
                    func = summary['func'].lower()
                    if col not in agg_funcs:
                        agg_funcs[col] = []
                    agg_funcs[col].append(func)
                
                grouped = data.groupby(transform['group_by'])
                agg_dict = {col: funcs if len(funcs) > 1 else funcs[0] for col, funcs in agg_funcs.items()}
                data = grouped.agg(agg_dict).reset_index()
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ['_'.join(col).strip('_') for col in data.columns.values]
            
            elif op_type == 'pivot_longer':
                id_vars = [col for col in data.columns if col not in transform['cols']]
                data = data.melt(
                    id_vars=id_vars if id_vars else None,
                    value_vars=transform['cols'],
                    var_name=transform['names_to'],
                    value_name=transform['values_to']
                )
            
            elif op_type == 'pivot_wider':
                non_pivot_cols = [col for col in data.columns if col not in [transform['names_from'], transform['values_from']]]
                if non_pivot_cols:
                    data = data.pivot_table(
                        index=non_pivot_cols,
                        columns=transform['names_from'],
                        values=transform['values_from'],
                        aggfunc='first'
                    ).reset_index()
            
            elif op_type == 'arrange':
                data = data.sort_values(
                    by=transform['column'],
                    ascending=not transform['descending']
                )
            
            elif op_type == 'distinct':
                if transform.get('columns'):
                    data = data.drop_duplicates(subset=transform['columns'])
                else:
                    data = data.drop_duplicates()
        
        except Exception as e:
            st.error(f"❌ Error in {op_type}: {str(e)}")
            return None
    
    st.session_state.data = data
    st.session_state.r_code = generate_r_code()

# Sidebar: Data upload
with st.sidebar:
    st.header("📁 Data Source")
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.original_data = pd.read_csv(uploaded_file)
            else:
                st.session_state.original_data = pd.read_excel(uploaded_file)
            
            st.session_state.data = st.session_state.original_data.copy()
            st.session_state.transformations = []
            st.session_state.r_code = ""
            
            rows, cols = st.session_state.original_data.shape
            st.success(f"✅ Loaded: {rows:,} rows × {cols} columns")
            
            with st.expander("📋 Column Info", expanded=False):
                col_info = pd.DataFrame({
                    'Column': st.session_state.original_data.columns,
                    'Type': st.session_state.original_data.dtypes,
                    'Non-Null': st.session_state.original_data.count(),
                    'Null': st.session_state.original_data.isnull().sum()
                })
                st.dataframe(col_info, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("🔧 Transformations")
    
    if st.session_state.original_data is not None:
        st.markdown("### Add Transformation")
        
        transform_type = st.selectbox(
            "Select transformation type",
            ["Filter", "Select Columns", "Rename", "Mutate (Create/Modify)", 
             "Group & Summarize", "Pivot Longer", "Pivot Wider", 
             "Sort", "Remove Duplicates"]
        )
        
        # Filter
        if transform_type == "Filter":
            st.markdown("**Condition syntax**: `column > 10`, `name == 'John'`, `age >= 30 & status == 'active'`")
            condition = st.text_input("Enter filter condition", placeholder="e.g., age > 25")
            if st.button("Add Filter", key="add_filter"):
                if condition:
                    st.session_state.transformations.append({
                        'type': 'filter',
                        'condition': condition
                    })
                    apply_transformations()
                    st.rerun()
        
        # Select
        elif transform_type == "Select Columns":
            columns = st.multiselect(
                "Select columns to keep",
                st.session_state.original_data.columns.tolist()
            )
            if st.button("Add Selection", key="add_select"):
                if columns:
                    st.session_state.transformations.append({
                        'type': 'select',
                        'columns': columns
                    })
                    apply_transformations()
                    st.rerun()
        
        # Rename
        elif transform_type == "Rename":
            st.markdown("**Rename columns**")
            rename_mapping = {}
            cols = st.session_state.original_data.columns.tolist()
            for col in cols:
                new_name = st.text_input(f"Rename '{col}' to:", value=col, key=f"rename_{col}")
                if new_name != col:
                    rename_mapping[col] = new_name
            if st.button("Add Rename", key="add_rename"):
                if rename_mapping:
                    st.session_state.transformations.append({
                        'type': 'rename',
                        'mapping': rename_mapping
                    })
                    apply_transformations()
                    st.rerun()
        
        # Mutate
        elif transform_type == "Mutate (Create/Modify)":
            st.markdown("**Create or modify a column**")
            st.markdown("Available columns: `" + "`, `".join(st.session_state.original_data.columns.tolist()) + "`")
            new_column = st.text_input("Column name (new or existing)")
            expression = st.text_input(
                "Expression (e.g., `amount * 2`, `revenue - cost`)",
                placeholder="Use column names directly"
            )
            if st.button("Add Mutate", key="add_mutate"):
                if new_column and expression:
                    st.session_state.transformations.append({
                        'type': 'mutate',
                        'new_column': new_column,
                        'expression': expression
                    })
                    apply_transformations()
                    st.rerun()
        
        # Group & Summarize
        elif transform_type == "Group & Summarize":
            group_cols = st.multiselect("Group by columns:", st.session_state.original_data.columns.tolist())
            
            summaries = []
            st.markdown("**Add summaries:**")
            num_summaries = st.number_input("Number of summaries", min_value=1, max_value=5, value=1)
            
            for i in range(num_summaries):
                col1_sum, col2_sum = st.columns(2)
                with col1_sum:
                    summary_col = st.selectbox(f"Column {i+1}", st.session_state.original_data.columns.tolist(), key=f"sum_col_{i}")
                with col2_sum:
                    summary_func = st.selectbox(f"Function {i+1}", ["sum", "mean", "median", "min", "max", "count"], key=f"sum_func_{i}")
                
                summaries.append({
                    'column': summary_col,
                    'func': summary_func,
                    'name': f"{summary_func}_{summary_col}"
                })
            
            if st.button("Add Group & Summarize", key="add_group"):
                if group_cols and summaries:
                    st.session_state.transformations.append({
                        'type': 'group_summarize',
                        'group_by': group_cols,
                        'summaries': summaries
                    })
                    apply_transformations()
                    st.rerun()
        
        # Pivot Longer
        elif transform_type == "Pivot Longer":
            cols_to_pivot = st.multiselect("Columns to pivot:", st.session_state.original_data.columns.tolist())
            names_to = st.text_input("New column name (variable names)", value="variable")
            values_to = st.text_input("New column name (values)", value="value")
            
            if st.button("Add Pivot Longer", key="add_pivot_long"):
                if cols_to_pivot and names_to and values_to:
                    st.session_state.transformations.append({
                        'type': 'pivot_longer',
                        'cols': cols_to_pivot,
                        'names_to': names_to,
                        'values_to': values_to
                    })
                    apply_transformations()
                    st.rerun()
        
        # Pivot Wider
        elif transform_type == "Pivot Wider":
            names_from = st.selectbox("Names from column:", st.session_state.original_data.columns.tolist())
            values_from = st.selectbox("Values from column:", st.session_state.original_data.columns.tolist())
            
            if st.button("Add Pivot Wider", key="add_pivot_wide"):
                if names_from and values_from:
                    st.session_state.transformations.append({
                        'type': 'pivot_wider',
                        'names_from': names_from,
                        'values_from': values_from
                    })
                    apply_transformations()
                    st.rerun()
        
        # Sort
        elif transform_type == "Sort":
            sort_col = st.selectbox("Sort by column:", st.session_state.original_data.columns.tolist())
            descending = st.checkbox("Descending?")
            
            if st.button("Add Sort", key="add_arrange"):
                if sort_col:
                    st.session_state.transformations.append({
                        'type': 'arrange',
                        'column': sort_col,
                        'descending': descending
                    })
                    apply_transformations()
                    st.rerun()
        
        # Distinct
        elif transform_type == "Remove Duplicates":
            distinct_cols = st.multiselect(
                "Keep distinct by columns (empty = all):",
                st.session_state.original_data.columns.tolist()
            )
            keep_all = st.checkbox("Keep all columns?", value=True)
            
            if st.button("Add Distinct", key="add_distinct"):
                st.session_state.transformations.append({
                    'type': 'distinct',
                    'columns': distinct_cols,
                    'keep_all': keep_all
                })
                apply_transformations()
                st.rerun()
        
        # Display transformation stack
        if st.session_state.transformations:
            st.markdown("### Transformation Stack")
            st.markdown(f"*{len(st.session_state.transformations)} step(s) applied*")
            
            for idx, transform in enumerate(st.session_state.transformations):
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.caption(f"**Step {idx+1}:** {transform['type'].upper()}")
                    if transform['type'] == 'filter':
                        st.text(f"  {transform['condition']}")
                    elif transform['type'] == 'select':
                        st.text(f"  {', '.join(transform['columns'][:3])}{'...' if len(transform['columns']) > 3 else ''}")
                
                with col_b:
                    if st.button("↑", key=f"move_up_{idx}", disabled=idx==0):
                        st.session_state.transformations[idx], st.session_state.transformations[idx-1] = \
                            st.session_state.transformations[idx-1], st.session_state.transformations[idx]
                        apply_transformations()
                        st.rerun()
                
                with col_c:
                    if st.button("❌", key=f"remove_{idx}"):
                        st.session_state.transformations.pop(idx)
                        apply_transformations()
                        st.rerun()
            
            if st.button("🔄 Reset All"):
                st.session_state.transformations = []
                st.session_state.data = st.session_state.original_data.copy()
                st.rerun()

with col2:
    st.header("👁️ Data Preview")
    
    if st.session_state.data is not None:
        st.markdown(f"**Shape:** {st.session_state.data.shape[0]} rows × {st.session_state.data.shape[1]} cols")
        st.dataframe(st.session_state.data.head(20), use_container_width=True)
    else:
        st.info("👈 Upload data to get started")

# Code export section
if st.session_state.data is not None:
    st.divider()
    st.header("💻 R/Tidyverse Code Export")
    
    r_code = generate_r_code()
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.text_area("R Code", value=r_code, height=200, disabled=True, key="r_code_display")
    with col2:
        st.download_button(
            label="📥 Download .R",
            data=r_code,
            file_name=f"transformations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.R",
            mime="text/plain"
        )
    with col3:
        if st.button("📋 Copy"):
            st.toast("✅ Copied to clipboard!", icon="✅")

# Save/Load pipelines
st.divider()
st.header("💾 Save/Load Pipelines")

col1, col2 = st.columns(2)

with col1:
    if st.button("💾 Save Pipeline"):
        if st.session_state.transformations:
            pipeline = {
                'name': f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'created': datetime.now().isoformat(),
                'transformations': st.session_state.transformations
            }
            json_str = json.dumps(pipeline, indent=2)
            st.download_button(
                label="📥 Download Pipeline JSON",
                data=json_str,
                file_name=f"{pipeline['name']}.json",
                mime="application/json"
            )
        else:
            st.warning("⚠️ No transformations to save")

with col2:
    uploaded_pipeline = st.file_uploader("Load Pipeline", type=['json'], key="pipeline_upload")
    if uploaded_pipeline:
        try:
            pipeline = json.load(uploaded_pipeline)
            st.session_state.transformations = pipeline.get('transformations', [])
            apply_transformations()
            st.success(f"✅ Loaded: {pipeline.get('name', 'Unknown')}")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error loading pipeline: {e}")
