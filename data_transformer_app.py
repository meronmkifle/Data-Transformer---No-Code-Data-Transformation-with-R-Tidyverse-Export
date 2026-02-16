import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import json
import io

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

st.set_page_config(page_title="Data Transformer Studio Pro", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main { background-color: #f8f9fb; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    h1, h2, h3 { color: #1a1f36; }
    .hero-banner {
        background: linear-gradient(135deg, #1a1f36 0%, #2d3a6d 40%, #4a6cf7 100%);
        color: white; padding: 28px 32px; border-radius: 12px;
        margin-bottom: 24px; box-shadow: 0 4px 20px rgba(74,108,247,0.25);
    }
    .hero-banner h1 { color: white; margin: 0 0 6px 0; font-size: 1.8em; font-weight: 700; }
    .hero-banner p { color: #c7cde6; margin: 0; font-size: 0.95em; }
    .section-header {
        padding: 14px 20px; border-radius: 8px; margin: 24px 0 16px 0;
        font-size: 1.15em; font-weight: 600;
        background: linear-gradient(135deg, #1a1f36 0%, #2d3a6d 100%);
        color: white; box-shadow: 0 2px 8px rgba(26,31,54,0.15);
    }
    .pipeline-step {
        background: #eef1ff; border-left: 4px solid #4a6cf7;
        padding: 10px 14px; border-radius: 0 6px 6px 0;
        margin: 6px 0; font-family: monospace; font-size: 0.85em;
    }
    .stat-card {
        background: white; border-radius: 10px; padding: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06); border: 1px solid #e8ebf0;
        text-align: center;
    }
    .stat-card h3 { color: #4a6cf7; margin: 0; font-size: 1.6em; }
    .stat-card p { color: #6b7280; margin: 4px 0 0 0; font-size: 0.85em; }
    .verb-tag {
        display: inline-block; background: #eef1ff; color: #4a6cf7;
        padding: 3px 10px; border-radius: 6px; font-size: 0.8em;
        font-weight: 600; border: 1px solid #c7d0ff; margin: 2px;
    }
    .spacer { margin: 16px 0; }
    .code-block { background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px; font-family: monospace; font-size: 0.85em; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════
PUBLICATION_THEMES = {
    "theme_minimal()": {"r": "theme_minimal()", "pkg": "ggplot2"},
    "theme_bw()": {"r": "theme_bw()", "pkg": "ggplot2"},
    "theme_classic()": {"r": "theme_classic()", "pkg": "ggplot2"},
    "theme_linedraw()": {"r": "theme_linedraw()", "pkg": "ggplot2"},
    "theme_light()": {"r": "theme_light()", "pkg": "ggplot2"},
    "theme_void()": {"r": "theme_void()", "pkg": "ggplot2"},
    "theme_ipsum()": {"r": "theme_ipsum()", "pkg": "hrbrthemes"},
    "theme_ft_rc()": {"r": "theme_ft_rc()", "pkg": "hrbrthemes"},
    "theme_economist()": {"r": "theme_economist()", "pkg": "ggthemes"},
    "theme_wsj()": {"r": "theme_wsj()", "pkg": "ggthemes"},
    "theme_fivethirtyeight()": {"r": "theme_fivethirtyeight()", "pkg": "ggthemes"},
    "theme_tufte()": {"r": "theme_tufte()", "pkg": "ggthemes"},
    "theme_solarized()": {"r": "theme_solarized()", "pkg": "ggthemes"},
    "theme_pubr()": {"r": "theme_pubr()", "pkg": "ggpubr"},
}

COLOR_PALETTES = {
    "Default": ["#4a6cf7", "#f97316", "#10b981", "#ef4444", "#8b5cf6", "#06b6d4"],
    "Viridis": ["#440154", "#31688e", "#26828e", "#1f9e89", "#6ece58", "#b5de2b"],
    "Brewer Set1": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"],
    "Nature": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488", "#F39B7F"],
    "Economist": ["#01a2d9", "#014d64", "#6794a7", "#76c0c1", "#7ad2f6"],
}

# ════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════
defaults = {
    'data': None, 'original_df': None,
    'pipeline_steps': [], 'viz_code_blocks': [],
    'custom_theme': '', 'step_counter': 0,
    'data_snapshots': [], 'joined_datasets': {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════
def add_step(category, description, r_code):
    st.session_state.step_counter += 1
    st.session_state.pipeline_steps.append({
        'order': st.session_state.step_counter,
        'category': category, 'description': description, 'r_code': r_code,
    })


def save_snapshot():
    if st.session_state.data is not None:
        st.session_state.data_snapshots.append(st.session_state.data.copy())
        if len(st.session_state.data_snapshots) > 15:
            st.session_state.data_snapshots.pop(0)


def undo_last():
    if st.session_state.data_snapshots:
        st.session_state.data = st.session_state.data_snapshots.pop()
        if st.session_state.pipeline_steps:
            st.session_state.pipeline_steps.pop()
        return True
    return False


def build_r_pipeline(categories=None):
    steps = st.session_state.pipeline_steps
    if categories:
        steps = [s for s in steps if s['category'] in categories]
    steps = sorted(steps, key=lambda x: x['order'])
    all_code = " ".join([s['r_code'] for s in steps])
    pkgs = ["tidyverse"]
    if any(k in all_code for k in ["str_", "regex"]): pkgs.append("stringr")
    if any(k in all_code for k in ["ymd", "mdy", "dmy", "make_date", "year(", "month("]): pkgs.append("lubridate")
    if "fct_" in all_code: pkgs.append("forcats")
    lines = [
        "# R Analysis Pipeline — Data Transformer Studio Pro",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "# R4DS 2e (Wickham, Cetinkaya-Rundel, Grolemund)", "",
    ]
    for pkg in pkgs: lines.append(f"library({pkg})")
    lines += ["", 'data <- read_csv("your_data.csv")', ""]
    if not steps:
        lines.append("# No operations yet"); return "\n".join(lines)
    wrangle = [s for s in steps if s['category'] != 'viz']
    viz = [s for s in steps if s['category'] == 'viz']
    if wrangle:
        cat_labels = {'transform': 'Transform', 'string_date': 'String/Date', 'clean': 'Cleaning', 'tidy': 'Tidy/Reshape'}
        lines.append("data_clean <- data |>")
        parts = []
        cur = None
        for s in wrangle:
            if s['category'] != cur:
                cur = s['category']
                parts.append(f"  # -- {cat_labels.get(cur, cur)} --")
            parts.append(f"  {s['r_code']}")
        lines.append(" |>\n".join(parts))
        lines.append("")
    if viz:
        lines.append("# -- Visualization --")
        for s in viz:
            lines.append(f"\n# {s['description']}")
            lines.append(s['r_code'])
    return "\n".join(lines)


def generate_ggplot_code(plot_type, xc, yc, color_by, title, x_label, y_label, caption,
                          theme_name, size, alpha, color, show_legend):
    """Generate complete, publication-ready ggplot2 R code."""
    sel_theme = PUBLICATION_THEMES.get(theme_name, PUBLICATION_THEMES["theme_minimal()"])
    custom_t = f" +\n  {st.session_state.custom_theme}" if st.session_state.custom_theme else ""
    color_aes = f", color = {color_by}" if color_by and color_by != "None" else ""
    fill_aes = f", fill = {color_by}" if color_by and color_by != "None" else ""

    cap_line = f',\n    caption = "{caption}"' if caption else ""
    legend_pos = "bottom" if show_legend else "none"

    geom_map = {
        "Bar":       f'geom_col(alpha = {alpha}, fill = "{color}"{", position = position_dodge()" if color_by and color_by != "None" else ""})',
        "Line":      f'geom_line(size = {size/10:.1f}, alpha = {alpha}) +\n  geom_point(size = {size/10+0.5:.1f}, alpha = {alpha})',
        "Scatter":   f'geom_point(size = {size/10:.1f}, alpha = {alpha})',
        "Histogram": f'geom_histogram(bins = 30, alpha = {alpha}, fill = "{color}")',
        "Box":       f'geom_boxplot(alpha = {alpha}, fill = "{color}")',
        "Violin":    f'geom_violin(alpha = {alpha}, fill = "{color}") +\n  geom_boxplot(width = 0.1, alpha = 0.7)',
        "Density":   f'geom_density(alpha = {alpha}, fill = "{color}")',
    }

    if plot_type in ["Bar", "Histogram", "Box", "Violin", "Density"]:
        color_aes_used = fill_aes
    else:
        color_aes_used = color_aes

    y_part = f", y = {yc}" if yc and plot_type not in ["Histogram", "Density"] else ""

    r_code = f"""# {title}
ggplot(data_clean, aes(x = {xc}{y_part}{color_aes_used})) +
  {geom_map.get(plot_type, 'geom_point()')} +
  labs(
    title = "{title}",
    x = "{x_label}",
    y = "{y_label}"{cap_line}
  ) +
  {sel_theme["r"]} +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "{legend_pos}"
  ){custom_t}"""
    return r_code


# ════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════
st.markdown("""
<div class='hero-banner'>
    <h1>📊 Data Transformer Studio Pro</h1>
    <p>R4DS 2e Complete Workflow — Import → Tidy → Transform → Visualize → Communicate — Live R Pipeline</p>
</div>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    with st.expander("Theme & Colors", expanded=False):
        theme_source = st.radio("Theme source", ["Built-in", "Custom R Code"], key="theme_src")
        if theme_source == "Custom R Code":
            tc = st.text_area("Paste R theme", height=80, key="cust_th")
            if tc: st.session_state.custom_theme = tc; st.success("✓ Custom theme saved")
        palette_choice = st.selectbox("Palette", list(COLOR_PALETTES.keys()), key="pal")
    st.markdown("---")
    st.markdown("### 📁 Data Import")
    uploaded_file = st.file_uploader("CSV / Excel / TSV", type=['csv', 'xlsx', 'xls', 'tsv'])
    with st.expander("Second Dataset (Joins)", expanded=False):
        jf = st.file_uploader("Upload", type=['csv', 'xlsx', 'xls'], key="j_up")
        jn = st.text_input("Name", "data2", key="j_nm")
        if jf:
            try:
                dj = pd.read_csv(jf) if jf.name.endswith('.csv') else pd.read_excel(jf)
                st.session_state.joined_datasets[jn] = dj
                st.success(f"✓ {jn}: {dj.shape[0]}×{dj.shape[1]}")
            except Exception as e: st.error(str(e))
    if uploaded_file:
        try:
            ext = uploaded_file.name.split('.')[-1].lower()
            if ext == 'csv': dup = pd.read_csv(uploaded_file)
            elif ext == 'tsv': dup = pd.read_csv(uploaded_file, sep='\t')
            else: dup = pd.read_excel(uploaded_file)
            st.session_state.original_df = dup.copy()
            st.session_state.data = dup.copy()
            st.success(f"✓ {uploaded_file.name}")
            st.metric("Rows", f"{dup.shape[0]:,}")
            st.metric("Columns", dup.shape[1])
            st.metric("Missing", f"{dup.isnull().sum().sum():,}")
        except Exception as e: st.error(str(e))
    st.markdown("---")
    st.markdown("### 🔧 Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("↩ Undo", use_container_width=True):
            if undo_last(): st.rerun()
            else: st.warning("Nothing to undo")
    with c2:
        if st.button("🔄 Reset", use_container_width=True):
            if st.session_state.original_df is not None:
                for k in ['pipeline_steps', 'viz_code_blocks', 'data_snapshots']:
                    st.session_state[k] = []
                st.session_state.step_counter = 0
                st.session_state.data = st.session_state.original_df.copy()
                st.rerun()
    st.write(f"**Steps recorded:** {len(st.session_state.pipeline_steps)}")
    st.write(f"**Snapshots:** {len(st.session_state.data_snapshots)}")

# ════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════
if st.session_state.data is not None:
    df = st.session_state.data

    tab_eda, tab_transform, tab_tidy, tab_strings, tab_clean, tab_viz, tab_editor, tab_pipeline, tab_export = st.tabs([
        "📊 EDA", "🔄 Transform", "📐 Tidy/Join", "📝 Str/Date/Factor",
        "🧹 Clean", "📈 Visualize", "🎨 Plot Editor", "⚙️ R Pipeline", "💾 Export"
    ])

    # ════════════════════════════════════════════════════
    # TAB: EDA
    # ════════════════════════════════════════════════════
    with tab_eda:
        st.markdown("<div class='section-header'>📊 Exploratory Data Analysis (R4DS 2e Ch. 2 & 10)</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(f"<div class='stat-card'><h3>{df.shape[0]:,}</h3><p>Rows</p></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='stat-card'><h3>{df.shape[1]}</h3><p>Cols</p></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='stat-card'><h3>{df.isnull().sum().sum():,}</h3><p>Missing</p></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='stat-card'><h3>{df.duplicated().sum():,}</h3><p>Dupes</p></div>", unsafe_allow_html=True)
        with c5: st.markdown(f"<div class='stat-card'><h3>{df.memory_usage(deep=True).sum()/1024**2:.1f}</h3><p>MB</p></div>", unsafe_allow_html=True)

        with st.expander("📋 glimpse()", expanded=True):
            info = pd.DataFrame({
                'Column': df.columns, 'Type': df.dtypes.astype(str),
                'NonNull': df.notnull().sum(),
                'Null%': (df.isnull().sum()/len(df)*100).round(1),
                'Unique': df.nunique(),
                'Sample': [str(df[c].dropna().iloc[0])[:40] if df[c].notnull().any() else 'NA' for c in df.columns]
            })
            st.dataframe(info, use_container_width=True, height=280)

        with st.expander("📈 summary()"):
            nc = df.select_dtypes(include=['number']).columns.tolist()
            if nc:
                desc = df[nc].describe().T
                desc['IQR'] = desc['75%'] - desc['25%']
                desc['skew'] = df[nc].skew()
                st.dataframe(desc.round(3), use_container_width=True)
            else:
                st.info("No numeric columns")

        with st.expander("🔍 Distributions"):
            if HAS_PLOTLY:
                dc = st.selectbox("Variable", df.columns.tolist(), key="eda_dc")
                if df[dc].dtype in ['int64', 'float64']:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig = px.histogram(df, x=dc, nbins=40, template="plotly_white", color_discrete_sequence=["#4a6cf7"])
                        fig.update_layout(height=300, margin=dict(t=20,b=20), title=f"Distribution of {dc}")
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.box(df, y=dc, template="plotly_white", color_discrete_sequence=["#4a6cf7"])
                        fig.update_layout(height=300, margin=dict(t=20,b=20), title=f"Boxplot of {dc}")
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    cts = df[dc].value_counts().head(20)
                    fig = px.bar(x=cts.index.astype(str), y=cts.values, template="plotly_white",
                                 color_discrete_sequence=["#4a6cf7"], labels={"x": dc, "y": "Count"})
                    fig.update_layout(height=350, title=f"Value Counts: {dc}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Install plotly: `pip install plotly`")

        with st.expander("🔗 Correlation Heatmap"):
            nc = df.select_dtypes(include=['number']).columns.tolist()
            if HAS_PLOTLY and len(nc) >= 2:
                corr = df[nc].corr()
                fig = go.Figure(data=go.Heatmap(
                    z=corr.values, x=corr.columns, y=corr.columns,
                    colorscale='RdBu', zmid=0,
                    text=corr.values.round(2), texttemplate='%{text}'
                ))
                fig.update_layout(height=500, template="plotly_white", title="Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
            elif len(nc) < 2:
                st.info("Need at least 2 numeric columns for correlation")
            else:
                st.warning("Install plotly")

        with st.expander("🔎 Missing Values Pattern"):
            miss = df.isnull().sum()
            miss = miss[miss > 0]
            if len(miss) > 0:
                miss_df = miss.reset_index()
                miss_df.columns = ['Column', 'Missing']
                miss_df['Pct'] = (miss_df['Missing'] / len(df) * 100).round(1)
                if HAS_PLOTLY:
                    fig = px.bar(miss_df, x='Column', y='Missing', text='Pct',
                                 color='Pct', color_continuous_scale='Reds', template="plotly_white")
                    fig.update_traces(texttemplate='%{text}%', textposition='outside')
                    fig.update_layout(height=350, title="Missing Values by Column")
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(miss_df, use_container_width=True)
            else:
                st.success("✅ No missing values")

        st.markdown("**Data Preview (first 15 rows)**")
        st.dataframe(df.head(15), use_container_width=True, height=250)

    # ════════════════════════════════════════════════════
    # TAB: TRANSFORM
    # ════════════════════════════════════════════════════
    with tab_transform:
        st.markdown("<div class='section-header'>🔄 Transform (R4DS 2e Ch. 3 — dplyr)</div>", unsafe_allow_html=True)
        sub_r, sub_c, sub_g, sub_a = st.tabs(["Rows", "Columns", "Groups", "Advanced"])

        with sub_r:
            st.markdown("<span class='verb-tag'>filter()</span>", unsafe_allow_html=True)
            fc = st.selectbox("Column to filter", ["— select —"] + df.columns.tolist(), key="tf_fc")
            if fc != "— select —":
                if df[fc].dtype == 'object':
                    vals = st.multiselect("Keep values", sorted(df[fc].dropna().unique().tolist()), key="tf_fv")
                    neg = st.checkbox("Negate (exclude these values)", key="tf_fn")
                    if vals and st.button("Apply filter()", key="tf_fb"):
                        save_snapshot()
                        if neg:
                            st.session_state.data = st.session_state.data[~st.session_state.data[fc].isin(vals)]
                        else:
                            st.session_state.data = st.session_state.data[st.session_state.data[fc].isin(vals)]
                        vs = ", ".join([f'"{v}"' for v in vals])
                        add_step('transform', f'Filter {fc}', f'filter({"!" if neg else ""}{fc} %in% c({vs}))')
                        st.rerun()
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1: op = st.selectbox("Operator", [">=", "<=", "==", "!=", ">", "<", "between"], key="tf_fop")
                    with c2: v1 = st.number_input("Value 1", value=float(df[fc].min()), key="tf_v1")
                    with c3:
                        v2 = st.number_input("Value 2 (between)", value=float(df[fc].max()), key="tf_v2") if op == "between" else None
                    if st.button("Apply filter()", key="tf_fnb"):
                        save_snapshot()
                        if op == "between":
                            st.session_state.data = st.session_state.data[
                                (st.session_state.data[fc] >= v1) & (st.session_state.data[fc] <= v2)]
                            add_step('transform', f'Filter {fc}', f'filter(between({fc}, {v1}, {v2}))')
                        else:
                            mask = st.session_state.data[fc].apply(lambda x: eval(f"x {op} {v1}"))
                            st.session_state.data = st.session_state.data[mask]
                            add_step('transform', f'Filter {fc}{op}{v1}', f'filter({fc} {op} {v1})')
                        st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>distinct()</span>", unsafe_allow_html=True)
            dcc = st.multiselect("Distinct by (empty=all rows)", df.columns.tolist(), key="tf_dc")
            if st.button("distinct()", key="tf_db"):
                save_snapshot()
                before = len(st.session_state.data)
                st.session_state.data = st.session_state.data.drop_duplicates(subset=dcc if dcc else None)
                after = len(st.session_state.data)
                cs = ", ".join(dcc) if dcc else ""
                add_step('transform', f'distinct (removed {before-after})',
                         f'distinct({cs + ", .keep_all = TRUE" if cs else ""})')
                st.success(f"Removed {before-after:,} duplicate rows")
                st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>arrange()</span>", unsafe_allow_html=True)
            sc = st.multiselect("Sort by columns", df.columns.tolist(), key="tf_sc")
            if sc:
                asc = st.radio("Direction", ["Ascending", "Descending"], horizontal=True, key="tf_sd") == "Ascending"
                if st.button("arrange()", key="tf_sb"):
                    save_snapshot()
                    st.session_state.data = st.session_state.data.sort_values(sc, ascending=asc)
                    cs = ", ".join([c if asc else f"desc({c})" for c in sc])
                    add_step('transform', f'Sort by {", ".join(sc)}', f'arrange({cs})')
                    st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>slice()</span>", unsafe_allow_html=True)
            sl = st.radio("Variant", ["head", "tail", "sample", "min", "max"], horizontal=True, key="tf_sl")
            sn = st.number_input("n", value=5, min_value=1, key="tf_sn")
            ncc = df.select_dtypes(include=['number']).columns.tolist()
            so = st.selectbox("Order column (for min/max)", ncc, key="tf_so") if sl in ["min", "max"] and ncc else None
            if st.button("slice()", key="tf_slb"):
                save_snapshot()
                n = int(sn)
                if sl == "head":
                    st.session_state.data = st.session_state.data.head(n); rc = f'slice_head(n = {n})'
                elif sl == "tail":
                    st.session_state.data = st.session_state.data.tail(n); rc = f'slice_tail(n = {n})'
                elif sl == "sample":
                    st.session_state.data = st.session_state.data.sample(min(n, len(st.session_state.data))); rc = f'slice_sample(n = {n})'
                elif sl == "min" and so:
                    st.session_state.data = st.session_state.data.nsmallest(n, so); rc = f'slice_min({so}, n = {n})'
                elif sl == "max" and so:
                    st.session_state.data = st.session_state.data.nlargest(n, so); rc = f'slice_max({so}, n = {n})'
                else:
                    st.session_state.data = st.session_state.data.head(n); rc = f'slice_head(n = {n})'
                add_step('transform', f'slice_{sl} n={n}', rc)
                st.rerun()

        with sub_c:
            st.markdown("<span class='verb-tag'>select()</span>", unsafe_allow_html=True)
            sm = st.radio("Mode", ["Keep columns", "Drop columns"], horizontal=True, key="tf_sm")
            if sm == "Keep columns":
                kc = st.multiselect("Columns to keep", df.columns.tolist(), default=df.columns.tolist(), key="tf_kc")
                if kc and len(kc) < len(df.columns) and st.button("select() — keep", key="tf_kb"):
                    save_snapshot()
                    st.session_state.data = st.session_state.data[kc]
                    add_step('transform', f'Select {len(kc)} columns', f'select({", ".join(kc)})')
                    st.rerun()
            else:
                dc2 = st.multiselect("Columns to drop", df.columns.tolist(), key="tf_dc2")
                if dc2 and st.button("select() — drop", key="tf_db2"):
                    save_snapshot()
                    st.session_state.data = st.session_state.data.drop(columns=dc2)
                    add_step('transform', f'Drop {len(dc2)} columns', f'select(-c({", ".join(dc2)}))')
                    st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>rename()</span>", unsafe_allow_html=True)
            rf = st.selectbox("Column to rename", df.columns.tolist(), key="tf_rf")
            rt = st.text_input("New name", rf, key="tf_rt")
            if st.button("rename()", key="tf_rb") and rt != rf and rt:
                save_snapshot()
                st.session_state.data = st.session_state.data.rename(columns={rf: rt})
                add_step('transform', f'Rename {rf}→{rt}', f'rename({rt} = {rf})')
                st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>relocate()</span>", unsafe_allow_html=True)
            rel_col = st.selectbox("Column to move", df.columns.tolist(), key="rel_col")
            rel_pos = st.selectbox("Position", ["First", "Last"] + [f"After: {c}" for c in df.columns if c != rel_col], key="rel_pos")
            if st.button("relocate()", key="rel_btn"):
                save_snapshot()
                cols = list(st.session_state.data.columns)
                cols.remove(rel_col)
                if rel_pos == "First":
                    cols = [rel_col] + cols
                    rc = f'relocate({rel_col})'
                elif rel_pos == "Last":
                    cols = cols + [rel_col]
                    rc = f'relocate({rel_col}, .after = last_col())'
                else:
                    after_col = rel_pos.replace("After: ", "")
                    idx = cols.index(after_col) + 1
                    cols.insert(idx, rel_col)
                    rc = f'relocate({rel_col}, .after = {after_col})'
                st.session_state.data = st.session_state.data[cols]
                add_step('transform', f'Relocate {rel_col}', rc)
                st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>mutate()</span>", unsafe_allow_html=True)
            mn = st.text_input("New column name", key="tf_mn")
            mt = st.selectbox("Operation", ["col ± col", "col ± const", "log/sqrt/abs", "lag/lead", "rank", "custom expr"], key="tf_mt")
            ncc = df.select_dtypes(include=['number']).columns.tolist()
            if mn and ncc:
                if mt == "col ± col" and len(ncc) >= 2:
                    c1, c2, c3 = st.columns(3)
                    with c1: a = st.selectbox("Column A", ncc, key="m_a")
                    with c2: op = st.selectbox("Op", ["+", "-", "*", "/"], key="m_op")
                    with c3: b = st.selectbox("Column B", ncc, key="m_b")
                    if st.button("Create column", key="m1"):
                        save_snapshot()
                        st.session_state.data[mn] = eval(f"st.session_state.data['{a}']{op}st.session_state.data['{b}']")
                        add_step('transform', f'{mn}={a}{op}{b}', f'mutate({mn} = {a} {op} {b})')
                        st.rerun()
                elif mt == "col ± const":
                    c1, c2, c3 = st.columns(3)
                    with c1: a = st.selectbox("Column", ncc, key="m_ac")
                    with c2: op = st.selectbox("Op", ["+", "-", "*", "/", "**"], key="m_aop")
                    with c3: v = st.number_input("Value", 1.0, key="m_av")
                    if st.button("Create column", key="m2"):
                        save_snapshot()
                        st.session_state.data[mn] = eval(f"st.session_state.data['{a}']{op}{v}")
                        rop = "^" if op == "**" else op
                        add_step('transform', f'{mn}={a}{rop}{v}', f'mutate({mn} = {a} {rop} {v})')
                        st.rerun()
                elif mt == "log/sqrt/abs":
                    c1, c2 = st.columns(2)
                    with c1: a = st.selectbox("Column", ncc, key="m_fc")
                    with c2: fn = st.selectbox("Function", ["log", "log2", "log10", "sqrt", "abs", "exp"], key="m_fn")
                    if st.button("Create column", key="m3"):
                        save_snapshot()
                        fns = {"log": np.log, "log2": np.log2, "log10": np.log10, "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp}
                        if fn.startswith("log"):
                            st.session_state.data[mn] = fns[fn](st.session_state.data[a].clip(lower=0.001))
                        elif fn == "sqrt":
                            st.session_state.data[mn] = fns[fn](st.session_state.data[a].clip(lower=0))
                        else:
                            st.session_state.data[mn] = fns[fn](st.session_state.data[a])
                        add_step('transform', f'{mn}={fn}({a})', f'mutate({mn} = {fn}({a}))')
                        st.rerun()
                elif mt == "lag/lead":
                    c1, c2, c3 = st.columns(3)
                    with c1: a = st.selectbox("Column", ncc, key="m_lc")
                    with c2: lf = st.selectbox("Function", ["lag", "lead"], key="m_lf")
                    with c3: ln = st.number_input("n", 1, key="m_ln")
                    if st.button("Create column", key="m4"):
                        save_snapshot()
                        st.session_state.data[mn] = st.session_state.data[a].shift(int(ln) if lf == "lag" else -int(ln))
                        add_step('transform', f'{mn}={lf}({a})', f'mutate({mn} = {lf}({a}, n = {int(ln)}))')
                        st.rerun()
                elif mt == "rank":
                    c1, c2 = st.columns(2)
                    with c1: a = st.selectbox("Column", ncc, key="m_rc")
                    with c2: rf2 = st.selectbox("Function", ["percent_rank", "row_number", "min_rank", "dense_rank", "ntile(4)"], key="m_rf")
                    if st.button("Create column", key="m5"):
                        save_snapshot()
                        if rf2 == "percent_rank": st.session_state.data[mn] = st.session_state.data[a].rank(pct=True)
                        elif rf2 == "row_number": st.session_state.data[mn] = range(1, len(st.session_state.data) + 1)
                        elif rf2 == "min_rank": st.session_state.data[mn] = st.session_state.data[a].rank(method='min')
                        elif rf2 == "dense_rank": st.session_state.data[mn] = st.session_state.data[a].rank(method='dense')
                        elif rf2 == "ntile(4)": st.session_state.data[mn] = pd.qcut(st.session_state.data[a], 4, labels=False) + 1
                        add_step('transform', f'{mn}={rf2}({a})', f'mutate({mn} = {rf2}({a}))')
                        st.rerun()
                elif mt == "custom expr":
                    st.info("Use `df['col']` to reference columns. Example: `df['a'] * 2 + df['b']`")
                    expr = st.text_input("Python expression", key="m_expr")
                    if expr and st.button("Create column", key="m6"):
                        save_snapshot()
                        try:
                            st.session_state.data[mn] = eval(expr, {"df": st.session_state.data, "np": np, "pd": pd})
                            add_step('transform', f'{mn} custom', f'mutate({mn} = ...  # custom expression)')
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {e}")
            elif not mn:
                st.info("Enter a new column name above to enable mutate operations")

        with sub_g:
            st.markdown("<span class='verb-tag'>group_by()</span> + <span class='verb-tag'>summarize()</span>", unsafe_allow_html=True)
            gc = st.multiselect("Group by columns", df.columns.tolist(), key="tf_gc")
            if gc:
                ncc = df.select_dtypes(include=['number']).columns.tolist()
                if ncc:
                    ns = st.number_input("Number of summary columns", 1, 5, 1, key="tf_ns")
                    sums = []
                    for i in range(int(ns)):
                        c1, c2, c3 = st.columns(3)
                        with c1: ac = st.selectbox(f"Column {i+1}", ncc, key=f"g_ac{i}")
                        with c2: af = st.selectbox(f"Function {i+1}", ["mean", "sum", "median", "min", "max", "sd", "n", "n_distinct"], key=f"g_af{i}")
                        with c3: an = st.text_input(f"Name {i+1}", f"{af}_{ac}", key=f"g_an{i}")
                        sums.append((ac, af, an))
                    if st.button("summarize()", key="tf_gb"):
                        save_snapshot()
                        agg = {}; rp = []
                        fm = {"mean": "mean", "sum": "sum", "median": "median", "min": "min", "max": "max", "sd": "std", "n": "count", "n_distinct": "nunique"}
                        for ac, af, an in sums:
                            agg[an] = pd.NamedAgg(column=ac, aggfunc=fm[af])
                            if af == "n": rp.append(f"{an} = n()")
                            elif af == "n_distinct": rp.append(f"{an} = n_distinct({ac})")
                            else: rp.append(f"{an} = {af}({ac}, na.rm = TRUE)")
                        st.session_state.data = st.session_state.data.groupby(gc).agg(**agg).reset_index()
                        gs = ", ".join(gc); rs = ", ".join(rp)
                        add_step('transform', f'Summarize by {gs}', f'group_by({gs}) |>\n  summarize({rs}, .groups = "drop")')
                        st.rerun()
                else:
                    st.warning("No numeric columns to summarize")
            else:
                st.info("Select at least one group-by column")

            st.markdown("---")
            st.markdown("<span class='verb-tag'>count()</span>", unsafe_allow_html=True)
            cc = st.multiselect("Count by", df.columns.tolist(), key="tf_cc")
            srt = st.checkbox("Sort by count", True, key="tf_cs2")
            if cc and st.button("count()", key="tf_cb"):
                save_snapshot()
                st.session_state.data = st.session_state.data.groupby(cc).size().reset_index(name='n')
                if srt: st.session_state.data = st.session_state.data.sort_values('n', ascending=False)
                add_step('transform', f'Count by {", ".join(cc)}', f'count({", ".join(cc)}, sort = {"TRUE" if srt else "FALSE"})')
                st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>add_count()</span> — add count column without collapsing", unsafe_allow_html=True)
            ac_cols = st.multiselect("add_count by", df.columns.tolist(), key="tf_acc")
            ac_name = st.text_input("Count column name", "n", key="tf_acn")
            if ac_cols and st.button("add_count()", key="tf_acb"):
                save_snapshot()
                counts = st.session_state.data.groupby(ac_cols)[ac_cols[0]].transform('count')
                st.session_state.data[ac_name] = counts
                add_step('transform', f'add_count({", ".join(ac_cols)})', f'add_count({", ".join(ac_cols)}, name = "{ac_name}")')
                st.rerun()

        with sub_a:
            st.markdown("<span class='verb-tag'>case_when()</span>", unsafe_allow_html=True)
            cw = st.text_input("New column name", key="tf_cw")
            cws = st.selectbox("Based on column", df.columns.tolist(), key="tf_cws")
            nc2 = st.number_input("Number of conditions", 2, 6, 3, key="tf_nc")
            conds = []
            for i in range(int(nc2)):
                c1, c2, c3 = st.columns(3)
                with c1: cm = st.selectbox(f"Op {i+1}", ["<", "<=", ">=", ">", "=="], key=f"cw_cm{i}")
                with c2: th = st.number_input(f"Value {i+1}", key=f"cw_th{i}")
                with c3: lb = st.text_input(f"Label {i+1}", f"Cat_{i+1}", key=f"cw_lb{i}")
                conds.append((cm, th, lb))
            if cw and st.button("case_when()", key="tf_cwb"):
                save_snapshot()
                result = pd.Series([conds[-1][2]] * len(st.session_state.data), index=st.session_state.data.index)
                rp = []
                for cm, th, lb in conds[:-1]:
                    mask = st.session_state.data[cws].apply(lambda x, c=cm, t=th: eval(f"x {c} {t}"))
                    result = result.where(~mask, lb)
                    rp.append(f'{cws} {cm} {th} ~ "{lb}"')
                rp.append(f'.default = "{conds[-1][2]}"')
                st.session_state.data[cw] = result
                add_step('transform', f'case_when → {cw}', f'mutate({cw} = case_when({", ".join(rp)}))')
                st.rerun()

            st.markdown("---")
            st.markdown("<span class='verb-tag'>if_else()</span>", unsafe_allow_html=True)
            ie_col = st.text_input("New column name", key="ie_col")
            ie_src = st.selectbox("Source column", df.columns.tolist(), key="ie_src")
            c1, c2, c3 = st.columns(3)
            with c1: ie_op = st.selectbox("Op", ["<", "<=", ">", ">=", "==", "!="], key="ie_op")
            with c2: ie_val = st.number_input("Threshold", key="ie_val")
            if df[ie_src].dtype == 'object':
                with c3: ie_val_str = st.text_input("Or string value", key="ie_str")
            ie_true = st.text_input("If TRUE → value", "Yes", key="ie_true")
            ie_false = st.text_input("If FALSE → value", "No", key="ie_false")
            if ie_col and st.button("if_else()", key="ie_btn"):
                save_snapshot()
                try:
                    if df[ie_src].dtype in ['int64', 'float64']:
                        mask = st.session_state.data[ie_src].apply(lambda x: eval(f"x {ie_op} {ie_val}"))
                    else:
                        mask = st.session_state.data[ie_src] == ie_val_str
                    st.session_state.data[ie_col] = np.where(mask, ie_true, ie_false)
                    add_step('transform', f'if_else → {ie_col}', f'mutate({ie_col} = if_else({ie_src} {ie_op} {ie_val}, "{ie_true}", "{ie_false}"))')
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

            st.markdown("---")
            st.markdown("<span class='verb-tag'>across()</span>", unsafe_allow_html=True)
            ac3 = st.multiselect("Apply to columns", df.select_dtypes(include=['number']).columns.tolist(), key="tf_ac2")
            afn = st.selectbox("Function", ["scale(0-1)", "z-score", "log", "sqrt", "round(2)", "replace_na(0)", "abs", "cumsum"], key="tf_afn")
            if ac3 and st.button("across()", key="tf_ab"):
                save_snapshot()
                for c in ac3:
                    if c in st.session_state.data.columns and st.session_state.data[c].dtype in ['int64', 'float64']:
                        if afn == "scale(0-1)":
                            mn2, mx = st.session_state.data[c].min(), st.session_state.data[c].max()
                            if mx != mn2: st.session_state.data[c] = (st.session_state.data[c] - mn2) / (mx - mn2)
                        elif afn == "z-score":
                            m, s = st.session_state.data[c].mean(), st.session_state.data[c].std()
                            if s: st.session_state.data[c] = (st.session_state.data[c] - m) / s
                        elif afn == "log":
                            st.session_state.data[c] = np.log(st.session_state.data[c].clip(lower=0.001))
                        elif afn == "sqrt":
                            st.session_state.data[c] = np.sqrt(st.session_state.data[c].clip(lower=0))
                        elif afn == "round(2)":
                            st.session_state.data[c] = st.session_state.data[c].round(2)
                        elif afn == "replace_na(0)":
                            st.session_state.data[c] = st.session_state.data[c].fillna(0)
                        elif afn == "abs":
                            st.session_state.data[c] = st.session_state.data[c].abs()
                        elif afn == "cumsum":
                            st.session_state.data[c] = st.session_state.data[c].cumsum()
                fn2 = afn.split("(")[0].replace("-", "_")
                add_step('transform', f'across({fn2})', f'mutate(across(c({", ".join(ac3)}), ~{fn2}(.x)))')
                st.rerun()

        st.markdown("---")
        st.dataframe(st.session_state.data, use_container_width=True, height=250)
        st.write(f"**{st.session_state.data.shape[0]:,} × {st.session_state.data.shape[1]}**")

    # ════════════════════════════════════════════════════
    # TAB: TIDY/JOIN
    # ════════════════════════════════════════════════════
    with tab_tidy:
        st.markdown("<div class='section-header'>📐 Tidy & Joins (R4DS 2e Ch. 5 & 19)</div>", unsafe_allow_html=True)
        sub_pv, sub_sp, sub_jn = st.tabs(["Pivot", "Separate/Unite", "Joins"])

        with sub_pv:
            pd2 = st.radio("Direction", ["pivot_longer (wide → long)", "pivot_wider (long → wide)"], horizontal=True, key="td_pd")
            if "longer" in pd2:
                st.markdown("<span class='verb-tag'>pivot_longer()</span>", unsafe_allow_html=True)
                pc = st.multiselect("Columns to pivot into rows", df.columns.tolist(), key="td_pc")
                c1, c2 = st.columns(2)
                with c1: nt = st.text_input("names_to", "name", key="td_nt")
                with c2: vt = st.text_input("values_to", "value", key="td_vt")
                if pc:
                    idc = [c for c in df.columns if c not in pc]
                    st.write(f"ID columns (kept): {idc}")
                    if st.button("pivot_longer()", key="td_plb"):
                        save_snapshot()
                        st.session_state.data = st.session_state.data.melt(
                            id_vars=idc, value_vars=pc, var_name=nt, value_name=vt)
                        add_step('tidy', 'pivot_longer', f'pivot_longer(cols = c({", ".join(pc)}), names_to = "{nt}", values_to = "{vt}")')
                        st.rerun()
            else:
                st.markdown("<span class='verb-tag'>pivot_wider()</span>", unsafe_allow_html=True)
                nf = st.selectbox("names_from", df.columns.tolist(), key="td_nf")
                vf = st.selectbox("values_from", [c for c in df.columns if c != nf], key="td_vf")
                st.write(f"Unique values in names_from: {df[nf].nunique()}")
                if st.button("pivot_wider()", key="td_pwb"):
                    save_snapshot()
                    idx = [c for c in st.session_state.data.columns if c not in [nf, vf]]
                    try:
                        st.session_state.data = st.session_state.data.pivot_table(
                            index=idx, columns=nf, values=vf, aggfunc='first').reset_index()
                        st.session_state.data.columns.name = None
                        if isinstance(st.session_state.data.columns, pd.MultiIndex):
                            st.session_state.data.columns = ['_'.join(str(x) for x in c).strip('_') for c in st.session_state.data.columns]
                        add_step('tidy', 'pivot_wider', f'pivot_wider(names_from = {nf}, values_from = {vf})')
                        st.rerun()
                    except Exception as e:
                        st.error(f"pivot_wider failed: {e}")

        with sub_sp:
            sa = st.radio("Action", ["separate()", "unite()"], horizontal=True, key="td_sa")
            if sa == "separate()":
                strc = df.select_dtypes(include=['object']).columns.tolist()
                if strc:
                    sc2 = st.selectbox("Column to split", strc, key="td_sc2")
                    sep = st.text_input("Separator", "_", key="td_sep")
                    sample = str(df[sc2].dropna().iloc[0]) if df[sc2].notnull().any() else ""
                    parts = sample.split(sep)
                    st.write(f"**Preview:** `{sample}` → {parts}")
                    nn = st.text_input("New column names (comma-separated)", ", ".join([f"part{i+1}" for i in range(len(parts))]), key="td_nn")
                    if st.button("separate()", key="td_sb"):
                        save_snapshot()
                        names = [n.strip() for n in nn.split(",")]
                        sp = st.session_state.data[sc2].str.split(sep, expand=True)
                        for i, n in enumerate(names):
                            if i < sp.shape[1]: st.session_state.data[n] = sp[i]
                        st.session_state.data = st.session_state.data.drop(columns=[sc2])
                        ns = ", ".join([f'"{n}"' for n in names])
                        add_step('tidy', f'separate {sc2}', f'separate({sc2}, into = c({ns}), sep = "{sep}")')
                        st.rerun()
                else:
                    st.info("No string columns available")
            else:
                uc = st.multiselect("Columns to unite", df.columns.tolist(), key="td_uc")
                c1, c2 = st.columns(2)
                with c1: un = st.text_input("New column name", "combined", key="td_un")
                with c2: us = st.text_input("Separator", "_", key="td_us")
                rm = st.checkbox("Remove original columns", True, key="td_rm")
                if uc and st.button("unite()", key="td_ub"):
                    save_snapshot()
                    st.session_state.data[un] = st.session_state.data[uc].astype(str).agg(us.join, axis=1)
                    if rm: st.session_state.data = st.session_state.data.drop(columns=uc)
                    add_step('tidy', f'unite → {un}', f'unite("{un}", {", ".join(uc)}, sep = "{us}")')
                    st.rerun()

        with sub_jn:
            if st.session_state.joined_datasets:
                jd = st.selectbox("Join with dataset", list(st.session_state.joined_datasets.keys()), key="td_jd")
                jt = st.selectbox("Join type", ["left_join", "right_join", "inner_join", "full_join", "semi_join", "anti_join"], key="td_jt")
                df2 = st.session_state.joined_datasets[jd]
                st.write(f"**{jd}** shape: {df2.shape[0]} × {df2.shape[1]}")
                common = list(set(df.columns) & set(df2.columns))
                if common:
                    jb = st.multiselect("Join by columns", common, default=common[:1], key="td_jb")
                    st.dataframe(df2.head(3), use_container_width=True)
                    if jb and st.button(f"{jt}()", key="td_jb2"):
                        save_snapshot()
                        hm = {"left_join": "left", "right_join": "right", "inner_join": "inner", "full_join": "outer"}
                        if jt in hm:
                            st.session_state.data = st.session_state.data.merge(df2, on=jb, how=hm[jt], suffixes=('.x', '.y'))
                        elif jt == "semi_join":
                            st.session_state.data = st.session_state.data[st.session_state.data[jb[0]].isin(df2[jb[0]])]
                        elif jt == "anti_join":
                            st.session_state.data = st.session_state.data[~st.session_state.data[jb[0]].isin(df2[jb[0]])]
                        bs = ", ".join([f'"{b}"' for b in jb])
                        add_step('tidy', jt, f'{jt}({jd}, by = c({bs}))')
                        st.rerun()
                else:
                    st.warning("No common columns between datasets for joining")
            else:
                st.info("Upload a second dataset in the sidebar under 'Second Dataset (Joins)'")

        st.markdown("---")
        st.dataframe(st.session_state.data, use_container_width=True, height=200)
        st.write(f"**{st.session_state.data.shape[0]:,} × {st.session_state.data.shape[1]}**")

    # ════════════════════════════════════════════════════
    # TAB: STRINGS / DATES / FACTORS
    # ════════════════════════════════════════════════════
    with tab_strings:
        st.markdown("<div class='section-header'>📝 Strings, Dates & Factors (R4DS 2e Ch. 14-17)</div>", unsafe_allow_html=True)
        sub_s, sub_d, sub_f = st.tabs(["Strings", "Dates", "Factors"])

        with sub_s:
            strc = df.select_dtypes(include=['object']).columns.tolist()
            if strc:
                sc3 = st.selectbox("String column", strc, key="s_sc")
                sop = st.selectbox("Operation", [
                    "str_to_upper", "str_to_lower", "str_to_title", "str_trim", "str_squish",
                    "str_replace", "str_detect (filter)", "str_extract (regex)",
                    "str_remove", "str_length", "str_sub", "str_pad", "str_starts", "str_ends"
                ], key="s_op")

                if sop in ["str_to_upper", "str_to_lower", "str_to_title"]:
                    if st.button("Apply", key="s_case"):
                        save_snapshot()
                        fm = {"str_to_upper": "upper", "str_to_lower": "lower", "str_to_title": "title"}
                        st.session_state.data[sc3] = getattr(st.session_state.data[sc3].str, fm[sop])()
                        add_step('string_date', f'{sop}({sc3})', f'mutate({sc3} = {sop}({sc3}))')
                        st.rerun()

                elif sop == "str_trim":
                    side = st.radio("Trim side", ["both", "left", "right"], horizontal=True, key="s_trim_side")
                    if st.button("Apply", key="s_trim"):
                        save_snapshot()
                        if side == "both": st.session_state.data[sc3] = st.session_state.data[sc3].str.strip()
                        elif side == "left": st.session_state.data[sc3] = st.session_state.data[sc3].str.lstrip()
                        else: st.session_state.data[sc3] = st.session_state.data[sc3].str.rstrip()
                        add_step('string_date', 'str_trim', f'mutate({sc3} = str_trim({sc3}, side = "{side}"))')
                        st.rerun()

                elif sop == "str_squish":
                    if st.button("Apply", key="s_sq"):
                        save_snapshot()
                        st.session_state.data[sc3] = st.session_state.data[sc3].str.strip().str.replace(r'\s+', ' ', regex=True)
                        add_step('string_date', 'str_squish', f'mutate({sc3} = str_squish({sc3}))')
                        st.rerun()

                elif sop == "str_replace":
                    c1, c2 = st.columns(2)
                    with c1: pat = st.text_input("Pattern (regex)", key="s_pat")
                    with c2: rep = st.text_input("Replacement", key="s_rep")
                    all_occ = st.checkbox("Replace all occurrences (str_replace_all)", True, key="s_all")
                    if pat and st.button("Apply", key="s_repb"):
                        save_snapshot()
                        st.session_state.data[sc3] = st.session_state.data[sc3].str.replace(pat, rep, regex=True)
                        fn = "str_replace_all" if all_occ else "str_replace"
                        add_step('string_date', f'{fn}', f'mutate({sc3} = {fn}({sc3}, "{pat}", "{rep}"))')
                        st.rerun()

                elif sop == "str_detect (filter)":
                    pat = st.text_input("Pattern (regex)", key="s_dpat")
                    neg = st.checkbox("Negate — exclude matches", key="s_dn")
                    if pat:
                        matches = st.session_state.data[sc3].str.contains(pat, regex=True, na=False).sum()
                        st.write(f"Matches: **{matches:,}** rows")
                        if st.button("Apply filter", key="s_db"):
                            save_snapshot()
                            mask = st.session_state.data[sc3].str.contains(pat, regex=True, na=False)
                            st.session_state.data = st.session_state.data[~mask if neg else mask]
                            add_step('string_date', 'str_detect filter', f'filter({"!" if neg else ""}str_detect({sc3}, "{pat}"))')
                            st.rerun()

                elif sop == "str_extract (regex)":
                    pat = st.text_input("Regex pattern (use capture group)", r"(\w+)", key="s_epat")
                    en = st.text_input("New column name", f"{sc3}_ext", key="s_en")
                    if pat:
                        preview = df[sc3].str.extract(f'({pat})', expand=False).head(3).tolist()
                        st.write(f"Preview: {preview}")
                    if pat and st.button("Apply", key="s_eb"):
                        save_snapshot()
                        st.session_state.data[en] = st.session_state.data[sc3].str.extract(f'({pat})', expand=False)
                        add_step('string_date', f'str_extract → {en}', f'mutate({en} = str_extract({sc3}, "{pat}"))')
                        st.rerun()

                elif sop == "str_remove":
                    pat = st.text_input("Pattern to remove (regex)", key="s_rpat")
                    all_occ = st.checkbox("Remove all occurrences", True, key="s_rall")
                    if pat and st.button("Apply", key="s_rb"):
                        save_snapshot()
                        st.session_state.data[sc3] = st.session_state.data[sc3].str.replace(pat, "", regex=True)
                        fn = "str_remove_all" if all_occ else "str_remove"
                        add_step('string_date', fn, f'mutate({sc3} = {fn}({sc3}, "{pat}"))')
                        st.rerun()

                elif sop == "str_length":
                    en = st.text_input("New column name", f"{sc3}_len", key="s_ln")
                    if st.button("Apply", key="s_lb"):
                        save_snapshot()
                        st.session_state.data[en] = st.session_state.data[sc3].str.len()
                        add_step('string_date', 'str_length', f'mutate({en} = str_length({sc3}))')
                        st.rerun()

                elif sop == "str_sub":
                    c1, c2, c3 = st.columns(3)
                    with c1: ss = st.number_input("Start (1-indexed)", 1, key="s_ss")
                    with c2: se = st.number_input("End", 5, key="s_se")
                    with c3: en = st.text_input("New column", f"{sc3}_sub", key="s_sn")
                    sample_val = str(df[sc3].dropna().iloc[0]) if df[sc3].notnull().any() else ""
                    st.write(f"Preview: `{sample_val}` → `{sample_val[int(ss)-1:int(se)]}`")
                    if st.button("Apply", key="s_sb2"):
                        save_snapshot()
                        st.session_state.data[en] = st.session_state.data[sc3].str[int(ss)-1:int(se)]
                        add_step('string_date', 'str_sub', f'mutate({en} = str_sub({sc3}, {int(ss)}, {int(se)}))')
                        st.rerun()

                elif sop == "str_pad":
                    c1, c2, c3 = st.columns(3)
                    with c1: width = st.number_input("Width", 10, key="s_pw")
                    with c2: side_p = st.selectbox("Side", ["right", "left", "both"], key="s_ps")
                    with c3: pad_char = st.text_input("Pad char", " ", key="s_pc")
                    if st.button("Apply", key="s_padb"):
                        save_snapshot()
                        st.session_state.data[sc3] = st.session_state.data[sc3].str.pad(int(width), side=side_p, fillchar=pad_char[0] if pad_char else " ")
                        add_step('string_date', 'str_pad', f'mutate({sc3} = str_pad({sc3}, {int(width)}, side = "{side_p}", pad = "{pad_char}"))')
                        st.rerun()

                elif sop in ["str_starts", "str_ends"]:
                    pat = st.text_input("Pattern", key="s_se_pat")
                    neg = st.checkbox("Negate", key="s_se_neg")
                    en = st.text_input("New column name", f"{sc3}_{sop.replace('str_','')}", key="s_se_en")
                    if pat and st.button("Apply", key="s_se_btn"):
                        save_snapshot()
                        if sop == "str_starts":
                            result = st.session_state.data[sc3].str.startswith(pat)
                        else:
                            result = st.session_state.data[sc3].str.endswith(pat)
                        if neg: result = ~result
                        st.session_state.data[en] = result
                        add_step('string_date', sop, f'mutate({en} = {"!" if neg else ""}{sop}({sc3}, "{pat}"))')
                        st.rerun()

                # Show sample values
                st.write(f"**Sample values:** {df[sc3].dropna().unique()[:5].tolist()}")
            else:
                st.info("No string (character) columns in dataset")

        with sub_d:
            st.subheader("Parse Dates")
            dc3 = st.selectbox("Column to parse as date", df.columns.tolist(), key="d_dc")
            dfmt = st.selectbox("Expected format", ["ymd", "mdy", "dmy", "ymd_hms", "dmy_hms", "Auto-detect"], key="d_fmt")
            c1, c2 = st.columns(2)
            with c1:
                if df[dc3].notnull().any():
                    st.write(f"**Sample:** `{df[dc3].dropna().iloc[0]}`")
            with c2:
                if st.button("Parse as date", key="d_pb"):
                    save_snapshot()
                    try:
                        st.session_state.data[dc3] = pd.to_datetime(st.session_state.data[dc3], infer_datetime_format=True)
                        add_step('string_date', f'Parse date: {dc3}', f'mutate({dc3} = {dfmt}({dc3}))')
                        st.success("✅ Parsed successfully")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")

            st.markdown("---")
            st.subheader("Extract Date Components")
            dtc = st.session_state.data.select_dtypes(include=['datetime64']).columns.tolist()
            if dtc:
                dc4 = st.selectbox("Date column", dtc, key="d_dc4")
                comps = st.multiselect("Components to extract", ["year", "month", "day", "wday", "hour", "minute", "quarter", "week"], key="d_comps")
                if comps and st.button("Extract components", key="d_eb"):
                    save_snapshot()
                    rp = []
                    for comp in comps:
                        nn2 = f"{dc4}_{comp}"
                        comp_map = {
                            "year": "dt.year", "month": "dt.month", "day": "dt.day",
                            "wday": None, "hour": "dt.hour", "minute": "dt.minute",
                            "quarter": "dt.quarter", "week": "dt.isocalendar().week"
                        }
                        if comp == "wday":
                            st.session_state.data[nn2] = st.session_state.data[dc4].dt.dayofweek + 1
                        elif comp == "week":
                            st.session_state.data[nn2] = st.session_state.data[dc4].dt.isocalendar().week.astype(int)
                        else:
                            st.session_state.data[nn2] = getattr(st.session_state.data[dc4].dt, comp)
                        rp.append(f"{nn2} = {comp}({dc4})")
                    add_step('string_date', 'Extract date components', f'mutate({", ".join(rp)})')
                    st.rerun()

                st.markdown("---")
                st.subheader("Date Arithmetic")
                dc5 = st.selectbox("Date column", dtc, key="d_dc5")
                da_op = st.selectbox("Operation", ["days since today", "time difference between 2 cols", "add/subtract days"], key="da_op")
                if da_op == "days since today":
                    da_name = st.text_input("New column name", f"{dc5}_days_ago", key="da_name")
                    if st.button("Compute", key="da_btn"):
                        save_snapshot()
                        st.session_state.data[da_name] = (pd.Timestamp.now() - st.session_state.data[dc5]).dt.days
                        add_step('string_date', f'days since {dc5}', f'mutate({da_name} = as.numeric(today() - {dc5}))')
                        st.rerun()
                elif da_op == "time difference between 2 cols" and len(dtc) >= 2:
                    dc6 = st.selectbox("End date column", [c for c in dtc if c != dc5], key="dc6")
                    da_name = st.text_input("New column name", "days_diff", key="da_name2")
                    if st.button("Compute difference", key="da_btn2"):
                        save_snapshot()
                        st.session_state.data[da_name] = (st.session_state.data[dc6] - st.session_state.data[dc5]).dt.days
                        add_step('string_date', f'date diff', f'mutate({da_name} = as.numeric({dc6} - {dc5}))')
                        st.rerun()
                elif da_op == "add/subtract days":
                    da_n = st.number_input("Days to add (negative to subtract)", 7, key="da_n")
                    da_name = st.text_input("New column name", f"{dc5}_shifted", key="da_name3")
                    if st.button("Compute", key="da_btn3"):
                        save_snapshot()
                        st.session_state.data[da_name] = st.session_state.data[dc5] + pd.Timedelta(days=int(da_n))
                        add_step('string_date', 'shift date', f'mutate({da_name} = {dc5} + {int(da_n)})')
                        st.rerun()
            else:
                st.info("No datetime columns found. Parse a column as date first.")

        with sub_f:
            catc = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if catc:
                fc2 = st.selectbox("Factor column", catc, key="f_fc")
                st.write(f"**Levels ({df[fc2].nunique()}):** {sorted(df[fc2].dropna().unique().tolist())[:15]}")
                fop = st.selectbox("Operation", ["fct_infreq", "fct_rev", "fct_reorder", "fct_lump_n", "fct_recode", "fct_collapse", "fct_explicit_na"], key="f_op")

                if fop == "fct_infreq":
                    if st.button("Apply — order by frequency", key="f_infreq"):
                        save_snapshot()
                        order = st.session_state.data[fc2].value_counts().index.tolist()
                        st.session_state.data[fc2] = pd.Categorical(st.session_state.data[fc2], categories=order, ordered=True)
                        add_step('string_date', f'fct_infreq({fc2})', f'mutate({fc2} = fct_infreq({fc2}))')
                        st.rerun()

                elif fop == "fct_rev":
                    if st.button("Apply — reverse level order", key="f_rev"):
                        save_snapshot()
                        current_cats = st.session_state.data[fc2].unique().tolist()
                        if hasattr(st.session_state.data[fc2], 'cat'):
                            current_cats = list(st.session_state.data[fc2].cat.categories)
                        st.session_state.data[fc2] = pd.Categorical(st.session_state.data[fc2], categories=current_cats[::-1], ordered=True)
                        add_step('string_date', f'fct_rev({fc2})', f'mutate({fc2} = fct_rev({fc2}))')
                        st.rerun()

                elif fop == "fct_reorder":
                    ncc = df.select_dtypes(include=['number']).columns.tolist()
                    if ncc:
                        rc2 = st.selectbox("Order by numeric column (median)", ncc, key="f_rc")
                        asc_f = st.checkbox("Ascending", True, key="f_asc")
                        if st.button("Apply fct_reorder()", key="f_reorder"):
                            save_snapshot()
                            meds = st.session_state.data.groupby(fc2)[rc2].median().sort_values(ascending=asc_f)
                            st.session_state.data[fc2] = pd.Categorical(st.session_state.data[fc2], categories=meds.index, ordered=True)
                            add_step('string_date', 'fct_reorder', f'mutate({fc2} = fct_reorder({fc2}, {rc2}))')
                            st.rerun()
                    else:
                        st.info("Need a numeric column to reorder by")

                elif fop == "fct_lump_n":
                    ln = st.number_input("Keep top n levels (rest → 'Other')", 5, min_value=1, key="f_ln")
                    other_name = st.text_input("Name for 'other'", "Other", key="f_other")
                    if st.button("Apply fct_lump_n()", key="f_lump"):
                        save_snapshot()
                        top = st.session_state.data[fc2].value_counts().head(int(ln)).index
                        st.session_state.data[fc2] = st.session_state.data[fc2].where(st.session_state.data[fc2].isin(top), other_name)
                        add_step('string_date', f'fct_lump_n({fc2})', f'mutate({fc2} = fct_lump_n({fc2}, n = {ln}))')
                        st.rerun()

                elif fop == "fct_recode":
                    st.write("Recode a specific level:")
                    old = st.selectbox("Old level", sorted(df[fc2].dropna().unique().tolist()), key="f_old")
                    new = st.text_input("New level name", key="f_new")
                    if new and st.button("Recode", key="f_rec"):
                        save_snapshot()
                        st.session_state.data[fc2] = st.session_state.data[fc2].replace(old, new)
                        add_step('string_date', f'fct_recode {old}→{new}', f'mutate({fc2} = fct_recode({fc2}, "{new}" = "{old}"))')
                        st.rerun()

                elif fop == "fct_collapse":
                    gn = st.text_input("New group name", key="f_gn")
                    gv = st.multiselect("Levels to combine", sorted(df[fc2].dropna().unique().tolist()), key="f_gv")
                    if gn and gv and st.button("Collapse", key="f_col"):
                        save_snapshot()
                        st.session_state.data[fc2] = st.session_state.data[fc2].replace(dict.fromkeys(gv, gn))
                        vs = ", ".join([f'"{v}"' for v in gv])
                        add_step('string_date', 'fct_collapse', f'mutate({fc2} = fct_collapse({fc2}, "{gn}" = c({vs})))')
                        st.rerun()

                elif fop == "fct_explicit_na":
                    na_label = st.text_input("Label for NA values", "(Missing)", key="f_na_label")
                    if st.button("Apply fct_explicit_na()", key="f_na_btn"):
                        save_snapshot()
                        st.session_state.data[fc2] = st.session_state.data[fc2].fillna(na_label)
                        add_step('string_date', 'fct_explicit_na', f'mutate({fc2} = fct_explicit_na({fc2}, na_level = "{na_label}"))')
                        st.rerun()
            else:
                st.info("No categorical/character columns available")

    # ════════════════════════════════════════════════════
    # TAB: CLEAN
    # ════════════════════════════════════════════════════
    with tab_clean:
        st.markdown("<div class='section-header'>🧹 Data Cleaning (R4DS 2e Ch. 18)</div>", unsafe_allow_html=True)

        miss = st.session_state.data.isnull().sum()
        total = miss.sum()
        if total > 0:
            st.warning(f"⚠️ {total:,} missing values across {(miss > 0).sum()} columns")
            miss_df = miss[miss > 0].reset_index()
            miss_df.columns = ['Column', 'Missing']
            miss_df['Pct%'] = (miss_df['Missing'] / len(st.session_state.data) * 100).round(1)
            st.dataframe(miss_df, use_container_width=True)

            act = st.radio("Action", ["drop_na", "Drop columns with missing", "fill (forward)", "fill (backward)", "replace_na (value)", "Fill mean/median"], key="c_act")

            if act == "drop_na":
                sc4 = st.multiselect("Only for columns (empty = all)", miss[miss > 0].index.tolist(), key="c_sc4")
                if st.button("Apply drop_na()", key="c_dna"):
                    save_snapshot()
                    before = len(st.session_state.data)
                    if sc4:
                        st.session_state.data = st.session_state.data.dropna(subset=sc4)
                        add_step('clean', f'drop_na({", ".join(sc4)})', f'drop_na({", ".join(sc4)})')
                    else:
                        st.session_state.data = st.session_state.data.dropna()
                        add_step('clean', 'drop_na()', 'drop_na()')
                    after = len(st.session_state.data)
                    st.success(f"Removed {before-after:,} rows")
                    st.rerun()

            elif act == "Drop columns with missing":
                dc5 = st.multiselect("Drop columns", miss[miss > 0].index.tolist(), key="c_dc5")
                if dc5 and st.button("Drop columns", key="c_dcb"):
                    save_snapshot()
                    st.session_state.data = st.session_state.data.drop(columns=dc5)
                    add_step('clean', f'Drop {len(dc5)} cols', f'select(-c({", ".join(dc5)}))')
                    st.rerun()

            elif act == "fill (forward)":
                fill_cols = st.multiselect("Columns to fill (empty = all)", miss[miss > 0].index.tolist(), key="c_ff_cols")
                if st.button("Apply fill (forward)", key="c_ffill"):
                    save_snapshot()
                    if fill_cols:
                        st.session_state.data[fill_cols] = st.session_state.data[fill_cols].ffill()
                    else:
                        st.session_state.data = st.session_state.data.ffill()
                    add_step('clean', 'fill forward', 'fill(everything(), .direction = "down")')
                    st.rerun()

            elif act == "fill (backward)":
                if st.button("Apply fill (backward)", key="c_bfill"):
                    save_snapshot()
                    st.session_state.data = st.session_state.data.bfill()
                    add_step('clean', 'fill backward', 'fill(everything(), .direction = "up")')
                    st.rerun()

            elif act == "replace_na (value)":
                val = st.text_input("Replacement value", "0", key="c_val")
                rep_cols = st.multiselect("Columns (empty = all numeric)", df.select_dtypes(include=['number']).columns.tolist(), key="c_rcols")
                if st.button("Apply replace_na()", key="c_rnab"):
                    save_snapshot()
                    try:
                        fill_val = float(val)
                    except:
                        fill_val = val
                    if rep_cols:
                        st.session_state.data[rep_cols] = st.session_state.data[rep_cols].fillna(fill_val)
                    else:
                        st.session_state.data = st.session_state.data.fillna(fill_val)
                    add_step('clean', f'replace_na({val})', f'mutate(across(everything(), ~replace_na(.x, {val})))')
                    st.rerun()

            elif act.startswith("Fill mean"):
                fm2 = st.radio("Statistic", ["mean", "median"], horizontal=True, key="c_fm")
                fill_cols2 = st.multiselect("Columns (empty = all numeric)", df.select_dtypes(include=['number']).columns.tolist(), key="c_fmc")
                if st.button(f"Apply fill {fm2}", key="c_fmb"):
                    save_snapshot()
                    target_cols = fill_cols2 if fill_cols2 else st.session_state.data.select_dtypes(include=['number']).columns.tolist()
                    for c in target_cols:
                        v2 = st.session_state.data[c].mean() if fm2 == "mean" else st.session_state.data[c].median()
                        st.session_state.data[c] = st.session_state.data[c].fillna(v2)
                    add_step('clean', f'Fill {fm2}', f'mutate(across(where(is.numeric), ~replace_na(.x, {fm2}(.x, na.rm = TRUE))))')
                    st.rerun()
        else:
            st.success("✅ No missing values in current dataset")

        st.markdown("---")
        st.subheader("🔄 Type Conversion")
        c1, c2, c3 = st.columns(3)
        with c1: tc2 = st.selectbox("Column", df.columns.tolist(), key="c_tc")
        with c2: tt = st.selectbox("Convert to", ["numeric", "character", "integer", "factor", "logical"], key="c_tt")
        with c3:
            st.write(f"**Current type:** `{df[tc2].dtype}`")
            st.write(f"**Sample:** `{df[tc2].dropna().iloc[0] if df[tc2].notnull().any() else 'NA'}`")
        if st.button("Convert type", key="c_tcb"):
            save_snapshot()
            try:
                if tt == "numeric": st.session_state.data[tc2] = pd.to_numeric(st.session_state.data[tc2], errors='coerce')
                elif tt == "character": st.session_state.data[tc2] = st.session_state.data[tc2].astype(str)
                elif tt == "integer": st.session_state.data[tc2] = pd.to_numeric(st.session_state.data[tc2], errors='coerce').astype('Int64')
                elif tt == "factor": st.session_state.data[tc2] = st.session_state.data[tc2].astype('category')
                elif tt == "logical": st.session_state.data[tc2] = st.session_state.data[tc2].astype(bool)
                add_step('clean', f'Convert {tc2} to {tt}', f'mutate({tc2} = as.{tt}({tc2}))')
                st.success(f"✅ Converted {tc2} to {tt}")
                st.rerun()
            except Exception as e:
                st.error(f"Conversion failed: {e}")

        st.markdown("---")
        st.subheader("📊 Outlier Detection & Treatment (IQR method)")
        ncc = st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        if ncc:
            oc = st.selectbox("Numeric column", ncc, key="c_oc")
            Q1 = st.session_state.data[oc].quantile(0.25)
            Q3 = st.session_state.data[oc].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            nout = ((st.session_state.data[oc] < lo) | (st.session_state.data[oc] > hi)).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Q1", f"{Q1:.2f}"); c2.metric("Q3", f"{Q3:.2f}")
            c3.metric("IQR bounds", f"[{lo:.2f}, {hi:.2f}]"); c4.metric("Outliers", f"{nout:,}")

            if HAS_PLOTLY:
                fig = px.box(st.session_state.data, y=oc, template="plotly_white", color_discrete_sequence=["#4a6cf7"])
                fig.update_layout(height=250, margin=dict(t=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

            if nout > 0:
                oa = st.radio("Action", ["Remove outlier rows", "Cap (Winsorize)", "Flag as new column"], horizontal=True, key="c_oa")
                if st.button("Apply outlier treatment", key="c_ob"):
                    save_snapshot()
                    if oa == "Remove outlier rows":
                        st.session_state.data = st.session_state.data[(st.session_state.data[oc] >= lo) & (st.session_state.data[oc] <= hi)]
                        add_step('clean', f'Remove outliers in {oc}', f'filter({oc} >= {lo:.2f} & {oc} <= {hi:.2f})')
                    elif oa == "Cap (Winsorize)":
                        st.session_state.data[oc] = st.session_state.data[oc].clip(lo, hi)
                        add_step('clean', f'Cap outliers in {oc}', f'mutate({oc} = pmin(pmax({oc}, {lo:.2f}), {hi:.2f}))')
                    else:
                        st.session_state.data[f'{oc}_outlier'] = (st.session_state.data[oc] < lo) | (st.session_state.data[oc] > hi)
                        add_step('clean', f'Flag outliers in {oc}', f'mutate({oc}_outlier = {oc} < {lo:.2f} | {oc} > {hi:.2f})')
                    st.rerun()
            else:
                st.success(f"✅ No outliers detected in {oc}")

        st.markdown("---")
        st.subheader("🔍 Duplicate Detection")
        dup_cols = st.multiselect("Check duplicates by columns (empty = all)", df.columns.tolist(), key="c_dupc")
        dups = st.session_state.data.duplicated(subset=dup_cols if dup_cols else None)
        st.metric("Duplicate rows", f"{dups.sum():,}")
        if dups.sum() > 0:
            if st.checkbox("Show duplicate rows", key="c_showdup"):
                st.dataframe(st.session_state.data[dups], use_container_width=True, height=200)

    # ════════════════════════════════════════════════════
    # TAB: VISUALIZE — Quick Charts
    # ════════════════════════════════════════════════════
    with tab_viz:
        st.markdown("<div class='section-header'>📈 Visualize (R4DS 2e Ch. 1 & 9 — ggplot2)</div>", unsafe_allow_html=True)

        if not HAS_PLOTLY:
            st.error("Install plotly: `pip install plotly`")
        else:
            all_cols = df.columns.tolist()
            num_cols = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            viz_sub = st.tabs(["Quick Plot", "Distribution", "Relationship", "Facets", "Time Series"])

            # ── Quick Plot ──────────────────────────────────────
            with viz_sub[0]:
                c1, c2, c3 = st.columns(3)
                with c1:
                    pt = st.selectbox("Plot type", ["Bar", "Line", "Scatter", "Histogram", "Box", "Violin", "Density", "Heatmap (count)"], key="vq_pt")
                with c2:
                    xc = st.selectbox("X axis", all_cols, key="vq_xc")
                with c3:
                    yc = st.selectbox("Y axis", ["(none)"] + num_cols, key="vq_yc") if pt not in ["Histogram", "Density"] else "(none)"

                c1, c2 = st.columns(2)
                with c1:
                    color_by = st.selectbox("Color by", ["None"] + cat_cols + num_cols, key="vq_cb")
                with c2:
                    theme = st.selectbox("ggplot2 theme", list(PUBLICATION_THEMES.keys()), key="vq_theme")

                c1, c2, c3 = st.columns(3)
                with c1: ttl = st.text_input("Title", f"{pt} of {xc}", key="vq_ttl")
                with c2: xl = st.text_input("X label", xc, key="vq_xl")
                with c3: yl = st.text_input("Y label", yc if yc != "(none)" else "Count", key="vq_yl")
                cap = st.text_input("Caption (optional)", "", key="vq_cap")

                try:
                    fig = None
                    cb = color_by if color_by != "None" else None
                    pal = COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"])
                    # Only use discrete palette for categorical color columns
                    _cb_cat = (cb is not None and df[cb].dtype == 'object')
                    _pal_safe = pal if (cb is None or _cb_cat) else None

                    if pt == "Bar":
                        if yc != "(none)":
                            fig = px.bar(df, x=xc, y=yc, color=cb,
                                         template="plotly_white", color_discrete_sequence=_pal_safe)
                        else:
                            vc = df[xc].value_counts().reset_index()
                            vc.columns = [xc, 'count']
                            fig = px.bar(vc, x=xc, y='count', template="plotly_white", color_discrete_sequence=pal)
                    elif pt == "Line":
                        if yc != "(none)":
                            fig = px.line(df, x=xc, y=yc, color=cb, markers=True,
                                          template="plotly_white", color_discrete_sequence=_pal_safe)
                    elif pt == "Scatter":
                        if yc != "(none)":
                            fig = px.scatter(df, x=xc, y=yc, color=cb,
                                             template="plotly_white", color_discrete_sequence=_pal_safe)
                    elif pt == "Histogram":
                        # only pass color_discrete_sequence when color col is categorical
                        cb_is_cat = cb is not None and df[cb].dtype == 'object' if cb else False
                        fig = px.histogram(df, x=xc,
                                           color=cb if cb else None,
                                           nbins=40, template="plotly_white",
                                           color_discrete_sequence=pal if cb_is_cat or not cb else None,
                                           marginal="box", opacity=0.8,
                                           barmode='overlay' if cb else 'relative')
                    elif pt == "Box":
                        fig = px.box(df, x=cb if cb else None,
                                     y=xc if yc == "(none)" else yc,
                                     color=cb if cb else None,
                                     template="plotly_white", color_discrete_sequence=_pal_safe)
                    elif pt == "Violin":
                        if cb:
                            fig = px.violin(df, x=cb, y=xc if yc == "(none)" else yc,
                                            color=cb, box=True, template="plotly_white",
                                            color_discrete_sequence=_pal_safe)
                        else:
                            fig = px.violin(df, y=xc if yc == "(none)" else yc,
                                            box=True, template="plotly_white",
                                            color_discrete_sequence=pal)
                    elif pt == "Density":
                        cb_is_cat = cb is not None and df[cb].dtype == 'object' if cb else False
                        fig = px.histogram(df, x=xc, histnorm='density',
                                           color=cb if cb else None,
                                           template="plotly_white",
                                           color_discrete_sequence=pal if cb_is_cat or not cb else None,
                                           marginal="rug", opacity=0.6,
                                           barmode='overlay' if cb else 'relative')
                    elif pt == "Heatmap (count)":
                        if len(cat_cols) >= 2:
                            hx = st.selectbox("Heatmap X", cat_cols, key="hx")
                            hy = st.selectbox("Heatmap Y", [c for c in cat_cols if c != hx], key="hy")
                            ct = df.groupby([hx, hy]).size().reset_index(name='count')
                            fig = px.density_heatmap(ct, x=hx, y=hy, z='count', template="plotly_white")

                    if fig:
                        fig.update_layout(
                            title=dict(text=ttl, font=dict(size=16, color="#1a1f36")),
                            xaxis_title=xl, yaxis_title=yl, height=450,
                            font=dict(family="Arial, sans-serif", size=12)
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        r_code = generate_ggplot_code(
                            pt, xc, yc if yc != "(none)" else None, color_by,
                            ttl, xl, yl, cap, theme, 5, 0.8, pal[0], True
                        )
                        with st.expander("Generated R (ggplot2) Code"):
                            st.code(r_code, language="r")
                        # Store for save button
                        st.session_state['vq_last_r_code'] = r_code
                        st.session_state['vq_last_title'] = ttl
                        st.session_state['vq_last_pt'] = pt
                    else:
                        st.info("Select appropriate X/Y columns for this chart type")
                except Exception as e:
                    st.error(f"Chart error: {e}")

                # Save button outside try/if fig block
                if st.session_state.get('vq_last_r_code'):
                    if st.button("Save to R Pipeline", key="vq_save"):
                        rc = st.session_state['vq_last_r_code']
                        st.session_state.viz_code_blocks.append(rc)
                        add_step('viz', f"{st.session_state['vq_last_pt']}: {st.session_state['vq_last_title']}", rc)
                        st.success("Added to R pipeline")

            # ── Distribution ──────────────────────────────────────
            with viz_sub[1]:
                st.subheader("Distribution Analysis")
                if num_cols:
                    d_col = st.selectbox("Column", num_cols, key="dist_col")
                    d_type = st.radio("Chart", ["Histogram + Box", "ECDF", "Q-Q Plot"], horizontal=True, key="dist_type")
                    d_color = st.selectbox("Color by", ["None"] + cat_cols, key="dist_color")
                    cb2 = d_color if d_color != "None" else None

                    if d_type == "Histogram + Box":
                        pal_d = COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"])
                        if cb2:
                            fig = px.histogram(df, x=d_col, color=cb2, marginal="box", nbins=40,
                                               template="plotly_white", color_discrete_sequence=pal_d,
                                               opacity=0.75, barmode='overlay')
                        else:
                            fig = px.histogram(df, x=d_col, marginal="box", nbins=40,
                                               template="plotly_white", color_discrete_sequence=pal_d,
                                               opacity=0.85)
                        fig.update_layout(height=450, title=f"Distribution: {d_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif d_type == "ECDF":
                        fig = px.ecdf(df, x=d_col, color=cb2, template="plotly_white",
                                      color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]))
                        fig.update_layout(height=400, title=f"ECDF: {d_col}")
                        st.plotly_chart(fig, use_container_width=True)

                    elif d_type == "Q-Q Plot":
                        col_data = df[d_col].dropna().sort_values()
                        n = len(col_data)
                        theoretical = np.array([np.percentile(np.random.normal(0, 1, 10000), (i/(n+1))*100) for i in range(1, n+1)])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=theoretical, y=col_data.values, mode='markers', marker=dict(color='#4a6cf7', size=4)))
                        mn_v, mx_v = min(theoretical.min(), col_data.min()), max(theoretical.max(), col_data.max())
                        fig.add_trace(go.Scatter(x=[mn_v, mx_v], y=[col_data.mean() + col_data.std() * mn_v, col_data.mean() + col_data.std() * mx_v],
                                                  mode='lines', line=dict(color='red', dash='dash'), name='Reference line'))
                        fig.update_layout(height=400, template="plotly_white", title=f"Q-Q Plot: {d_col}",
                                          xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles")
                        st.plotly_chart(fig, use_container_width=True)

                    # Stats summary
                    col_data = df[d_col].dropna()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Mean", f"{col_data.mean():.3f}")
                    c2.metric("Median", f"{col_data.median():.3f}")
                    c3.metric("Std Dev", f"{col_data.std():.3f}")
                    c4.metric("Skewness", f"{col_data.skew():.3f}")
                else:
                    st.info("No numeric columns")

            # ── Relationship ──────────────────────────────────────
            with viz_sub[2]:
                st.subheader("Relationships & Scatter Matrix")
                if len(num_cols) >= 2:
                    rel_type = st.radio("Type", ["Scatter", "Scatter Matrix", "Scatter + Regression", "Bubble"], horizontal=True, key="rel_type")

                    if rel_type == "Scatter":
                        c1, c2, c3 = st.columns(3)
                        with c1: rx = st.selectbox("X", num_cols, key="rel_x")
                        with c2: ry = st.selectbox("Y", [c for c in num_cols if c != rx], key="rel_y")
                        with c3: rc = st.selectbox("Color", ["None"] + cat_cols + num_cols, key="rel_c")
                        rc_val = rc if rc != "None" else None
                        fig = px.scatter(df, x=rx, y=ry, color=rc_val, template="plotly_white",
                                         color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]),
                                         title=f"{ry} vs {rx}")
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)

                    elif rel_type == "Scatter Matrix":
                        sm_cols = st.multiselect("Columns for scatter matrix", num_cols, default=num_cols[:min(4, len(num_cols))], key="sm_cols")
                        sm_c = st.selectbox("Color by", ["None"] + cat_cols, key="sm_c")
                        if sm_cols:
                            fig = px.scatter_matrix(df, dimensions=sm_cols, color=sm_c if sm_c != "None" else None,
                                                    template="plotly_white",
                                                    color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]))
                            fig.update_layout(height=600)
                            st.plotly_chart(fig, use_container_width=True)

                    elif rel_type == "Scatter + Regression":
                        c1, c2 = st.columns(2)
                        with c1: rx = st.selectbox("X", num_cols, key="reg_x")
                        with c2: ry = st.selectbox("Y", [c for c in num_cols if c != rx], key="reg_y")
                        try:
                            fig = px.scatter(df, x=rx, y=ry, trendline="ols", template="plotly_white",
                                             color_discrete_sequence=["#4a6cf7"], title=f"{ry} ~ {rx}")
                        except Exception:
                            fig = px.scatter(df, x=rx, y=ry, trendline="lowess", template="plotly_white",
                                             color_discrete_sequence=["#4a6cf7"], title=f"{ry} ~ {rx} (lowess)")
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)
                        # Correlation
                        corr = df[rx].corr(df[ry])
                        st.metric("Pearson correlation", f"{corr:.4f}")

                    elif rel_type == "Bubble":
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: bx = st.selectbox("X", num_cols, key="bub_x")
                        with c2: by = st.selectbox("Y", [c for c in num_cols if c != bx], key="bub_y")
                        with c3: bsize = st.selectbox("Size", num_cols, key="bub_s")
                        with c4: bc = st.selectbox("Color", ["None"] + cat_cols, key="bub_c")
                        fig = px.scatter(df, x=bx, y=by, size=bsize, color=bc if bc != "None" else None,
                                         template="plotly_white", size_max=40, title=f"Bubble: {by} vs {bx}")
                        fig.update_layout(height=450)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Need at least 2 numeric columns")

            # ── Facets ──────────────────────────────────────
            with viz_sub[3]:
                st.subheader("Facet Plots (facet_wrap / facet_grid)")
                if cat_cols and num_cols:
                    c1, c2, c3 = st.columns(3)
                    with c1: fac_x = st.selectbox("X axis", num_cols, key="fac_x")
                    with c2: fac_y = st.selectbox("Y axis", num_cols, key="fac_y")
                    with c3: fac_col = st.selectbox("Facet by", cat_cols, key="fac_col")
                    fac_type = st.radio("Facet type", ["facet_wrap", "facet_grid (add row)"], horizontal=True, key="fac_type")
                    fac_row = st.selectbox("Row facet (facet_grid only)", ["None"] + cat_cols, key="fac_row") if "grid" in fac_type else "None"
                    fac_pt = st.radio("Geom", ["Scatter", "Bar", "Box"], horizontal=True, key="fac_pt")

                    # Limit to avoid overplotting
                    max_facets = df[fac_col].nunique()
                    st.write(f"Facets: {max_facets}")
                    if max_facets > 20:
                        st.warning(f"⚠️ {max_facets} facets — consider lumping levels first")

                    try:
                        if fac_pt == "Scatter":
                            if fac_row != "None":
                                fig = px.scatter(df, x=fac_x, y=fac_y, facet_col=fac_col, facet_row=fac_row, template="plotly_white",
                                                 color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]))
                            else:
                                fig = px.scatter(df, x=fac_x, y=fac_y, facet_col=fac_col, template="plotly_white",
                                                 color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]))
                        elif fac_pt == "Bar":
                            fig = px.bar(df, x=fac_x, y=fac_y, facet_col=fac_col, template="plotly_white",
                                         color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]))
                        elif fac_pt == "Box":
                            fig = px.box(df, x=fac_col, y=fac_y, color=fac_col, facet_col=fac_row if fac_row != "None" else None,
                                         template="plotly_white",
                                         color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]))
                        fig.update_layout(height=max(400, min(800, max_facets * 50)), title=f"Faceted by {fac_col}")
                        st.plotly_chart(fig, use_container_width=True)

                        r_code_facet = f"""# Facet plot
ggplot(data_clean, aes(x = {fac_x}, y = {fac_y})) +
  geom_{'point' if fac_pt=='Scatter' else 'col' if fac_pt=='Bar' else 'boxplot'}() +
  facet_wrap(~{fac_col}) +
  labs(title = "Faceted by {fac_col}", x = "{fac_x}", y = "{fac_y}") +
  theme_minimal()"""
                        with st.expander("📋 R Code"):
                            st.code(r_code_facet, language="r")
                    except Exception as e:
                        st.error(f"Facet error: {e}")
                else:
                    st.info("Need at least one categorical and one numeric column")

            # ── Time Series ──────────────────────────────────────
            with viz_sub[4]:
                st.subheader("Time Series")
                date_cols = st.session_state.data.select_dtypes(include=['datetime64']).columns.tolist()
                if not date_cols:
                    st.info("No datetime columns. Parse a column as date in the Str/Date/Factor tab first.")
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1: ts_date = st.selectbox("Date column", date_cols, key="ts_date")
                    with c2: ts_y = st.selectbox("Value column", num_cols, key="ts_y")
                    with c3: ts_color = st.selectbox("Group by", ["None"] + cat_cols, key="ts_c")
                    ts_agg = st.selectbox("Aggregate by", ["None", "Day", "Week", "Month", "Quarter", "Year"], key="ts_agg")

                    ts_df = st.session_state.data.copy()
                    if ts_agg != "None":
                        freq_map = {"Day": "D", "Week": "W", "Month": "M", "Quarter": "Q", "Year": "Y"}
                        ts_df = ts_df.set_index(ts_date).resample(freq_map[ts_agg])[ts_y].mean().reset_index()
                        ts_date_plot = ts_date; ts_color_plot = None
                    else:
                        ts_date_plot = ts_date
                        ts_color_plot = ts_color if ts_color != "None" else None

                    fig = px.line(ts_df, x=ts_date_plot, y=ts_y, color=ts_color_plot,
                                  template="plotly_white", markers=True,
                                  color_discrete_sequence=COLOR_PALETTES.get(palette_choice, COLOR_PALETTES["Default"]),
                                  title=f"{ts_y} over time")
                    fig.update_layout(height=450)
                    st.plotly_chart(fig, use_container_width=True)

                    # Range slider
                    fig2 = px.line(ts_df, x=ts_date_plot, y=ts_y, template="plotly_white",
                                   title=f"{ts_y} — with range selector")
                    fig2.update_xaxes(rangeslider_visible=True,
                                      rangeselector=dict(buttons=list([
                                          dict(count=7, label="7d", step="day", stepmode="backward"),
                                          dict(count=1, label="1m", step="month", stepmode="backward"),
                                          dict(count=3, label="3m", step="month", stepmode="backward"),
                                          dict(step="all")
                                      ])))
                    fig2.update_layout(height=350)
                    st.plotly_chart(fig2, use_container_width=True)

    # ════════════════════════════════════════════════════
    # TAB: PLOT EDITOR — Publication-Ready
    # ════════════════════════════════════════════════════
    with tab_editor:
        st.markdown("<div class='section-header'>🎨 Plot Editor — Publication-Ready Charts</div>", unsafe_allow_html=True)
        st.caption("Fine-tune every aesthetic parameter and get clean ggplot2 R code")

        if not HAS_PLOTLY:
            st.error("Install plotly: `pip install plotly`")
        else:
            all_cols_e = df.columns.tolist()
            num_cols_e = df.select_dtypes(include=['number']).columns.tolist()
            cat_cols_e = df.select_dtypes(include=['object', 'category']).columns.tolist()

            # Plot type and axes
            c1, c2, c3, c4 = st.columns(4)
            with c1: pt = st.selectbox("Plot type", ["Bar", "Line", "Scatter", "Box", "Violin", "Histogram", "Density"], key="pe_pt")
            with c2: xc = st.selectbox("X axis", all_cols_e, key="pe_xc")
            with c3:
                y_options = ["(count)"] + num_cols_e if pt in ["Histogram", "Density"] else num_cols_e
                yc = st.selectbox("Y axis", y_options, key="pe_yc")
            with c4:
                cat_pe = cat_cols_e + num_cols_e
                cb = st.selectbox("Color by", ["None"] + cat_pe, key="pe_cb")

            # Labels
            st.subheader("📝 Labels")
            c1, c2, c3 = st.columns(3)
            with c1: ttl = st.text_input("Title", f"{pt} chart", key="pe_ttl")
            with c2: xl = st.text_input("X label", xc, key="pe_xl")
            with c3: yl = st.text_input("Y label", yc if yc != "(count)" else "Count", key="pe_yl")
            c1, c2 = st.columns(2)
            with c1: cap = st.text_input("Caption", "", key="pe_cap")
            with c2: subtitle = st.text_input("Subtitle", "", key="pe_sub")

            # Aesthetics
            st.subheader("🎨 Aesthetics")
            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: theme = st.selectbox("Theme", list(PUBLICATION_THEMES.keys()), key="pe_theme")
            with c2: fs = st.number_input("Font size", 8, 20, 12, key="pe_fs")
            with c3: ht = st.number_input("Height (px)", 300, 900, 500, step=50, key="pe_ht")
            with c4: ms = st.slider("Point/bar size", 1, 20, 5, key="pe_ms")
            with c5: op = st.slider("Opacity", 0.1, 1.0, 0.8, 0.05, key="pe_op")

            # Color options
            c1, c2, c3 = st.columns(3)
            with c1: clr = st.color_picker("Base color", "#4a6cf7", key="pe_clr")
            with c2: palette_pe = st.selectbox("Color palette", list(COLOR_PALETTES.keys()), key="pe_pal")
            with c3: lg = st.checkbox("Show legend", True, key="pe_lg")

            # Advanced options
            with st.expander("⚙️ Advanced options"):
                c1, c2, c3 = st.columns(3)
                with c1: add_ref = st.checkbox("Add reference line", False, key="pe_ref")
                with c2: ref_val = st.number_input("Reference line value", 0.0, key="pe_refv") if add_ref else None
                with c3: ref_col = st.selectbox("Reference color", ["red", "blue", "green", "orange", "black"], key="pe_refc") if add_ref else "red"
                smooth = st.checkbox("Add smooth/trend line (scatter only)", False, key="pe_smooth")
                flip = st.checkbox("Flip coordinates (coord_flip)", False, key="pe_flip")
                log_x = st.checkbox("Log scale X", False, key="pe_logx")
                log_y = st.checkbox("Log scale Y", False, key="pe_logy")

            # Init session state keys for plot editor
            for _k in ['pe_last_r_code', 'pe_last_title', 'pe_last_pt']:
                if _k not in st.session_state:
                    st.session_state[_k] = None

            if st.button("Generate Publication-Ready Plot", key="pe_gen", type="primary"):
                try:
                    pal = COLOR_PALETTES.get(palette_pe, COLOR_PALETTES["Default"])
                    cb_val = cb if cb != "None" else None
                    yc_val = yc if yc != "(count)" else None
                    # Only use discrete palette when color column is categorical
                    cb_is_cat = (cb_val is not None and df[cb_val].dtype == 'object')
                    _pal = pal if (cb_val is None or cb_is_cat) else None

                    fig = None

                    if pt == "Bar":
                        if yc_val:
                            fig = px.bar(df, x=xc, y=yc_val, color=cb_val,
                                         template="plotly_white", color_discrete_sequence=_pal, opacity=op)
                        else:
                            vc = df[xc].value_counts().reset_index()
                            vc.columns = [xc, 'count']
                            fig = px.bar(vc, x=xc, y='count',
                                         template="plotly_white", color_discrete_sequence=pal, opacity=op)
                    elif pt == "Line":
                        if yc_val:
                            fig = px.line(df, x=xc, y=yc_val, color=cb_val, markers=True,
                                          template="plotly_white", color_discrete_sequence=_pal)
                    elif pt == "Scatter":
                        if yc_val:
                            if smooth:
                                fig = px.scatter(df, x=xc, y=yc_val, color=cb_val,
                                                 trendline="lowess", template="plotly_white",
                                                 color_discrete_sequence=_pal, opacity=op)
                            else:
                                fig = px.scatter(df, x=xc, y=yc_val, color=cb_val,
                                                 template="plotly_white",
                                                 color_discrete_sequence=_pal, opacity=op)
                            if fig:
                                fig.update_traces(selector=dict(mode='markers'), marker=dict(size=int(ms)))
                    elif pt == "Box":
                        _y_box = yc_val if yc_val else xc
                        fig = px.box(df, x=cb_val, y=_y_box, color=cb_val,
                                     template="plotly_white", color_discrete_sequence=_pal)
                    elif pt == "Violin":
                        _y_vio = yc_val if yc_val else xc
                        if cb_val:
                            fig = px.violin(df, x=cb_val, y=_y_vio, color=cb_val,
                                            box=True, template="plotly_white", color_discrete_sequence=_pal)
                        else:
                            fig = px.violin(df, y=_y_vio, box=True,
                                            template="plotly_white", color_discrete_sequence=pal)
                    elif pt == "Histogram":
                        _cb_h = cb_val if cb_is_cat else None
                        fig = px.histogram(df, x=xc, color=_cb_h, nbins=40, marginal="box",
                                           template="plotly_white",
                                           color_discrete_sequence=pal if _cb_h else None,
                                           opacity=op,
                                           barmode='overlay' if _cb_h else 'relative')
                    elif pt == "Density":
                        _cb_d = cb_val if cb_is_cat else None
                        fig = px.histogram(df, x=xc, color=_cb_d, histnorm='density',
                                           template="plotly_white",
                                           color_discrete_sequence=pal if _cb_d else None,
                                           opacity=op, marginal="rug",
                                           barmode='overlay' if _cb_d else 'relative')

                    if fig is None:
                        st.warning("Could not create plot — check your column selections (X and Y may be needed).")
                    else:
                        fig.update_layout(
                            title=dict(
                                text="<b>" + ttl + "</b>" + ("<br><sub>" + subtitle + "</sub>" if subtitle else ""),
                                font=dict(size=int(fs) + 2, color="#1a1f36")
                            ),
                            xaxis_title=xl, yaxis_title=yl,
                            height=int(ht), showlegend=lg,
                            font=dict(family="Arial, sans-serif", size=int(fs)),
                            plot_bgcolor='white', paper_bgcolor='white',
                            xaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='#333'),
                            yaxis=dict(showgrid=True, gridcolor='#f0f0f0', linecolor='#333'),
                        )
                        if cap:
                            fig.add_annotation(
                                text=cap, xref="paper", yref="paper", x=0, y=-0.12,
                                showarrow=False, font=dict(size=int(fs)-2, color="#9ca3af"), xanchor='left')
                        if add_ref and ref_val is not None:
                            fig.add_hline(y=float(ref_val), line_dash="dash", line_color=ref_col,
                                          annotation_text=f"Ref: {ref_val}", annotation_position="top right")
                        if flip:
                            fig.update_layout(xaxis_title=yl, yaxis_title=xl)
                        if log_x:
                            fig.update_xaxes(type="log")
                        if log_y:
                            fig.update_yaxes(type="log")

                        st.plotly_chart(fig, use_container_width=True)

                        # Build R code
                        r_code = generate_ggplot_code(pt, xc, yc_val, cb, ttl, xl, yl, cap, theme, ms, op, clr, lg)
                        if subtitle:
                            r_code = r_code.replace(f'title = "{ttl}"', f'title = "{ttl}",\n    subtitle = "{subtitle}"')
                        if flip:
                            r_code += " +\n  coord_flip()"
                        if log_x:
                            r_code += " +\n  scale_x_log10()"
                        if log_y:
                            r_code += " +\n  scale_y_log10()"
                        if add_ref and ref_val is not None:
                            r_code += f" +\n  geom_hline(yintercept = {ref_val}, linetype = 'dashed', color = '{ref_col}')"

                        st.markdown("---")
                        st.subheader("Publication-Ready R Code (ggplot2)")
                        st.code(r_code, language="r")

                        sel_pkg = PUBLICATION_THEMES.get(theme, {}).get("pkg", "ggplot2")
                        if sel_pkg != "ggplot2":
                            st.info(f"Required: `install.packages('{sel_pkg}')`")

                        # Store for save button outside this block
                        st.session_state.pe_last_r_code = r_code
                        st.session_state.pe_last_title = ttl
                        st.session_state.pe_last_pt = pt

                except Exception as e:
                    st.error(f"Plot error: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # Save button OUTSIDE the generate block (Streamlit nested button limitation)
            if st.session_state.get('pe_last_r_code'):
                st.markdown("---")
                if st.button("Save last plot to R Pipeline", key="pe_save"):
                    rc = st.session_state.pe_last_r_code
                    st.session_state.viz_code_blocks.append(rc)
                    add_step('viz', f"{st.session_state.pe_last_pt}: {st.session_state.pe_last_title}", rc)
                    st.success("Added to R pipeline — check the Pipeline tab")

    # ════════════════════════════════════════════════════
    # TAB: R PIPELINE
    # ════════════════════════════════════════════════════
    with tab_pipeline:
        st.markdown("<div class='section-header'>⚙️ R Pipeline — Smart Code Generation</div>", unsafe_allow_html=True)
        st.caption("All operations tracked and compiled into a single reproducible R script using the native pipe |>")

        if not st.session_state.pipeline_steps:
            st.info("No operations recorded yet. Start transforming, cleaning, or visualizing your data.")
        else:
            st.subheader("📋 Operation History")
            cat_icons = {'transform': '🔄', 'string_date': '📝', 'clean': '🧹', 'tidy': '📐', 'viz': '📈'}
            for s in st.session_state.pipeline_steps:
                icon = cat_icons.get(s['category'], '⚙️')
                st.markdown(f"<div class='pipeline-step'><strong>{icon} Step {s['order']}</strong> [{s['category']}] — {s['description']}</div>", unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("⚙️ Generated R Code")

            code_mode = st.radio("Output mode", [
                "Complete Pipeline (wrangling + viz)",
                "Wrangling Only",
                "Visualization Only",
                "Separate Blocks (by category)"
            ], key="p_mode")

            if code_mode == "Complete Pipeline (wrangling + viz)":
                code = build_r_pipeline()
                st.code(code, language="r")

            elif code_mode == "Wrangling Only":
                code = build_r_pipeline(categories=['transform', 'string_date', 'clean', 'tidy'])
                st.code(code, language="r")

            elif code_mode == "Visualization Only":
                code = build_r_pipeline(categories=['viz'])
                st.code(code, language="r")

            elif code_mode == "Separate Blocks (by category)":
                cat_map = {
                    'transform': '🔄 Transform', 'string_date': '📝 String/Date',
                    'clean': '🧹 Cleaning', 'tidy': '📐 Tidy/Reshape', 'viz': '📈 Visualization'
                }
                cats_present = list(dict.fromkeys(s['category'] for s in st.session_state.pipeline_steps))
                for cat in cats_present:
                    steps_in_cat = [s for s in st.session_state.pipeline_steps if s['category'] == cat]
                    with st.expander(f"{cat_map.get(cat, cat)} ({len(steps_in_cat)} steps)", expanded=True):
                        if cat == 'viz':
                            for s in steps_in_cat:
                                st.code(s['r_code'], language="r")
                        else:
                            pipe_parts = [s['r_code'] for s in steps_in_cat]
                            code = "data |>\n  " + " |>\n  ".join(pipe_parts)
                            st.code(code, language="r")

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                full_code = build_r_pipeline()
                st.download_button(
                    "📥 Download Complete R Script", full_code,
                    f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.R", "text/plain",
                    use_container_width=True
                )
            with c2:
                if st.button("🗑️ Clear Pipeline", key="p_clear", use_container_width=True):
                    st.session_state.pipeline_steps = []
                    st.session_state.viz_code_blocks = []
                    st.session_state.step_counter = 0
                    st.rerun()

    # ════════════════════════════════════════════════════
    # TAB: EXPORT
    # ════════════════════════════════════════════════════
    with tab_export:
        st.markdown("<div class='section-header'>💾 Export Data & Scripts</div>", unsafe_allow_html=True)

        # Data summary
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original rows", f"{st.session_state.original_df.shape[0]:,}" if st.session_state.original_df is not None else "—")
        c2.metric("Current rows", f"{st.session_state.data.shape[0]:,}")
        c3.metric("Columns", st.session_state.data.shape[1])
        c4.metric("Steps applied", len(st.session_state.pipeline_steps))

        st.subheader("📄 Download Data")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button(
                "📄 CSV", st.session_state.data.to_csv(index=False),
                f"data_transformed_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
                use_container_width=True
            )
        with c2:
            tsv_data = st.session_state.data.to_csv(index=False, sep='\t')
            st.download_button(
                "📄 TSV", tsv_data,
                f"data_transformed_{datetime.now().strftime('%Y%m%d_%H%M')}.tsv", "text/plain",
                use_container_width=True
            )
        with c3:
            try:
                buf = io.BytesIO()
                st.session_state.data.to_excel(buf, index=False, engine='openpyxl')
                buf.seek(0)
                st.download_button(
                    "📊 Excel (.xlsx)", buf.getvalue(),
                    f"data_transformed_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except Exception as _e:
                st.info(f"Excel export unavailable: {_e}")
        with c4:
            json_data = st.session_state.data.to_json(orient='records', indent=2)
            st.download_button(
                "📋 JSON", json_data,
                f"data_transformed_{datetime.now().strftime('%Y%m%d_%H%M')}.json", "application/json",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📝 Download Scripts")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📝 Complete R Script", build_r_pipeline(),
                f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.R", "text/plain",
                use_container_width=True
            )
        with c2:
            wrangle_code = build_r_pipeline(categories=['transform', 'string_date', 'clean', 'tidy'])
            st.download_button(
                "📝 Wrangling R Script", wrangle_code,
                f"wrangling_{datetime.now().strftime('%Y%m%d_%H%M')}.R", "text/plain",
                use_container_width=True
            )
        with c3:
            viz_code = build_r_pipeline(categories=['viz'])
            st.download_button(
                "📝 Visualization R Script", viz_code,
                f"viz_{datetime.now().strftime('%Y%m%d_%H%M')}.R", "text/plain",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("📚 ggplot2 Theme Reference")
        theme_ref = """# ══════════════════════════════════════════════════════════════
# Publication-Ready ggplot2 Themes Reference
# Data Transformer Studio Pro — R4DS 2e
# ══════════════════════════════════════════════════════════════

# ── Core libraries ────────────────────────────────────────────
library(tidyverse)
library(ggplot2)

# ── Built-in (ggplot2) ────────────────────────────────────────
theme_minimal()         # Clean, no background, subtle gridlines
theme_bw()              # White background, black grid
theme_classic()         # Classic axes, no grid — ideal for journals
theme_linedraw()        # Black borders and lines
theme_light()           # Light grey axis and grid
theme_void()            # Nothing but data — great for maps

# ── hrbrthemes (install.packages("hrbrthemes")) ───────────────
theme_ipsum()           # Typography-focused, Inter font
theme_ipsum_rc()        # Roboto Condensed
theme_ft_rc()           # Financial Times style

# ── ggthemes (install.packages("ggthemes")) ───────────────────
theme_economist()       # The Economist
theme_wsj()             # Wall Street Journal
theme_fivethirtyeight() # FiveThirtyEight data journalism
theme_tufte()           # Edward Tufte — minimal ink
theme_solarized()       # Solarized colour palette
theme_clean()           # Very clean, minimal

# ── ggpubr (install.packages("ggpubr")) ───────────────────────
theme_pubr()            # Publication-ready academic
theme_pubclean()        # Cleaner version of pubr

# ══════════════════════════════════════════════════════════════
# CUSTOM THEME EXAMPLE
# ══════════════════════════════════════════════════════════════
guardian_theme <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 16, color = "#1a1f36"),
    plot.subtitle = element_text(size = 13, color = "#4a5568", margin = margin(b = 10)),
    plot.caption  = element_text(size = 9, color = "#9ca3af", hjust = 0),
    axis.title    = element_text(size = 12, face = "bold", color = "#374151"),
    axis.text     = element_text(size = 11, color = "#4a5568"),
    legend.position   = "bottom",
    legend.title      = element_text(face = "bold", size = 11),
    panel.grid.minor  = element_blank(),
    panel.grid.major  = element_line(color = "#f0f0f0"),
    strip.text        = element_text(face = "bold", size = 12),
    plot.background   = element_rect(fill = "white", color = NA),
    panel.background  = element_rect(fill = "white", color = NA),
    plot.margin       = margin(15, 20, 10, 15)
  )

# ══════════════════════════════════════════════════════════════
# COLOUR SCALES
# ══════════════════════════════════════════════════════════════
scale_color_viridis_d()                          # Discrete viridis
scale_fill_viridis_c()                           # Continuous viridis
scale_fill_brewer(palette = "Set1")              # ColorBrewer categorical
scale_fill_brewer(palette = "Blues")             # ColorBrewer sequential
scale_fill_brewer(palette = "RdBu", direction = -1)  # Diverging

# Nature journal palette
scale_color_manual(values = c("#E64B35", "#4DBBD5", "#00A087", "#3C5488"))

# Economist palette
scale_color_manual(values = c("#01a2d9", "#014d64", "#6794a7", "#76c0c1"))

# ══════════════════════════════════════════════════════════════
# SAVING PUBLICATION-QUALITY FIGURES
# ══════════════════════════════════════════════════════════════
ggsave("figure1.pdf", width = 8, height = 6, dpi = 300)
ggsave("figure1.png", width = 8, height = 6, dpi = 300, bg = "white")
ggsave("figure1.svg", width = 8, height = 6)
"""
        with st.expander("📋 Show Theme Reference Code"):
            st.code(theme_ref, language="r")
        st.download_button("📥 Download Theme Reference (.R)", theme_ref, "ggplot2_themes_reference.R", "text/plain")

        st.markdown("---")
        st.subheader("📊 Final Data Preview")
        st.dataframe(st.session_state.data, use_container_width=True, height=350)
        st.write(f"**Final dataset: {st.session_state.data.shape[0]:,} rows × {st.session_state.data.shape[1]} columns**")

        # Column types summary
        with st.expander("Column types summary"):
            type_summary = pd.DataFrame({
                'Column': st.session_state.data.columns,
                'Type': st.session_state.data.dtypes.astype(str),
                'Non-null': st.session_state.data.notnull().sum(),
                'Unique': st.session_state.data.nunique()
            })
            st.dataframe(type_summary, use_container_width=True)

# ════════════════════════════════════════════════════
# LANDING PAGE (no data)
# ════════════════════════════════════════════════════
else:
    st.info("👆 Upload a CSV, Excel, or TSV file in the sidebar to get started")
    st.markdown("---")
    st.markdown("""
### 🚀 Features (R4DS 2e Complete Coverage)

| Category | R4DS Chapter | Operations |
|---|---|---|
| **📊 EDA** | Ch. 2, 10 | glimpse, summary, distributions, correlation heatmap, missing value patterns |
| **🔄 Transform** | Ch. 3 | filter, arrange, select, relocate, rename, mutate (8 modes), group_by + summarize, count, add_count, slice variants, distinct |
| **📐 Tidy** | Ch. 5 | pivot_longer, pivot_wider, separate, unite |
| **📝 Strings** | Ch. 14–15 | 14 string operations incl. regex, str_pad, str_starts/ends, str_extract |
| **📅 Dates** | Ch. 17 | Parse (ymd/mdy/dmy), extract 8 components, date arithmetic |
| **🏷️ Factors** | Ch. 16 | fct_infreq, fct_rev, fct_reorder, fct_lump_n, fct_recode, fct_collapse, fct_explicit_na |
| **🧹 Missing** | Ch. 18 | drop_na, fill (fwd/bwd), replace_na, mean/median fill, type conversion, outlier IQR treatment |
| **🔗 Joins** | Ch. 19 | left/right/inner/full/semi/anti joins |
| **📈 Visualize** | Ch. 1, 9 | 8 geoms + distribution analysis + scatter matrix + facets + time series |
| **🎨 Plot Editor** | Ch. 1, 9 | Fine-tune aesthetics, log scales, reference lines, trend lines, full ggplot2 code |
| **⚙️ Advanced** | Ch. 3 | case_when, if_else, across (8 functions), lag/lead, rank variants, add_count |

### 🔧 Pipeline
Every operation generates ggplot2/dplyr R code using the modern `|>` pipe. Download as `.R` script.
    """)
