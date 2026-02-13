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

st.set_page_config(page_title="Data Transformer Studio Pro", page_icon="", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .main { background-color: #f8f9fb; font-family: 'Inter', sans-serif; }
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
</style>
""", unsafe_allow_html=True)

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

defaults = {
    'data': None, 'original_df': None,
    'pipeline_steps': [], 'viz_code_blocks': [],
    'custom_theme': '', 'step_counter': 0,
    'data_snapshots': [], 'joined_datasets': {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


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


st.markdown("""
<div class='hero-banner'>
    <h1>📊 Data Transformer Studio Pro</h1>
    <p>R4DS 2e Complete Workflow — Import → Tidy → Transform → Visualize → Communicate — Live R Pipeline</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    with st.expander("Theme & Colors", expanded=False):
        theme_source = st.radio("Theme source", ["Built-in", "Custom R Code"], key="theme_src")
        if theme_source == "Custom R Code":
            tc = st.text_area("Paste R theme", height=80, key="cust_th")
            if tc: st.session_state.custom_theme = tc; st.success("✓")
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
            st.success(f"✓ {dup.shape[0]:,}×{dup.shape[1]}")
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
    with c2:
        if st.button("Reset", use_container_width=True):
            if st.session_state.original_df is not None:
                for k in ['pipeline_steps', 'viz_code_blocks', 'data_snapshots']:
                    st.session_state[k] = []
                st.session_state.step_counter = 0
                st.session_state.data = st.session_state.original_df.copy()
                st.rerun()
    st.write(f"**Steps:** {len(st.session_state.pipeline_steps)}")

if st.session_state.data is not None:
    df = st.session_state.data
    tab_eda, tab_transform, tab_tidy, tab_strings, tab_clean, tab_viz, tab_editor, tab_pipeline, tab_export = st.tabs([
        "EDA", "Transform", "Tidy/Join", "Str/Date/Factor",
        "Clean", "Visualize", "Plot Editor", "R Pipeline", "Export"
    ])

    with tab_eda:
        st.markdown("<div class='section-header'>Exploratory Data Analysis (R4DS 2e Ch. 2 & 10)</div>", unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1: st.markdown(f"<div class='stat-card'><h3>{df.shape[0]:,}</h3><p>Rows</p></div>", unsafe_allow_html=True)
        with c2: st.markdown(f"<div class='stat-card'><h3>{df.shape[1]}</h3><p>Cols</p></div>", unsafe_allow_html=True)
        with c3: st.markdown(f"<div class='stat-card'><h3>{df.isnull().sum().sum():,}</h3><p>Missing</p></div>", unsafe_allow_html=True)
        with c4: st.markdown(f"<div class='stat-card'><h3>{df.duplicated().sum():,}</h3><p>Dupes</p></div>", unsafe_allow_html=True)
        with c5: st.markdown(f"<div class='stat-card'><h3>{df.memory_usage(deep=True).sum()/1024**2:.1f}</h3><p>MB</p></div>", unsafe_allow_html=True)
        with st.expander("glimpse()", expanded=True):
            info = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.astype(str), 'NonNull': df.notnull().sum(),
                                'Null%': (df.isnull().sum()/len(df)*100).round(1), 'Unique': df.nunique(),
                                'Sample': [str(df[c].dropna().iloc[0])[:40] if df[c].notnull().any() else 'NA' for c in df.columns]})
            st.dataframe(info, use_container_width=True, height=280)
        with st.expander("summary()"):
            nc = df.select_dtypes(include=['number']).columns.tolist()
            if nc:
                desc = df[nc].describe().T; desc['IQR'] = desc['75%'] - desc['25%']; desc['skew'] = df[nc].skew()
                st.dataframe(desc.round(3), use_container_width=True)
        with st.expander("Distributions"):
            if HAS_PLOTLY:
                dc = st.selectbox("Variable", df.columns.tolist(), key="eda_dc")
                if df[dc].dtype in ['int64','float64']:
                    c1,c2 = st.columns(2)
                    with c1:
                        fig = px.histogram(df, x=dc, nbins=40, template="plotly_white", color_discrete_sequence=["#4a6cf7"])
                        fig.update_layout(height=300, margin=dict(t=20,b=20)); st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        fig = px.box(df, y=dc, template="plotly_white", color_discrete_sequence=["#4a6cf7"])
                        fig.update_layout(height=300, margin=dict(t=20,b=20)); st.plotly_chart(fig, use_container_width=True)
                else:
                    cts = df[dc].value_counts().head(20)
                    fig = px.bar(x=cts.index, y=cts.values, template="plotly_white", color_discrete_sequence=["#4a6cf7"])
                    fig.update_layout(height=300); st.plotly_chart(fig, use_container_width=True)
        with st.expander("Correlation"):
            nc = df.select_dtypes(include=['number']).columns.tolist()
            if HAS_PLOTLY and len(nc) >= 2:
                corr = df[nc].corr()
                fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu', zmid=0, text=corr.values.round(2), texttemplate='%{text}'))
                fig.update_layout(height=450, template="plotly_white"); st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df.head(15), use_container_width=True, height=250)

    with tab_transform:
        st.markdown("<div class='section-header'>Transform (R4DS 2e Ch. 3 — dplyr)</div>", unsafe_allow_html=True)
        sub_r, sub_c, sub_g, sub_a = st.tabs(["Rows", "Columns", "Groups", "Advanced"])
        with sub_r:
            st.markdown("<span class='verb-tag'>filter()</span>", unsafe_allow_html=True)
            fc = st.selectbox("Column", ["—"] + df.columns.tolist(), key="tf_fc")
            if fc != "—":
                if df[fc].dtype == 'object':
                    vals = st.multiselect("Values", sorted(df[fc].dropna().unique().tolist()), key="tf_fv")
                    neg = st.checkbox("Negate", key="tf_fn")
                    if vals and st.button("Apply", key="tf_fb"):
                        save_snapshot()
                        if neg: st.session_state.data = st.session_state.data[~st.session_state.data[fc].isin(vals)]
                        else: st.session_state.data = st.session_state.data[st.session_state.data[fc].isin(vals)]
                        vs = ", ".join([f'"{v}"' for v in vals])
                        add_step('transform', f'Filter {fc}', f'filter({"!" if neg else ""}{fc} %in% c({vs}))')
                        st.rerun()
                else:
                    c1,c2,c3 = st.columns(3)
                    with c1: op = st.selectbox("Op", [">=","<=","==","!=",">","<","between"], key="tf_fop")
                    with c2: v1 = st.number_input("Val1", value=float(df[fc].min()), key="tf_v1")
                    with c3: v2 = st.number_input("Val2", value=float(df[fc].max()), key="tf_v2") if op=="between" else None
                    if st.button("Apply", key="tf_fnb"):
                        save_snapshot()
                        if op=="between":
                            st.session_state.data = st.session_state.data[(st.session_state.data[fc]>=v1)&(st.session_state.data[fc]<=v2)]
                            add_step('transform', f'Filter {fc}', f'filter(between({fc}, {v1}, {v2}))')
                        else:
                            st.session_state.data = st.session_state.data[st.session_state.data[fc].apply(lambda x: eval(f"x {op} {v1}"))]
                            add_step('transform', f'Filter {fc}{op}{v1}', f'filter({fc} {op} {v1})')
                        st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>distinct()</span>", unsafe_allow_html=True)
            dcc = st.multiselect("By (empty=all)", df.columns.tolist(), key="tf_dc")
            if st.button("distinct()", key="tf_db"):
                save_snapshot(); b=len(st.session_state.data)
                st.session_state.data = st.session_state.data.drop_duplicates(subset=dcc if dcc else None)
                cs = ", ".join(dcc) if dcc else ""
                add_step('transform', 'distinct', f'distinct({cs}{"" if not cs else ", .keep_all = TRUE"})'); st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>arrange()</span>", unsafe_allow_html=True)
            sc = st.multiselect("Sort by", df.columns.tolist(), key="tf_sc")
            if sc:
                asc = st.radio("Dir", ["Asc","Desc"], horizontal=True, key="tf_sd")=="Asc"
                if st.button("arrange()", key="tf_sb"):
                    save_snapshot(); st.session_state.data = st.session_state.data.sort_values(sc, ascending=asc)
                    cs = ", ".join([c if asc else f"desc({c})" for c in sc])
                    add_step('transform', f'Sort', f'arrange({cs})'); st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>slice()</span>", unsafe_allow_html=True)
            sl = st.radio("Variant", ["head","tail","sample","min","max"], horizontal=True, key="tf_sl")
            sn = st.number_input("n", value=5, min_value=1, key="tf_sn")
            ncc = df.select_dtypes(include=['number']).columns.tolist()
            so = st.selectbox("Order col", ncc, key="tf_so") if sl in ["min","max"] and ncc else None
            if st.button("slice()", key="tf_slb"):
                save_snapshot(); n=int(sn)
                if sl=="head": st.session_state.data=st.session_state.data.head(n); rc=f'slice_head(n={n})'
                elif sl=="tail": st.session_state.data=st.session_state.data.tail(n); rc=f'slice_tail(n={n})'
                elif sl=="sample": st.session_state.data=st.session_state.data.sample(min(n,len(st.session_state.data))); rc=f'slice_sample(n={n})'
                elif sl=="min" and so: st.session_state.data=st.session_state.data.nsmallest(n,so); rc=f'slice_min({so}, n={n})'
                elif sl=="max" and so: st.session_state.data=st.session_state.data.nlargest(n,so); rc=f'slice_max({so}, n={n})'
                else: st.session_state.data=st.session_state.data.head(n); rc=f'slice_head(n={n})'
                add_step('transform', f'slice {sl} n={n}', rc); st.rerun()

        with sub_c:
            st.markdown("<span class='verb-tag'>select()</span>", unsafe_allow_html=True)
            sm = st.radio("Mode", ["Keep","Drop"], horizontal=True, key="tf_sm")
            if sm=="Keep":
                kc = st.multiselect("Keep", df.columns.tolist(), default=df.columns.tolist(), key="tf_kc")
                if kc and len(kc)<len(df.columns) and st.button("select()", key="tf_kb"):
                    save_snapshot(); st.session_state.data=st.session_state.data[kc]
                    add_step('transform', f'Select {len(kc)}', f'select({", ".join(kc)})'); st.rerun()
            else:
                dc2 = st.multiselect("Drop", df.columns.tolist(), key="tf_dc2")
                if dc2 and st.button("Apply", key="tf_db2"):
                    save_snapshot(); st.session_state.data=st.session_state.data.drop(columns=dc2)
                    add_step('transform', f'Drop {len(dc2)}', f'select(-c({", ".join(dc2)}))'); st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>rename()</span>", unsafe_allow_html=True)
            rf = st.selectbox("Col", df.columns.tolist(), key="tf_rf")
            rt = st.text_input("New", rf, key="tf_rt")
            if st.button("rename()", key="tf_rb") and rt!=rf and rt:
                save_snapshot(); st.session_state.data=st.session_state.data.rename(columns={rf:rt})
                add_step('transform', f'{rf}→{rt}', f'rename({rt} = {rf})'); st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>mutate()</span>", unsafe_allow_html=True)
            mn = st.text_input("New col name", key="tf_mn")
            mt = st.selectbox("Op", ["col ± col","col ± const","log/sqrt/abs","lag/lead","rank","custom expr"], key="tf_mt")
            ncc = df.select_dtypes(include=['number']).columns.tolist()
            if mn and ncc:
                if mt=="col ± col" and len(ncc)>=2:
                    c1,c2,c3 = st.columns(3)
                    with c1: a=st.selectbox("A",ncc,key="m_a")
                    with c2: op=st.selectbox("Op",["+","-","*","/"],key="m_op")
                    with c3: b=st.selectbox("B",ncc,key="m_b")
                    if st.button("Create",key="m1"):
                        save_snapshot(); st.session_state.data[mn]=eval(f"st.session_state.data['{a}']{op}st.session_state.data['{b}']")
                        add_step('transform',f'{mn}={a}{op}{b}',f'mutate({mn} = {a} {op} {b})'); st.rerun()
                elif mt=="col ± const":
                    c1,c2,c3=st.columns(3)
                    with c1: a=st.selectbox("Col",ncc,key="m_ac")
                    with c2: op=st.selectbox("Op",["+","-","*","/","**"],key="m_aop")
                    with c3: v=st.number_input("Val",1.0,key="m_av")
                    if st.button("Create",key="m2"):
                        save_snapshot(); st.session_state.data[mn]=eval(f"st.session_state.data['{a}']{op}{v}")
                        rop="^" if op=="**" else op
                        add_step('transform',f'{mn}={a}{rop}{v}',f'mutate({mn} = {a} {rop} {v})'); st.rerun()
                elif mt=="log/sqrt/abs":
                    c1,c2=st.columns(2)
                    with c1: a=st.selectbox("Col",ncc,key="m_fc")
                    with c2: fn=st.selectbox("Fn",["log","log2","log10","sqrt","abs","exp"],key="m_fn")
                    if st.button("Create",key="m3"):
                        save_snapshot()
                        fns={"log":np.log,"log2":np.log2,"log10":np.log10,"sqrt":np.sqrt,"abs":np.abs,"exp":np.exp}
                        if fn.startswith("log"): st.session_state.data[mn]=fns[fn](st.session_state.data[a].clip(lower=0.001))
                        elif fn=="sqrt": st.session_state.data[mn]=fns[fn](st.session_state.data[a].clip(lower=0))
                        else: st.session_state.data[mn]=fns[fn](st.session_state.data[a])
                        add_step('transform',f'{mn}={fn}({a})',f'mutate({mn} = {fn}({a}))'); st.rerun()
                elif mt=="lag/lead":
                    c1,c2,c3=st.columns(3)
                    with c1: a=st.selectbox("Col",ncc,key="m_lc")
                    with c2: lf=st.selectbox("Fn",["lag","lead"],key="m_lf")
                    with c3: ln=st.number_input("n",1,key="m_ln")
                    if st.button("Create",key="m4"):
                        save_snapshot()
                        st.session_state.data[mn]=st.session_state.data[a].shift(int(ln) if lf=="lag" else -int(ln))
                        add_step('transform',f'{mn}={lf}({a})',f'mutate({mn} = {lf}({a}, n = {int(ln)}))'); st.rerun()
                elif mt=="rank":
                    c1,c2=st.columns(2)
                    with c1: a=st.selectbox("Col",ncc,key="m_rc")
                    with c2: rf2=st.selectbox("Fn",["percent_rank","row_number","min_rank","dense_rank","ntile(4)"],key="m_rf")
                    if st.button("Create",key="m5"):
                        save_snapshot()
                        if rf2=="percent_rank": st.session_state.data[mn]=st.session_state.data[a].rank(pct=True)
                        elif rf2=="row_number": st.session_state.data[mn]=range(1,len(st.session_state.data)+1)
                        elif rf2=="min_rank": st.session_state.data[mn]=st.session_state.data[a].rank(method='min')
                        elif rf2=="dense_rank": st.session_state.data[mn]=st.session_state.data[a].rank(method='dense')
                        elif rf2=="ntile(4)": st.session_state.data[mn]=pd.qcut(st.session_state.data[a],4,labels=False)+1
                        add_step('transform',f'{mn}={rf2}({a})',f'mutate({mn} = {rf2}({a}))'); st.rerun()
                elif mt=="custom expr":
                    expr=st.text_input("Python expr (use df['col'])",key="m_expr")
                    if expr and st.button("Create",key="m6"):
                        save_snapshot()
                        try:
                            st.session_state.data[mn]=eval(expr,{"df":st.session_state.data,"np":np,"pd":pd})
                            add_step('transform',f'{mn} custom',f'mutate({mn} = ...)'); st.rerun()
                        except Exception as e: st.error(str(e))

        with sub_g:
            st.markdown("<span class='verb-tag'>group_by()</span> + <span class='verb-tag'>summarize()</span>", unsafe_allow_html=True)
            gc = st.multiselect("Group by", df.columns.tolist(), key="tf_gc")
            if gc:
                ncc = df.select_dtypes(include=['number']).columns.tolist()
                ns = st.number_input("# summaries", 1, 5, 1, key="tf_ns")
                sums = []
                for i in range(int(ns)):
                    c1,c2,c3=st.columns(3)
                    with c1: ac=st.selectbox(f"Col{i+1}",ncc,key=f"g_ac{i}")
                    with c2: af=st.selectbox(f"Fn{i+1}",["mean","sum","median","min","max","sd","n","n_distinct"],key=f"g_af{i}")
                    with c3: an=st.text_input(f"Name{i+1}",f"{af}_{ac}",key=f"g_an{i}")
                    sums.append((ac,af,an))
                if st.button("summarize()",key="tf_gb"):
                    save_snapshot()
                    agg={}; rp=[]
                    fm={"mean":"mean","sum":"sum","median":"median","min":"min","max":"max","sd":"std","n":"count","n_distinct":"nunique"}
                    for ac,af,an in sums:
                        agg[an]=pd.NamedAgg(column=ac,aggfunc=fm[af])
                        if af=="n": rp.append(f"{an} = n()")
                        elif af=="n_distinct": rp.append(f"{an} = n_distinct({ac})")
                        else: rp.append(f"{an} = {af}({ac}, na.rm = TRUE)")
                    st.session_state.data=st.session_state.data.groupby(gc).agg(**agg).reset_index()
                    gs=", ".join(gc); rs=", ".join(rp)
                    add_step('transform',f'Summarize by {gs}',f'group_by({gs}) |>\n  summarize({rs}, .groups = "drop")'); st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>count()</span>", unsafe_allow_html=True)
            cc=st.multiselect("Count by",df.columns.tolist(),key="tf_cc")
            srt=st.checkbox("Sort",True,key="tf_cs2")
            if cc and st.button("count()",key="tf_cb"):
                save_snapshot()
                st.session_state.data=st.session_state.data.groupby(cc).size().reset_index(name='n')
                if srt: st.session_state.data=st.session_state.data.sort_values('n',ascending=False)
                add_step('transform',f'Count',f'count({", ".join(cc)}, sort = {"TRUE" if srt else "FALSE"})'); st.rerun()

        with sub_a:
            st.markdown("<span class='verb-tag'>case_when()</span>", unsafe_allow_html=True)
            cw=st.text_input("New col",key="tf_cw")
            cws=st.selectbox("Based on",df.columns.tolist(),key="tf_cws")
            nc2=st.number_input("# conditions",2,6,3,key="tf_nc")
            conds=[]
            for i in range(int(nc2)):
                c1,c2,c3=st.columns(3)
                with c1: cm=st.selectbox(f"Op{i+1}",["<","<=",">=",">","=="],key=f"cw_cm{i}")
                with c2: th=st.number_input(f"Val{i+1}",key=f"cw_th{i}")
                with c3: lb=st.text_input(f"Label{i+1}",f"Cat_{i+1}",key=f"cw_lb{i}")
                conds.append((cm,th,lb))
            if cw and st.button("case_when()",key="tf_cwb"):
                save_snapshot()
                result=pd.Series(conds[-1][2],index=st.session_state.data.index)
                rp=[]
                for cm,th,lb in conds[:-1]:
                    mask=st.session_state.data[cws].apply(lambda x,c=cm,t=th:eval(f"x {c} {t}"))
                    result=result.where(~mask,lb); rp.append(f'{cws} {cm} {th} ~ "{lb}"')
                rp.append(f'.default = "{conds[-1][2]}"')
                st.session_state.data[cw]=result
                add_step('transform',f'case_when→{cw}',f'mutate({cw} = case_when({", ".join(rp)}))'); st.rerun()
            st.markdown("---")
            st.markdown("<span class='verb-tag'>across()</span>", unsafe_allow_html=True)
            ac3=st.multiselect("Cols",df.select_dtypes(include=['number']).columns.tolist(),key="tf_ac2")
            afn=st.selectbox("Fn",["scale(0-1)","z-score","log","sqrt","round(2)","replace_na(0)","abs","cumsum"],key="tf_afn")
            if ac3 and st.button("across()",key="tf_ab"):
                save_snapshot()
                for c in ac3:
                    if c in st.session_state.data.columns and st.session_state.data[c].dtype in ['int64','float64']:
                        if afn=="scale(0-1)":
                            mn2,mx=st.session_state.data[c].min(),st.session_state.data[c].max()
                            if mx!=mn2: st.session_state.data[c]=(st.session_state.data[c]-mn2)/(mx-mn2)
                        elif afn=="z-score":
                            m,s=st.session_state.data[c].mean(),st.session_state.data[c].std()
                            if s: st.session_state.data[c]=(st.session_state.data[c]-m)/s
                        elif afn=="log": st.session_state.data[c]=np.log(st.session_state.data[c].clip(lower=0.001))
                        elif afn=="sqrt": st.session_state.data[c]=np.sqrt(st.session_state.data[c].clip(lower=0))
                        elif afn=="round(2)": st.session_state.data[c]=st.session_state.data[c].round(2)
                        elif afn=="replace_na(0)": st.session_state.data[c]=st.session_state.data[c].fillna(0)
                        elif afn=="abs": st.session_state.data[c]=st.session_state.data[c].abs()
                        elif afn=="cumsum": st.session_state.data[c]=st.session_state.data[c].cumsum()
                fn2=afn.split("(")[0].replace("-","_")
                add_step('transform',f'across({fn2})',f'mutate(across(c({", ".join(ac3)}), ~{fn2}(.x)))'); st.rerun()
        st.markdown("---"); st.dataframe(st.session_state.data, use_container_width=True, height=250)
        st.write(f"**{st.session_state.data.shape[0]:,} × {st.session_state.data.shape[1]}**")

    with tab_tidy:
        st.markdown("<div class='section-header'>📐 Tidy & Joins (R4DS 2e Ch. 5 & 19)</div>", unsafe_allow_html=True)
        sub_pv, sub_sp, sub_jn = st.tabs(["Pivot", "Separate/Unite", "Joins"])
        with sub_pv:
            pd2 = st.radio("Dir", ["pivot_longer (wide→long)", "pivot_wider (long→wide)"], horizontal=True, key="td_pd")
            if "longer" in pd2:
                pc = st.multiselect("Cols to pivot", df.columns.tolist(), key="td_pc")
                nt = st.text_input("names_to", "name", key="td_nt"); vt = st.text_input("values_to", "value", key="td_vt")
                if pc and st.button("pivot_longer()", key="td_plb"):
                    save_snapshot(); idc = [c for c in st.session_state.data.columns if c not in pc]
                    st.session_state.data = st.session_state.data.melt(id_vars=idc, value_vars=pc, var_name=nt, value_name=vt)
                    add_step('tidy', f'pivot_longer', f'pivot_longer(cols = c({", ".join(pc)}), names_to = "{nt}", values_to = "{vt}")'); st.rerun()
            else:
                nf = st.selectbox("names_from", df.columns.tolist(), key="td_nf")
                vf = st.selectbox("values_from", [c for c in df.columns if c != nf], key="td_vf")
                if st.button("pivot_wider()", key="td_pwb"):
                    save_snapshot(); idx = [c for c in st.session_state.data.columns if c not in [nf, vf]]
                    try:
                        st.session_state.data = st.session_state.data.pivot_table(index=idx, columns=nf, values=vf, aggfunc='first').reset_index()
                        st.session_state.data.columns.name = None
                        if isinstance(st.session_state.data.columns, pd.MultiIndex):
                            st.session_state.data.columns = ['_'.join(str(x) for x in c).strip('_') for c in st.session_state.data.columns]
                        add_step('tidy', 'pivot_wider', f'pivot_wider(names_from = {nf}, values_from = {vf})'); st.rerun()
                    except Exception as e: st.error(str(e))
        with sub_sp:
            sa = st.radio("Action", ["separate()", "unite()"], horizontal=True, key="td_sa")
            if sa == "separate()":
                strc = df.select_dtypes(include=['object']).columns.tolist()
                if strc:
                    sc2 = st.selectbox("Col", strc, key="td_sc2"); sep = st.text_input("Sep", "_", key="td_sep")
                    sample = str(df[sc2].dropna().iloc[0]) if df[sc2].notnull().any() else ""
                    parts = sample.split(sep); st.write(f"Preview: `{sample}` → {parts}")
                    nn = st.text_input("New names (comma)", ", ".join([f"p{i+1}" for i in range(len(parts))]), key="td_nn")
                    if st.button("separate()", key="td_sb"):
                        save_snapshot(); names = [n.strip() for n in nn.split(",")]
                        sp = st.session_state.data[sc2].str.split(sep, expand=True)
                        for i, n in enumerate(names):
                            if i < sp.shape[1]: st.session_state.data[n] = sp[i]
                        st.session_state.data = st.session_state.data.drop(columns=[sc2])
                        ns = ", ".join([f'"{n}"' for n in names])
                        add_step('tidy', f'Separate {sc2}', f'separate({sc2}, into = c({ns}), sep = "{sep}")'); st.rerun()
            else:
                uc = st.multiselect("Cols", df.columns.tolist(), key="td_uc")
                un = st.text_input("New name", "combined", key="td_un"); us = st.text_input("Sep", "_", key="td_us")
                rm = st.checkbox("Remove originals", True, key="td_rm")
                if uc and st.button("unite()", key="td_ub"):
                    save_snapshot()
                    st.session_state.data[un] = st.session_state.data[uc].astype(str).agg(us.join, axis=1)
                    if rm: st.session_state.data = st.session_state.data.drop(columns=uc)
                    add_step('tidy', f'Unite→{un}', f'unite("{un}", {", ".join(uc)}, sep = "{us}")'); st.rerun()
        with sub_jn:
            if st.session_state.joined_datasets:
                jd = st.selectbox("With", list(st.session_state.joined_datasets.keys()), key="td_jd")
                jt = st.selectbox("Type", ["left_join","right_join","inner_join","full_join","semi_join","anti_join"], key="td_jt")
                df2 = st.session_state.joined_datasets[jd]
                common = list(set(df.columns) & set(df2.columns))
                if common:
                    jb = st.multiselect("By", common, default=common[:1], key="td_jb")
                    st.dataframe(df2.head(3), use_container_width=True)
                    if jb and st.button(f"{jt}()", key="td_jb2"):
                        save_snapshot()
                        hm = {"left_join":"left","right_join":"right","inner_join":"inner","full_join":"outer"}
                        if jt in hm: st.session_state.data = st.session_state.data.merge(df2, on=jb, how=hm[jt], suffixes=('.x','.y'))
                        elif jt == "semi_join": st.session_state.data = st.session_state.data[st.session_state.data[jb[0]].isin(df2[jb[0]])]
                        elif jt == "anti_join": st.session_state.data = st.session_state.data[~st.session_state.data[jb[0]].isin(df2[jb[0]])]
                        bs = ", ".join([f'"{b}"' for b in jb])
                        add_step('tidy', jt, f'{jt}({jd}, by = c({bs}))'); st.rerun()
            else: st.info("Upload second dataset in sidebar")
        st.markdown("---"); st.dataframe(st.session_state.data, use_container_width=True, height=200)

    with tab_strings:
        st.markdown("<div class='section-header'>📝 Strings, Dates & Factors (R4DS 2e Ch. 14-17)</div>", unsafe_allow_html=True)
        sub_s, sub_d, sub_f = st.tabs(["Strings", "Dates", "Factors"])
        with sub_s:
            strc = df.select_dtypes(include=['object']).columns.tolist()
            if strc:
                sc3 = st.selectbox("Col", strc, key="s_sc")
                sop = st.selectbox("Op", ["str_to_upper","str_to_lower","str_to_title","str_trim","str_squish",
                    "str_replace","str_detect (filter)","str_extract (regex)","str_remove","str_length","str_sub"], key="s_op")
                if sop in ["str_to_upper","str_to_lower","str_to_title"]:
                    if st.button("Apply",key="s_case"):
                        save_snapshot()
                        fm={"str_to_upper":"upper","str_to_lower":"lower","str_to_title":"title"}
                        st.session_state.data[sc3]=getattr(st.session_state.data[sc3].str,fm[sop])()
                        add_step('string_date',f'{sop}({sc3})',f'mutate({sc3} = {sop}({sc3}))'); st.rerun()
                elif sop=="str_trim":
                    if st.button("Apply",key="s_trim"):
                        save_snapshot(); st.session_state.data[sc3]=st.session_state.data[sc3].str.strip()
                        add_step('string_date','trim',f'mutate({sc3} = str_trim({sc3}))'); st.rerun()
                elif sop=="str_squish":
                    if st.button("Apply",key="s_sq"):
                        save_snapshot(); st.session_state.data[sc3]=st.session_state.data[sc3].str.strip().str.replace(r'\s+',' ',regex=True)
                        add_step('string_date','squish',f'mutate({sc3} = str_squish({sc3}))'); st.rerun()
                elif sop=="str_replace":
                    pat=st.text_input("Pattern",key="s_pat"); rep=st.text_input("Replace",key="s_rep")
                    if pat and st.button("Apply",key="s_repb"):
                        save_snapshot(); st.session_state.data[sc3]=st.session_state.data[sc3].str.replace(pat,rep,regex=True)
                        add_step('string_date','replace',f'mutate({sc3} = str_replace_all({sc3}, "{pat}", "{rep}"))'); st.rerun()
                elif sop=="str_detect (filter)":
                    pat=st.text_input("Pattern",key="s_dpat"); neg=st.checkbox("Negate",key="s_dn")
                    if pat and st.button("Apply",key="s_db"):
                        save_snapshot(); mask=st.session_state.data[sc3].str.contains(pat,regex=True,na=False)
                        st.session_state.data=st.session_state.data[~mask if neg else mask]
                        add_step('string_date','detect',f'filter({"!" if neg else ""}str_detect({sc3}, "{pat}"))'); st.rerun()
                elif sop=="str_extract (regex)":
                    pat=st.text_input("Regex",key="s_epat"); en=st.text_input("New col",f"{sc3}_ext",key="s_en")
                    if pat and st.button("Apply",key="s_eb"):
                        save_snapshot(); st.session_state.data[en]=st.session_state.data[sc3].str.extract(f'({pat})',expand=False)
                        add_step('string_date','extract',f'mutate({en} = str_extract({sc3}, "{pat}"))'); st.rerun()
                elif sop=="str_remove":
                    pat=st.text_input("Pattern",key="s_rpat")
                    if pat and st.button("Apply",key="s_rb"):
                        save_snapshot(); st.session_state.data[sc3]=st.session_state.data[sc3].str.replace(pat,"",regex=True)
                        add_step('string_date','remove',f'mutate({sc3} = str_remove_all({sc3}, "{pat}"))'); st.rerun()
                elif sop=="str_length":
                    en=st.text_input("New col",f"{sc3}_len",key="s_ln")
                    if st.button("Apply",key="s_lb"):
                        save_snapshot(); st.session_state.data[en]=st.session_state.data[sc3].str.len()
                        add_step('string_date','length',f'mutate({en} = str_length({sc3}))'); st.rerun()
                elif sop=="str_sub":
                    c1,c2=st.columns(2)
                    with c1: ss=st.number_input("Start",1,key="s_ss")
                    with c2: se=st.number_input("End",5,key="s_se")
                    en=st.text_input("New col",f"{sc3}_sub",key="s_sn")
                    if st.button("Apply",key="s_sb2"):
                        save_snapshot(); st.session_state.data[en]=st.session_state.data[sc3].str[int(ss)-1:int(se)]
                        add_step('string_date','sub',f'mutate({en} = str_sub({sc3}, {int(ss)}, {int(se)}))'); st.rerun()
            else: st.info("No string columns")
        with sub_d:
            dc3=st.selectbox("Col",df.columns.tolist(),key="d_dc"); dfmt=st.selectbox("Fmt",["ymd","mdy","dmy","ymd_hms","Auto"],key="d_fmt")
            if st.button("Parse",key="d_pb"):
                save_snapshot()
                try:
                    st.session_state.data[dc3]=pd.to_datetime(st.session_state.data[dc3],infer_datetime_format=True)
                    add_step('string_date',f'Parse {dc3}',f'mutate({dc3} = {dfmt}({dc3}))'); st.success("✓")
                except Exception as e: st.error(str(e))
            dtc=df.select_dtypes(include=['datetime64']).columns.tolist()
            if dtc:
                dc4=st.selectbox("Date col",dtc,key="d_dc4")
                comps=st.multiselect("Extract",["year","month","day","wday","hour","quarter"],key="d_comps")
                if comps and st.button("Extract",key="d_eb"):
                    save_snapshot(); rp=[]
                    for comp in comps:
                        nn2=f"{dc4}_{comp}"
                        if comp=="year": st.session_state.data[nn2]=st.session_state.data[dc4].dt.year
                        elif comp=="month": st.session_state.data[nn2]=st.session_state.data[dc4].dt.month
                        elif comp=="day": st.session_state.data[nn2]=st.session_state.data[dc4].dt.day
                        elif comp=="wday": st.session_state.data[nn2]=st.session_state.data[dc4].dt.dayofweek+1
                        elif comp=="hour": st.session_state.data[nn2]=st.session_state.data[dc4].dt.hour
                        elif comp=="quarter": st.session_state.data[nn2]=st.session_state.data[dc4].dt.quarter
                        rp.append(f"{nn2} = {comp}({dc4})")
                    add_step('string_date','Extract dates',f'mutate({", ".join(rp)})'); st.rerun()
        with sub_f:
            catc=df.select_dtypes(include=['object','category']).columns.tolist()
            if catc:
                fc2=st.selectbox("Col",catc,key="f_fc")
                fop=st.selectbox("Op",["fct_infreq","fct_rev","fct_reorder","fct_lump_n","fct_recode","fct_collapse"],key="f_op")
                if fop=="fct_infreq":
                    if st.button("Apply",key="f_infreq"):
                        save_snapshot(); order=st.session_state.data[fc2].value_counts().index.tolist()
                        st.session_state.data[fc2]=pd.Categorical(st.session_state.data[fc2],categories=order,ordered=True)
                        add_step('string_date',f'fct_infreq({fc2})',f'mutate({fc2} = fct_infreq({fc2}))'); st.rerun()
                elif fop=="fct_reorder":
                    ncc=df.select_dtypes(include=['number']).columns.tolist()
                    if ncc:
                        rc2=st.selectbox("By",ncc,key="f_rc")
                        if st.button("Apply",key="f_reorder"):
                            save_snapshot()
                            meds=st.session_state.data.groupby(fc2)[rc2].median().sort_values()
                            st.session_state.data[fc2]=pd.Categorical(st.session_state.data[fc2],categories=meds.index,ordered=True)
                            add_step('string_date',f'fct_reorder',f'mutate({fc2} = fct_reorder({fc2}, {rc2}))'); st.rerun()
                elif fop=="fct_lump_n":
                    ln=st.number_input("Keep top n",5,key="f_ln")
                    if st.button("Apply",key="f_lump"):
                        save_snapshot(); top=st.session_state.data[fc2].value_counts().head(int(ln)).index
                        st.session_state.data[fc2]=st.session_state.data[fc2].where(st.session_state.data[fc2].isin(top),"Other")
                        add_step('string_date',f'fct_lump_n({fc2})',f'mutate({fc2} = fct_lump_n({fc2}, n = {ln}))'); st.rerun()
                elif fop=="fct_recode":
                    st.write(f"Levels: {df[fc2].unique().tolist()[:15]}")
                    old=st.text_input("Old",key="f_old"); new=st.text_input("New",key="f_new")
                    if old and new and st.button("Recode",key="f_rec"):
                        save_snapshot(); st.session_state.data[fc2]=st.session_state.data[fc2].replace(old,new)
                        add_step('string_date','recode',f'mutate({fc2} = fct_recode({fc2}, "{new}" = "{old}"))'); st.rerun()
                elif fop=="fct_collapse":
                    gn=st.text_input("Group name",key="f_gn")
                    gv=st.multiselect("Combine",df[fc2].unique().tolist(),key="f_gv")
                    if gn and gv and st.button("Collapse",key="f_col"):
                        save_snapshot(); st.session_state.data[fc2]=st.session_state.data[fc2].replace(gv,gn)
                        vs=", ".join([f'"{v}"' for v in gv])
                        add_step('string_date','collapse',f'mutate({fc2} = fct_collapse({fc2}, "{gn}" = c({vs})))'); st.rerun()
            else: st.info("No categorical columns")

    with tab_clean:
        st.markdown("<div class='section-header'>🧹 Data Cleaning (R4DS 2e Ch. 18)</div>", unsafe_allow_html=True)
        miss=st.session_state.data.isnull().sum(); total=miss.sum()
        if total>0:
            st.warning(f"Missing: {total:,} across {(miss>0).sum()} columns")
            st.dataframe(miss[miss>0].reset_index().rename(columns={'index':'Column',0:'Missing'}), use_container_width=True)
            act=st.radio("Action",["drop_na","Drop columns","fill (forward)","replace_na (value)","Fill mean/median"],key="c_act")
            if act=="drop_na":
                sc4=st.multiselect("Only cols (empty=all)",miss[miss>0].index.tolist(),key="c_sc4")
                if st.button("Apply",key="c_dna"):
                    save_snapshot()
                    if sc4: st.session_state.data=st.session_state.data.dropna(subset=sc4); add_step('clean','drop_na',f'drop_na({", ".join(sc4)})')
                    else: st.session_state.data=st.session_state.data.dropna(); add_step('clean','drop_na','drop_na()')
                    st.rerun()
            elif act=="Drop columns":
                dc5=st.multiselect("Drop",miss[miss>0].index.tolist(),key="c_dc5")
                if dc5 and st.button("Apply",key="c_dcb"):
                    save_snapshot(); st.session_state.data=st.session_state.data.drop(columns=dc5)
                    add_step('clean','Drop cols',f'select(-c({", ".join(dc5)}))'); st.rerun()
            elif act.startswith("fill"):
                if st.button("Apply",key="c_ffill"):
                    save_snapshot(); st.session_state.data=st.session_state.data.ffill()
                    add_step('clean','fill','fill(everything(), .direction = "down")'); st.rerun()
            elif act.startswith("replace"):
                val=st.text_input("Value","0",key="c_val")
                if st.button("Apply",key="c_rnab"):
                    save_snapshot()
                    try: st.session_state.data=st.session_state.data.fillna(float(val))
                    except: st.session_state.data=st.session_state.data.fillna(val)
                    add_step('clean',f'replace_na({val})',f'mutate(across(everything(), ~replace_na(.x, {val})))'); st.rerun()
            elif act.startswith("Fill mean"):
                fm2=st.radio("Stat",["mean","median"],horizontal=True,key="c_fm")
                if st.button("Apply",key="c_fmb"):
                    save_snapshot(); ncc=st.session_state.data.select_dtypes(include=['number']).columns
                    for c in ncc:
                        v2=st.session_state.data[c].mean() if fm2=="mean" else st.session_state.data[c].median()
                        st.session_state.data[c]=st.session_state.data[c].fillna(v2)
                    add_step('clean',f'Fill {fm2}',f'mutate(across(where(is.numeric), ~replace_na(.x, {fm2}(.x, na.rm = TRUE))))'); st.rerun()
        else: st.success("No missing values ✓")
        st.markdown("---")
        st.subheader("Type Conversion")
        tc2=st.selectbox("Col",df.columns.tolist(),key="c_tc"); tt=st.selectbox("To",["numeric","character","integer","factor"],key="c_tt")
        if st.button("Convert",key="c_tcb"):
            save_snapshot()
            try:
                if tt=="numeric": st.session_state.data[tc2]=pd.to_numeric(st.session_state.data[tc2],errors='coerce')
                elif tt=="character": st.session_state.data[tc2]=st.session_state.data[tc2].astype(str)
                elif tt=="integer": st.session_state.data[tc2]=pd.to_numeric(st.session_state.data[tc2],errors='coerce').astype('Int64')
                elif tt=="factor": st.session_state.data[tc2]=st.session_state.data[tc2].astype('category')
                add_step('clean',f'Convert {tc2}',f'mutate({tc2} = as.{tt}({tc2}))'); st.rerun()
            except Exception as e: st.error(str(e))
        st.markdown("---")
        st.subheader("Outliers (IQR)")
        ncc=st.session_state.data.select_dtypes(include=['number']).columns.tolist()
        if ncc:
            oc=st.selectbox("Col",ncc,key="c_oc")
            Q1=st.session_state.data[oc].quantile(0.25); Q3=st.session_state.data[oc].quantile(0.75); IQR=Q3-Q1
            lo,hi=Q1-1.5*IQR,Q3+1.5*IQR
            nout=((st.session_state.data[oc]<lo)|(st.session_state.data[oc]>hi)).sum()
            st.write(f"Outliers: **{nout}**")
            if nout>0:
                oa=st.radio("Action",["Remove","Cap","Flag"],horizontal=True,key="c_oa")
                if st.button("Apply",key="c_ob"):
                    save_snapshot()
                    if oa=="Remove":
                        st.session_state.data=st.session_state.data[(st.session_state.data[oc]>=lo)&(st.session_state.data[oc]<=hi)]
                        add_step('clean','Remove outliers',f'filter({oc} >= {lo:.2f} & {oc} <= {hi:.2f})')
                    elif oa=="Cap":
                        st.session_state.data[oc]=st.session_state.data[oc].clip(lo,hi)
                        add_step('clean','Cap outliers',f'mutate({oc} = pmin(pmax({oc}, {lo:.2f}), {hi:.2f}))')
                    else:
                        st.session_state.data[f'{oc}_outlier']=(st.session_state.data[oc]<lo)|(st.session_state.data[oc]>hi)
                        add_step('clean','Flag outliers',f'mutate({oc}_outlier = {oc} < {lo:.2f} | {oc} > {hi:.2f})')
                    st.rerun()
            cb = st.selectbox("Color by", ["None"] + cat_pe, key="pe_cb")
            lg = st.checkbox("Legend", True, key="pe_lg")

            try:
                fig = None
                if pt == "Bar":
                    fig = px.bar(st.session_state.data, x=xc, y=yc, color=cb if cb != "None" else None, template="plotly_white", height=ht)
                elif pt == "Line":
                    fig = px.line(st.session_state.data, x=xc, y=yc, color=cb if cb != "None" else None, markers=True, template="plotly_white", height=ht)
                elif pt == "Scatter":
                    fig = px.scatter(st.session_state.data, x=xc, y=yc, color=cb if cb != "None" else None, template="plotly_white", height=ht)
                    fig.update_traces(marker=dict(size=ms, opacity=op))
                elif pt == "Box":
                    fig = px.box(st.session_state.data, x=xc, y=yc, color=cb if cb != "None" else None, template="plotly_white", height=ht)

                if fig:
                    fig.update_layout(
                        title=dict(text=ttl, font=dict(size=fs+2, color=clr)),
                        xaxis_title=xl, yaxis_title=yl,
                        showlegend=lg, font=dict(size=fs),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    sel_theme = PUBLICATION_THEMES[theme]
                    custom_t = f" +\n  {st.session_state.custom_theme}" if st.session_state.custom_theme else ""
                    color_aes = f", color = {cb}" if cb != "None" else ""
                    geom_map = {"Bar": "geom_col", "Line": "geom_line", "Scatter": "geom_point", "Box": "geom_boxplot"}

                    r_code = f"""# {ttl}
ggplot(data_clean, aes(x = {xc}, y = {yc}{color_aes})) +
  {geom_map[pt]}(size = {ms/10:.1f}, alpha = {op}, fill = "{clr}") +
  labs(
    title = "{ttl}",
    x = "{xl}",
    y = "{yl}"{f',{chr(10)}    caption = "{cap}"' if cap else ""}
  ) +
  {sel_theme["r"]} +
  theme(
    plot.title = element_text(size = {fs+2}, face = "bold", color = "{clr}"),
    axis.text = element_text(size = {fs-1}),
    legend.position = "{'bottom' if lg else 'none'}"
  ){custom_t}"""

                    st.markdown("**Generated R Code (Publication-Ready):**")
                    st.code(r_code, language="r")

                    if st.button("Save plot to pipeline", key="pe_save"):
                        st.session_state.viz_code_blocks.append(r_code)
                        add_step('viz', f'{pt} plot: {ttl}', r_code)
                        st.success("✓ Saved")

            except Exception as e:
                st.error(f"Error: {e}")

    # ════════════════════════════════════════════════════
    # TAB: R PIPELINE (smart code generation)
    # ════════════════════════════════════════════════════
    with tab_pipeline:
        st.markdown("<div class='section-header'>R Pipeline — Smart Code Generation</div>", unsafe_allow_html=True)
        st.caption("All your operations, tracked and compiled into a single reproducible R script using the native pipe |>")

        if not st.session_state.pipeline_steps:
            st.info("No operations recorded yet. Start transforming, cleaning, or visualizing your data.")
        else:
            # Step tracker
            st.subheader("Operation History")
            for i, s in enumerate(st.session_state.pipeline_steps):
                cat_icons = {'transform': '', 'string_date': '', 'clean': '', 'tidy': '', 'viz': ''}
                icon = cat_icons.get(s['category'], '⚙️')
                st.markdown(f"<div class='pipeline-step'><strong>{icon} Step {s['order']}</strong> [{s['category']}] — {s['description']}</div>", unsafe_allow_html=True)

            st.markdown("---")

            # Code output options
            st.subheader("Generated R Code")

            code_mode = st.radio("Output mode", [
                "Complete Pipeline (wrangling + viz)",
                "Wrangling Only (Transform + Clean + Tidy + String/Date)",
                "Visualization Only",
                "Separate Blocks (by category)"
            ], key="p_mode")

            if code_mode.startswith("🔗"):
                code = build_r_pipeline()
                st.code(code, language="r")

            elif code_mode.startswith(""):
                code = build_r_pipeline(categories=['transform', 'string_date', 'clean', 'tidy'])
                st.code(code, language="r")

            elif code_mode.startswith(""):
                code = build_r_pipeline(categories=['viz'])
                st.code(code, language="r")

            elif code_mode.startswith(""):
                cat_map = {
                    'transform': 'Transform', 'string_date': '📝 String/Date',
                    'clean': 'Cleaning', 'tidy': '📐 Tidy/Reshape', 'viz': '📊 Visualization'
                }
                cats_present = list(set(s['category'] for s in st.session_state.pipeline_steps))
                for cat in cats_present:
                    steps = [s for s in st.session_state.pipeline_steps if s['category'] == cat]
                    with st.expander(f"{cat_map.get(cat, cat)} ({len(steps)} steps)", expanded=True):
                        if cat == 'viz':
                            for s in steps:
                                st.code(s['r_code'], language="r")
                        else:
                            pipe_parts = [s['r_code'] for s in steps]
                            code = "data |>\n  " + " |>\n  ".join(pipe_parts)
                            st.code(code, language="r")

            # Download
            st.markdown("---")
            full_code = build_r_pipeline()
            st.download_button("Download Complete R Script", full_code,
                             f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M')}.R", "text/plain")

    # ════════════════════════════════════════════════════
    # TAB: EXPORT
    # ════════════════════════════════════════════════════
    with tab_export:
        st.markdown("<div class='section-header'>Export Data & Scripts</div>", unsafe_allow_html=True)

        st.subheader("Data Downloads")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.download_button("📄 Transformed CSV", st.session_state.data.to_csv(index=False),
                             f"data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv")
        with c2:
            # Excel
            buf = io.BytesIO()
            try:
                st.session_state.data.to_excel(buf, index=False)
                st.download_button("Excel (.xlsx)", buf.getvalue(),
                                 f"data_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                                 "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except:
                st.info("Install openpyxl for Excel export")
        with c3:
            st.download_button("📋 Complete R Script", build_r_pipeline(),
                             f"analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.R", "text/plain")

        st.markdown("---")

        # Theme reference
        st.subheader("ggplot2 Theme Reference")
        theme_ref = """# ══════════════════════════════════════════════════
# Publication-Ready ggplot2 Themes Reference
# ══════════════════════════════════════════════════

# ── Built-in (ggplot2) ────────────────────────────
theme_minimal()     # Clean, no background
theme_bw()          # White background, grid
theme_classic()     # Classic axes, no grid
theme_linedraw()    # Black borders
theme_light()       # Light grey background
theme_void()        # Nothing but data

# ── hrbrthemes (install.packages("hrbrthemes")) ──
theme_ipsum()       # Typography-focused
theme_ft_rc()       # Financial Times style

# ── ggthemes (install.packages("ggthemes")) ──────
theme_economist()   # The Economist
theme_wsj()         # Wall Street Journal
theme_fivethirtyeight()  # FiveThirtyEight
theme_tufte()       # Edward Tufte minimal
theme_solarized()   # Solarized palette

# ── ggpubr (install.packages("ggpubr")) ──────────
theme_pubr()        # Publication-ready

# ── Custom Theme Example ─────────────────────────
my_theme <- theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16, color = "#1a1f36"),
    plot.subtitle = element_text(size = 12, color = "#6b7280"),
    plot.caption = element_text(size = 9, color = "#9ca3af"),
    axis.title = element_text(size = 12, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    plot.margin = margin(15, 15, 15, 15)
  )

# ── Color Scales ──────────────────────────────────
scale_color_viridis_d()      # Discrete viridis
scale_fill_brewer(palette = "Set1")  # ColorBrewer
scale_color_manual(values = c("#E64B35", "#4DBBD5", "#00A087"))

# ── Saving Publication-Quality ────────────────────
ggsave("plot.pdf", width = 8, height = 6, dpi = 300)
ggsave("plot.png", width = 8, height = 6, dpi = 300)
"""
        with st.expander("Theme Reference Code"):
            st.code(theme_ref, language="r")
            st.download_button("Download Theme Reference", theme_ref, "ggplot2_themes.R", "text/plain")

        st.markdown("---")
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data, use_container_width=True, height=350)
        st.write(f"**Final: {st.session_state.data.shape[0]:,} rows × {st.session_state.data.shape[1]} columns**")

else:
    st.info("Upload a CSV or Excel file in the sidebar to get started")
    st.markdown("---")
    st.markdown("""
    ### Features (R4DS 2e Complete Coverage)

    | Category | R4DS Chapter | Operations |
    |----------|-------------|------------|
    | **EDA** | Ch. 2, 10 | glimpse, summary, distributions, correlation, missing patterns |
    | **Transform** | Ch. 3 | filter, arrange, select, mutate, group_by, summarize, count, slice, distinct, relocate, rename |
    | **Tidy** | Ch. 5 | pivot_longer, pivot_wider, separate, unite |
    | **Strings** | Ch. 14-15 | str_to_upper/lower, str_trim, str_replace, str_detect, str_extract, str_remove, regex |
    | **Dates** | Ch. 17 | ymd/mdy/dmy parsing, year/month/day/wday extraction |
    | **Factors** | Ch. 16 | fct_infreq, fct_reorder, fct_lump, fct_recode, fct_collapse, fct_rev |
    | **Missing** | Ch. 18 | drop_na, fill, replace_na, type conversion |
    | **Joins** | Ch. 19 | left/right/inner/full/semi/anti joins |
    | **Visualize** | Ch. 1, 9 | bar, line, scatter, histogram, box, violin, density, heatmap, facet_wrap |
    | **Advanced** | Ch. 3 | case_when, if_else, across, cumulative fns, lag/lead, ranking |

    ### 🔧 Smart Pipeline
    - Every operation is tracked with its R equivalent
    - Generate **wrangling-only**, **viz-only**, or **combined** R scripts
    - Uses modern `|>` pipe operator
    - Automatic library detection
    - 14 publication-ready themes from ggplot2, hrbrthemes, ggthemes, ggpubr
    - Undo/Reset support with data snapshots
    """)
