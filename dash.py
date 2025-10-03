# --------------- FASTAPI ENDPOINTS FOR INTERNAL WEBSITE ---------------
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
import io

api = FastAPI()

@api.get('/kpi')
def get_kpi():
    df = pd.read_excel('assessment_ ihna.xlsx')
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    kpis = {}
    for col in numeric_cols:
        kpis[col] = {
            'total': float(df[col].sum()),
            'average': float(df[col].mean()),
            'min': float(df[col].min()),
            'max': float(df[col].max())
        }
    return JSONResponse(content=kpis)

@api.get('/chart/{col}')
def get_chart(col: str):
    df = pd.read_excel('assessment_ ihna.xlsx')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    # Bar chart of value counts for the column
    counts = df[col].value_counts()
    counts.plot.bar(ax=ax, color='#0078D4')  
    ax.set_ylabel('Count')
    ax.set_xlabel(col)
    ax.set_title(f'Bar Chart of {col}') 
    plt.xticks(rotation=45, ha='right')
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight') 
    buf.seek(0) 
    plt.close(fig)
    return StreamingResponse(buf, media_type='image/png')
# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from datetime import datetime

# --------------- PAGE CONFIG ---------------
st.set_page_config(layout="wide", page_title="Assessment IHNA Dashboard (PowerBI-style)")

# --------------- CUSTOM CSS FOR POWER BI STYLE ---------------
st.markdown("""
    <style>
        /* Navigation Header */
        .nav-header {
            background-color: #1F2C56;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            color: white;
            font-size: 22px;
            font-weight: bold;
        }
        .nav-menu {
            float: right;
            font-size: 16px;
            margin-top: -28px;
        }
        .nav-menu a {
            color: #ddd;
            margin-left: 20px;
            text-decoration: none;
        }
        .nav-menu a:hover {
            color: white;
        }

        /* KPI Tiles */
        .kpi-card {
            background-color: #f5f5f5;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            text-align: center;
            font-size: 20px;
            margin: 10px;
        }
        .kpi-title {
            font-size: 14px;
            color: #555;
            margin-bottom: 5px;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: bold;
            color: #1F2C56;
        }
    </style>
""", unsafe_allow_html=True)

# --------------- NAV HEADER ---------------
st.markdown("""
    <div class="nav-header">
        📊 Assessment Dashboard
        <div class="nav-menu">
            <a href="#">Home</a>
            <a href="#">Reports</a>
            <a href="#">Analytics</a>
            <a href="#">Settings</a>
        </div>
    </div>
""", unsafe_allow_html=True)


# --------------- KPI TILES (Example Data) ---------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Total Assessments</div><div class="kpi-value">1,245</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Completed</div><div class="kpi-value" style="color:green;">975</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Pending</div><div class="kpi-value" style="color:orange;">180</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="kpi-card"><div class="kpi-title">Failed</div><div class="kpi-value" style="color:red;">90</div></div>', unsafe_allow_html=True)


# --------------- CHART EXAMPLE (Pie) ---------------
# Dummy data for chart
data = {
    "Assessment Type": ["Knowledge Based", "Simulation Based", "Professional Experience", "Theory"],
    "Count": [50, 70, 40, 30]
}
df = pd.DataFrame(data)

colA, colB = st.columns([2,2])
with colA:
    pie_fig = px.pie(df, names="Assessment Type", values="Count", title="Assessment Distribution")
    st.plotly_chart(pie_fig, use_container_width=True)

with colB:
    bar_fig = px.bar(df, x="Assessment Type", y="Count", text="Count", title="Assessment Breakdown")
    st.plotly_chart(bar_fig, use_container_width=True)



@st.cache_data
def load_data(path="assessment_ ihna.xlsx"):
    return pd.read_excel(path)

df = load_data()   

# ---------- Helper: attempt to find columns by common patterns ----------
def find_column(df, patterns, prefer=None): 
    """Return first matching column in df for any of patterns (case-insensitive).
       patterns can be a list of substrings to search for.
       prefer: optional exact column name to prefer if present.
    """
    cols = df.columns.tolist()
    if prefer and prefer in cols:
        return prefer 
    lowered = {c.lower(): c for c in cols}
    for p in patterns:
        p_l = p.lower()
        for c_low, c_orig in lowered.items():
            if p_l == c_low or p_l in c_low or c_low in p_l: 
                return c_orig
    return None

# Common fields we might expect
course_col = find_column(df, ["course_id", "course", "course name", "courseid", "course_name"])
student_col = find_column(df, ["student_id", "student", "learner", "user_id", "user"])
assessment_col = find_column(df, ["assessment_type", "assessment", "assessment_name", "assessment type", "assessment_type_name"])
score_col = find_column(df, ["score", "marks", "mark", "grade", "score_value"])
max_score_col = find_column(df, ["max_score", "max_marks", "total_marks", "out_of"])
status_col = find_column(df, ["status", "assessment_status", "result", "pass_fail", "passed"])
submission_date_col = find_column(df, ["submission_date", "submitted_date", "submitted_on", "submission", "date_submitted", "date"])
due_date_col = find_column(df, ["due_date", "assessment_due_date", "due", "deadline"])
created_date_col = find_column(df, ["created_at", "created_on", "created", "date_created"])

# Convert potential date columns to datetime safely
for c in [submission_date_col, due_date_col, created_date_col]:
    if c and not pd.api.types.is_datetime64_any_dtype(df[c]):
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            pass

# Default group & aggregate selection (safe) 
cat_cols = df.select_dtypes(exclude="number").columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

# Sidebar - filters and controls (left pane like Power BI)
with st.sidebar:
    st.header("Filters & Settings")
    # file info
    st.markdown(f"**Rows:** {len(df):,}  \n**Columns:** {df.shape[1]}")
    st.markdown("---")

    # Top-N
    top_n = st.number_input("Top N (for top lists)", min_value=3, max_value=50, value=10, step=1)

    # Group and aggregation choices (restrict to safe options)
    group_col = st.selectbox("Group by (category)", options=cat_cols if cat_cols else df.columns.tolist(), index=0)
    agg_col = st.selectbox("Aggregate column (numeric)", options=num_cols if num_cols else df.columns.tolist(), index=0)
    agg_func = st.selectbox("Aggregation", ["sum", "mean", "count", "min", "max"])

    st.markdown("---")
    st.write("Quick filters")
    # dynamic filters: show top unique values for up to 3 categorical cols
    filter_cols = st.multiselect("Choose columns to filter", options=df.columns.tolist(), default=[])
    filters = {}
    for c in filter_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            mi, ma = float(df[c].min(skipna=True) or 0), float(df[c].max(skipna=True) or 0)
            selected = st.slider(f"{c}", mi, ma, (mi, ma))
            filters[c] = ("numeric_range", selected)
        elif pd.api.types.is_datetime64_any_dtype(df[c]):
            min_d, max_d = df[c].min(), df[c].max()
            sel = st.date_input(f"{c}", value=(min_d.date() if pd.notna(min_d) else None, max_d.date() if pd.notna(max_d) else None))
            filters[c] = ("date_range", sel)
        else:
            options = df[c].dropna().unique().tolist()
            sel = st.multiselect(f"{c}", options=options, default=options[:5] if len(options)>5 else options)
            filters[c] = ("multi", sel)

    st.markdown("---")
    st.write("Visualization options")
    show_percent = st.checkbox("Show % on donut/pie", value=True)
    st.write("---")
    st.caption("Designed to be tolerant of missing/renamed columns.")

# Apply filters from sidebar to a working copy
df_filtered = df.copy()
for c, rule in filters.items():
    typ, val = rule
    if typ == "numeric_range":
        low, high = val
        df_filtered = df_filtered[df_filtered[c].between(low, high)]
    elif typ == "date_range":
        if val and len(val) == 2 and val[0] and val[1]:
            st_0 = pd.to_datetime(val[0])
            st_1 = pd.to_datetime(val[1])
            df_filtered = df_filtered[(pd.to_datetime(df_filtered[c], errors="coerce") >= st_0) & (pd.to_datetime(df_filtered[c], errors="coerce") <= st_1)]
    elif typ == "multi":
        if val:
            df_filtered = df_filtered[df_filtered[c].isin(val)]

# ---------- Calculations / KPIs ----------
# Generic numeric KPIs
numeric_cols = df_filtered.select_dtypes(include="number").columns.tolist()

def fmt(x):
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

# KPI values to display (choose some meaningful ones if available)
total_rows = len(df_filtered)
unique_students = df_filtered[student_col].nunique() if student_col and student_col in df_filtered.columns else None
unique_courses = df_filtered[course_col].nunique() if course_col and course_col in df_filtered.columns else None

# Pass rate calculation (if we have status or score)
pass_rate = None
if status_col and status_col in df_filtered.columns:
    # assume values like 'pass','fail' or True/False
    lowered = df_filtered[status_col].astype(str).str.lower()
    pass_vals = lowered.isin(["pass", "passed", "p", "true", "yes", "y"])
    if pass_vals.sum() > 0:
        pass_rate = 100 * pass_vals.sum() / len(df_filtered)
elif score_col and score_col in df_filtered.columns and max_score_col and max_score_col in df_filtered.columns:
    # derive pass if score >= 40% of max_score (common)
    pass_rate = 100 * (df_filtered[score_col] >= 0.4 * df_filtered[max_score_col]).sum() / len(df_filtered)

# Late submissions % (if we have submission and due date)
late_pct = None
if submission_date_col and due_date_col and submission_date_col in df_filtered.columns and due_date_col in df_filtered.columns:
    tmp = df_filtered.copy()
    tmp["__sub"] = pd.to_datetime(tmp[submission_date_col], errors="coerce")
    tmp["__due"] = pd.to_datetime(tmp[due_date_col], errors="coerce")
    tmp = tmp.dropna(subset=["__sub", "__due"])
    if len(tmp) > 0:
        late_pct = 100 * (tmp["__sub"] > tmp["__due"]).sum() / len(tmp)

# Aggregate chosen metric
def aggregate_df(df_in, group_by, metric, func):
    if func == "sum":
        return df_in.groupby(group_by, as_index=False)[metric].sum()
    if func == "mean":
        return df_in.groupby(group_by, as_index=False)[metric].mean()
    if func == "count":
        return df_in.groupby(group_by, as_index=False)[metric].count()
    if func == "min":
        return df_in.groupby(group_by, as_index=False)[metric].min()
    if func == "max":
        return df_in.groupby(group_by, as_index=False)[metric].max()
    return pd.DataFrame()

agg_df = aggregate_df(df_filtered, group_col, agg_col, agg_func) if group_col in df_filtered.columns and agg_col in df_filtered.columns else pd.DataFrame()

# ---------- Layout: Header KPI row ----------
st.markdown("## Assessment IHNA Dashboard")
kpi1, kpi2, kpi3, kpi4 = st.columns([1.5,1.5,1.5,1.5])
with kpi1:
    st.metric(label="Total Rows", value=f"{total_rows:,}")
with kpi2:
    st.metric(label="Unique Students", value=f"{unique_students:,}" if unique_students is not None else "N/A")
with kpi3:
    st.metric(label="Unique Courses", value=f"{unique_courses:,}" if unique_courses is not None else "N/A")
with kpi4:
    if pass_rate is not None:
        st.metric(label="Pass Rate (%)", value=f"{pass_rate:.1f}%")
    else:
        st.metric(label="Pass Rate (%)", value="N/A")

# second KPI row for late and a sample numeric summary
kpi5, kpi6, kpi7, _ = st.columns([1.2,1.2,1.2,1])
with kpi5:
    if late_pct is not None:
        st.metric(label="Late Submissions (%)", value=f"{late_pct:.1f}%")
    else:
        st.metric(label="Late Submissions (%)", value="N/A")
with kpi6:
    if score_col:
        avg_score = df_filtered[score_col].mean() if score_col in df_filtered.columns else None
        st.metric(label=f"Avg {score_col}" if score_col else "Avg Score", value=f"{fmt(avg_score)}" if avg_score is not None else "N/A")
with kpi7:
    if agg_col:
        total_agg = df_filtered[agg_col].sum() if agg_col in df_filtered.columns else None
        st.metric(label=f"Total {agg_col}", value=f"{fmt(total_agg)}" if total_agg is not None else "N/A")

st.markdown("---")

# ---------- Main canvas: two columns style like PowerBI -->
left_col, right_col = st.columns((2, 1))

# Left: main visuals
with left_col:
    # Top N bar chart
    st.subheader(f"Top {top_n} by {agg_func} of {agg_col} (by {group_col})")
    if not agg_df.empty:
        top_df = agg_df.sort_values(agg_col, ascending=False).head(top_n)
        fig_bar = px.bar(top_df, x=group_col, y=agg_col, title=f"Top {top_n} {group_col}", labels={group_col:group_col, agg_col:agg_col})
        fig_bar.update_layout(margin=dict(l=10,r=10,t=40,b=10), height=420)
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No aggregated data available for the selected group/aggregate. Check filters or pick different columns.")

    st.markdown("### Trend / Time Series")
    # Trend chart - use submission_date_col or created_date_col if available
    date_for_trend = submission_date_col or created_date_col or find_column(df_filtered, ["date", "created", "submitted", "on"])
    if date_for_trend and date_for_trend in df_filtered.columns and agg_col in df_filtered.columns:
        # group by date (monthly)
        tmp = df_filtered.copy()
        tmp[date_for_trend] = pd.to_datetime(tmp[date_for_trend], errors="coerce")
        tmp = tmp.dropna(subset=[date_for_trend])
        tmp["period"] = tmp[date_for_trend].dt.to_period("M").dt.to_timestamp()
        trend_df = tmp.groupby("period", as_index=False)[agg_col].sum().sort_values("period")
        if len(trend_df) > 0:
            fig_trend = px.line(trend_df, x="period", y=agg_col, title=f"{agg_col} Trend Over Time ({date_for_trend})")
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Not enough time-series data to show trend.")
    else:
        st.info("No suitable date column found for trend chart (look for 'submission_date', 'created_on', etc.).")

    st.markdown("### Category Breakdown (Donut / Stacked)")
    # Donut chart using top_n categories to avoid label overlap
    if not agg_df.empty:
        top_df = agg_df.sort_values(agg_col, ascending=False).head(top_n)
        fig_donut = px.pie(top_df, names=group_col, values=agg_col, hole=0.4)
        if show_percent:
            fig_donut.update_traces(textinfo='percent+label')
        else:
            fig_donut.update_traces(textinfo='label')
        fig_donut.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    # Heatmap: cross-tab of assessment types vs courses (if both exist)
    if assessment_col and assessment_col in df_filtered.columns and course_col and course_col in df_filtered.columns:
        st.markdown("### Heatmap: Assessment Type vs Course (counts)")
        pivot = pd.crosstab(df_filtered[assessment_col].fillna("Unknown"), df_filtered[course_col].fillna("Unknown"))
        # take top rows/cols to keep heatmap readable
        pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(12).index,
                          pivot.sum(axis=0).sort_values(ascending=False).head(12).index]
        fig_heat = px.imshow(pivot, labels=dict(x="Course", y="Assessment Type", color="Count"), aspect="auto")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No assessment/course pair found for heatmap. Needs both assessment and course columns.")

with right_col:
    # Filters summary / details (like Power BI right pane)
    st.subheader("Filters & Top Lists")
    st.markdown("**Active Filters**")
    if filters:
        for k, v in filters.items():
            st.write(f"- {k}: {v[1]}")
    else:
        st.write("No extra filters applied.")

    st.markdown("---")
    # Table: Top N concepts (detailed)
    st.subheader(f"Top {top_n} Detail Table")
    if not agg_df.empty:
        st.dataframe(top_df.rename(columns={group_col: "Category", agg_col: "Value"}).reset_index(drop=True))
    else:
        st.info("No aggregated data to show in table.")

    st.markdown("---")
    # Distribution/Histogram for chosen numeric column
    st.subheader(f"Distribution: {agg_col}")
    if agg_col and agg_col in df_filtered.columns:
        fig_hist = px.histogram(df_filtered, x=agg_col, nbins=30, title=f"Distribution of {agg_col}")
        st.plotly_chart(fig_hist)
    else:
        st.info(f"No numeric column named {agg_col} found for distribution.")

    # Small summary list
    st.markdown("---")
    st.write("### Quick Insights")
    # Top courses by count
    if course_col and course_col in df_filtered.columns:
        counts = df_filtered[course_col].value_counts().head(10)
        st.write("Top courses by record count:")
        st.table(counts.rename_axis(course_col).reset_index(name="count"))
    else:
        st.info("Course column not found to list top courses.")

# ---------- Footer: raw data & download ----------
st.markdown("---")
st.subheader("Data Preview & Export")
st.write("Filtered dataset preview (first 200 rows):")
st.dataframe(df_filtered.head(200))

# allow CSV download of filtered data
@st.cache_data
def convert_df_to_csv(df_in):
    return df_in.to_csv(index=False).encode('utf-8')

csv_bytes = convert_df_to_csv(df_filtered)
st.download_button(label="Download filtered data as CSV", data=csv_bytes, file_name="assessment_ihna_filtered.csv", mime="text/csv")

st.markdown("### Notes")
st.write("""
- The dashboard is designed to be tolerant of missing/renamed columns — it attempts to find sensible defaults (e.g. `score`, `marks`, `submission_date`, `course`, `assessment_type`) and will still render what it can.
- For a true Power BI look you can embed this Streamlit app in an iframe within a Power BI report or display side-by-side with notes.
- If you want specific DAX-like measures (late penalty rules, pass thresholds, grade bands), tell me the exact column names or the business logic and I'll add them as additional KPIs / measures.
""")



