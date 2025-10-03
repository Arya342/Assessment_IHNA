# import streamlit as st
# import pandas as pd

# # Load the Excel file
# df = pd.read_excel("assessment_ ihna.xlsx")

# # Example aggregation logic from new.py
# # You can customize aggregation columns below
# st.sidebar.header("Aggregation & Grouping")
# group_col = st.sidebar.selectbox("Group by column", df.columns)
# agg_func = st.sidebar.selectbox("Aggregation function", ["sum", "mean", "count", "min", "max"])
# num_cols = df.select_dtypes(include='number').columns.tolist()
# agg_col = st.sidebar.selectbox("Aggregate column", num_cols)

# def safe_reset_index(agg_result, col):
#     # Convert Series to DataFrame for column check
#     if isinstance(agg_result, pd.Series):
#         agg_result = agg_result.to_frame()
#     if col in agg_result.columns:
#         return agg_result
#     else:
#         return agg_result.reset_index()

# if agg_func == "sum":
#     agg_df = df.groupby(group_col)[agg_col].sum()
#     agg_df = safe_reset_index(agg_df, group_col)
# elif agg_func == "mean":
#     agg_df = df.groupby(group_col)[agg_col].mean()
#     agg_df = safe_reset_index(agg_df, group_col)
# elif agg_func == "count":
#     agg_df = df.groupby(group_col)[agg_col].count()
#     agg_df = safe_reset_index(agg_df, group_col)
# elif agg_func == "min":
#     agg_df = df.groupby(group_col)[agg_col].min()
#     agg_df = safe_reset_index(agg_df, group_col)
# elif agg_func == "max":
#     agg_df = df.groupby(group_col)[agg_col].max()
#     agg_df = safe_reset_index(agg_df, group_col)
# else:
#     agg_df = pd.DataFrame()

# st.title("Assessment IHNA Dashboard")

# st.write("## Data Preview")
# st.dataframe(df)

# # Add basic interactivity: column selection and filtering
# columns = df.columns.tolist()
# selected_columns = st.multiselect("Select columns to display", columns, default=columns)
# st.dataframe(df[selected_columns])

# st.write("## Filter Data")
# for col in selected_columns:
#     if df[col].dtype == 'object':
#         options = df[col].unique().tolist()
#         selected_option = st.selectbox(f"Filter {col}", ["All"] + options)
#         if selected_option != "All":
#             df = df[df[col] == selected_option]
#     elif pd.api.types.is_numeric_dtype(df[col]):
#         min_val, max_val = df[col].min(), df[col].max()
#         selected_range = st.slider(f"Filter {col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
#         df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]

# st.write("## Filtered Data")
# st.dataframe(df[selected_columns])

# # Aggregation Table
# st.write(f"## Aggregated Data: {agg_func} of {agg_col} by {group_col}")
# st.dataframe(agg_df)

# # Visualizations
# st.write("## Charts")
# st.bar_chart(agg_df, x=group_col, y=agg_col)
# # Prepare DataFrame for line chart: remove duplicate group_col if present
# line_df = agg_df.copy()
# # Ensure group_col is present for indexing
# if group_col not in line_df.columns:
#     line_df = line_df.reset_index()
# # Only plot if both columns exist
# if group_col in line_df.columns and agg_col in line_df.columns:
#     st.line_chart(line_df.set_index(group_col)[agg_col])
# elif agg_col in line_df.columns:
#     st.line_chart(line_df[agg_col])
# else:
#     st.warning(f"Cannot plot line chart: '{group_col}' or '{agg_col}' not found in aggregated data.")
# st.write("### Pie Chart")
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.pie(agg_df[agg_col], labels=agg_df[group_col], autopct='%1.1f%%')
# st.pyplot(fig)



import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
df = pd.read_excel("assessment_ ihna.xlsx")

# Sidebar aggregation settings
st.sidebar.header("Aggregation & Grouping")

# Group by categorical columns only
cat_cols = df.select_dtypes(exclude='number').columns.tolist()
group_col = st.sidebar.selectbox("Group by column", cat_cols)

# Aggregate over numeric columns only
num_cols = df.select_dtypes(include='number').columns.tolist()
agg_col = st.sidebar.selectbox("Aggregate column", num_cols)

agg_func = st.sidebar.selectbox("Aggregation function", ["sum", "mean", "count", "min", "max"])


# --- Perform aggregation (always with as_index=False) ---
if agg_func == "sum":
    agg_df = df.groupby(group_col, as_index=False)[agg_col].sum()
elif agg_func == "mean":
    agg_df = df.groupby(group_col, as_index=False)[agg_col].mean() 
elif agg_func == "count":
    agg_df = df.groupby(group_col, as_index=False)[agg_col].count()
elif agg_func == "min":
    agg_df = df.groupby(group_col, as_index=False)[agg_col].min()
elif agg_func == "max":
    agg_df = df.groupby(group_col, as_index=False)[agg_col].max()
else:
    agg_df = pd.DataFrame()

# --- Dashboard ---
st.title("Assessment IHNA Dashboard")

# --- KPI cards: Show total, average, min, max for each numeric column ---
st.write("### Key Performance Indicators (KPIs)")
for num_col in df.select_dtypes(include='number').columns.tolist():
    total = df[num_col].sum()
    avg = df[num_col].mean()
    min_val = df[num_col].min()
    max_val = df[num_col].max()
    kpi_labels = [f"Total {num_col}", f"Average {num_col}", f"Min {num_col}", f"Max {num_col}"]
    kpi_values = [f"{total:,.2f}", f"{avg:,.2f}", f"{min_val:,.2f}", f"{max_val:,.2f}"]
    cols = st.columns(4, gap="medium")
    for i, col in enumerate(cols):
        col.markdown(f"""
<style>
.powerbi-card {{
    margin-right: 24px;
}}
</style>
<style>
.powerbi-card {{
    background:#fff;
    border-radius:10px;
    box-shadow:0 2px 8px rgba(128,128,128,0.15);
    width: 180px;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding:12px 8px 8px 8px;
    position:relative;
    margin-bottom:14px;
    transition: border 0.2s, transform 0.2s;
    border: 0.6px solid transparent;
}}
.powerbi-card:hover {{
    box-shadow:none;
    border: 0.6px solid #0078D4;
    transform: translateY(-2px) scale(1.01);
}}
</style>
<div class='powerbi-card'>
    <span style='font-size:14px;font-weight:600;color:#444;'>{kpi_labels[i]}</span><br>
    <span style='font-size:28px;font-weight:700;color:#0078D4;letter-spacing:1px;'>{kpi_values[i]}</span>
</div>
""", unsafe_allow_html=True)

# st.header("Dataset Columns")
# st.write(list(df.columns))
# st.header("Sample Data (First 10 Rows)")
# st.dataframe(df.head(10))
# st.write("## Data Preview")
# st.dataframe(df)


# --- Column selection ---
columns = df.columns.tolist()
selected_columns = st.multiselect("Select columns to display", columns, default=columns)
st.dataframe(df[selected_columns]) 

# --- Filtering ---
st.write("## Filter Data")
for col in selected_columns:
    if df[col].dtype == 'object':
        options = df[col].unique().tolist()
        selected_option = st.selectbox(f"Filter {col}", ["All"] + options)
        if selected_option != "All":
            df = df[df[col] == selected_option]
    elif pd.api.types.is_numeric_dtype(df[col]):
        min_val, max_val = df[col].min(), df[col].max()
        selected_range = st.slider(
            f"Filter {col}",
            float(min_val),
            float(max_val),
            (float(min_val), float(max_val))
        )
        df = df[(df[col] >= selected_range[0]) & (df[col] <= selected_range[1])]

st.write("## Filtered Data")
st.dataframe(df[selected_columns])

# --- Aggregated Table ---

# Show only Top 10 concepts
top_n = 10
agg_df_sorted = agg_df.sort_values(agg_col, ascending=False)
top_agg_df = agg_df_sorted.head(top_n)

st.write(f"## Top {top_n} Concepts: {agg_func} of {agg_col} by {group_col}")
st.dataframe(top_agg_df)

# --- Visualizations ---
st.write("## Charts")
# --- Business Growth/Loss Visualizations ---
st.subheader("Business Growth & Loss Visualizations")

# Identify likely time/date columns
date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# Trend chart: If date column exists, plot total of each numeric column over time
if len(date_cols) > 0 and len(numeric_cols) > 0:
    for num_col in numeric_cols:
        st.write(f"### Trend of {num_col} over {date_cols[0]}")
        trend_df = df.groupby(date_cols[0], as_index=False)[num_col].sum()
        fig = px.line(trend_df, x=date_cols[0], y=num_col, title=f"{num_col} Trend Over Time")
        st.plotly_chart(fig)

# KPI cards: Show total, average, min, max for each numeric column
st.write("### Key Performance Indicators (KPIs)")
for num_col in numeric_cols:
    total = df[num_col].sum()
    avg = df[num_col].mean()
    min_val = df[num_col].min()
    max_val = df[num_col].max()
    kpi_labels = [f"Total {num_col}", f"Average {num_col}", f"Min {num_col}", f"Max {num_col}"]
    kpi_values = [f"{total:,.2f}", f"{avg:,.2f}", f"{min_val:,.2f}", f"{max_val:,.2f}"]
    cols = st.columns(4, gap="medium")
    for i, col in enumerate(cols):
     col.markdown(f"""
<style>
.powerbi-card {{
    margin-right: 24px;
}}
</style>
<style>
.powerbi-card {{
    background:#fff;
    border-radius:10px;
    box-shadow:0 2px 8px rgba(128,128,128,0.15);
    width: 180px;
    height: 110px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding:12px 8px 8px 8px;
    position:relative;
    margin-bottom:14px;
    transition: border 0.2s, transform 0.2s;
    border: 0.6px solid transparent;
}}
.powerbi-card:hover {{
    box-shadow:none;
    border: 0.6px solid #0078D4;
    transform: translateY(-2px) scale(1.01);
}}
</style>
<div class='powerbi-card'>
    <span style='font-size:14px;font-weight:600;color:#444;'>{kpi_labels[i]}</span><br>
    <span style='font-size:28px;font-weight:700;color:#0078D4;letter-spacing:1px;'>{kpi_values[i]}</span>
</div>
""", unsafe_allow_html=True)

# Growth/Loss chart: Compare first and last value for each numeric column (if date column exists)
if len(date_cols) > 0 and len(numeric_cols) > 0:
    for num_col in numeric_cols:
        sorted_df = df.sort_values(date_cols[0])
        first = sorted_df[num_col].iloc[0]
        last = sorted_df[num_col].iloc[-1]
        change = last - first
        pct_change = (change / first * 100) if first != 0 else 0
        st.write(f"### Growth/Loss for {num_col}")
        st.write(f"First: {first:,.2f}, Last: {last:,.2f}, Change: {change:,.2f} ({pct_change:.2f}%)")
        fig = px.bar(x=["First", "Last"], y=[first, last], title=f"Growth/Loss of {num_col}")
        st.plotly_chart(fig)


# ✅ Bar Chart (Top 10)
if group_col in top_agg_df.columns and agg_col in top_agg_df.columns:
    st.bar_chart(top_agg_df, x=group_col, y=agg_col)
else:
    st.warning("Cannot plot bar chart: missing required columns.")




# ✅  Line Chart (Top 10)
# if group_col in top_agg_df.columns and agg_col in top_agg_df.columns:
#     try:
#         st.line_chart(top_agg_df.set_index(group_col)[[agg_col]])
#     except KeyError:
#         st.warning(f"Cannot set index: '{group_col}' not found in aggregated data.")
# elif agg_col in top_agg_df.columns:
#     st.line_chart(top_agg_df[[agg_col]])
# else:
#     st.warning(f"Cannot plot line chart: '{group_col}' or '{agg_col}' not found in aggregated data.")



# ✅ Donut Chart (Top N Concepts)
# st.write(f"### Donut Chart (Top {top_n} Concepts)")
# if group_col in top_agg_df.columns and agg_col in top_agg_df.columns:
#     fig, ax = plt.subplots()
#     wedges, texts, autotexts = ax.pie(
#         top_agg_df[agg_col], labels=top_agg_df[group_col], autopct='%1.1f%%', wedgeprops=dict(width=0.4)
#     )
#     # Draw circle for donut effect
#     centre_circle = plt.Circle((0,0),0.30,fc='white')
#     fig.gca().add_artist(centre_circle)
#     ax.axis('equal')
#     st.pyplot(fig)


import plotly.express as px

st.write(f"### Interactive Donut Chart (Top {top_n} Concepts)")
if group_col in top_agg_df.columns and agg_col in top_agg_df.columns:
    fig = px.pie(top_agg_df, names=group_col, values=agg_col, hole=0.4)
    st.plotly_chart(fig)

# st.write("### Bar Chart Instead of Pie")

# if group_col in top_agg_df.columns and agg_col in top_agg_df.columns:
#     fig, ax = plt.subplots(figsize=(10, 6))
#     top_agg_df.sort_values(agg_col, ascending=True).plot(
#         kind="barh", x=group_col, y=agg_col, ax=ax, legend=False
#     )
#     ax.set_xlabel(agg_col)
#     ax.set_ylabel(group_col)
#     st.pyplot(fig)
