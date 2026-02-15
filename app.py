"""
Product Line Profitability & Margin Performance Analysis
Nassau Candy Distributor - Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Nassau Candy | Profitability Analysis",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# 🎨 Modern UI Implementation (CSS & Theme)
# -----------------------------------------------------------------------------
def load_css(file_name="styles.css"):
    with open(file_name, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# -----------------------------------------------------------------------------
# 💾 Data Loading & Processing
# -----------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Load and preprocess the Nassau Candy dataset."""
    csv_path = Path(__file__).parent / "Nassau Candy Distributor.csv"
    df = pd.read_csv(csv_path)

    # Parse dates
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

    # Numeric conversion
    numeric_cols = ["Sales", "Units", "Gross Profit", "Cost"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cleaning
    df = df.dropna(subset=["Sales", "Gross Profit", "Cost"])
    df = df[df["Sales"] > 0]
    df = df[df["Units"].fillna(0) >= 0]
    
    return df

def apply_filters(df, date_range, divisions, margin_min, product_search):
    filtered = df.copy()
    if date_range:
        filtered = filtered[
            (filtered["Order Date"] >= pd.Timestamp(date_range[0])) &
            (filtered["Order Date"] <= pd.Timestamp(date_range[1]))
        ]
    if divisions:
        filtered = filtered[filtered["Division"].isin(divisions)]
    if product_search:
        mask = filtered["Product Name"].str.contains(product_search, case=False, na=False) | \
               filtered["Product ID"].str.contains(product_search, case=False, na=False)
        filtered = filtered[mask]
    return filtered

def compute_product_metrics(df):
    agg = df.groupby(["Product ID", "Product Name", "Division"]).agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Cost=("Cost", "sum"),
        Units=("Units", "sum"),
    ).reset_index()
    agg["Gross_Margin_Pct"] = np.where(agg["Sales"] > 0, agg["Gross_Profit"] / agg["Sales"] * 100, 0)
    agg["Profit_Per_Unit"] = np.where(agg["Units"] > 0, agg["Gross_Profit"] / agg["Units"], 0)
    return agg

def compute_division_metrics(df):
    agg = df.groupby("Division").agg(
        Sales=("Sales", "sum"),
        Gross_Profit=("Gross Profit", "sum"),
        Cost=("Cost", "sum"),
        Units=("Units", "sum"),
    ).reset_index()
    agg["Gross_Margin_Pct"] = np.where(agg["Sales"] > 0, agg["Gross_Profit"] / agg["Sales"] * 100, 0)
    agg["Profit_Per_Unit"] = np.where(agg["Units"] > 0, agg["Gross_Profit"] / agg["Units"], 0)
    return agg

def pareto_data(series, threshold=0.8):
    sorted_series = series.sort_values(ascending=False).reset_index(drop=True)
    cumsum = sorted_series.cumsum()
    total = sorted_series.sum()
    cum_pct = cumsum / total * 100 if total > 0 else cumsum * 0
    n_to_80 = (cum_pct >= threshold * 100).idxmax() + 1 if (cum_pct >= threshold * 100).any() else len(cum_pct)
    return sorted_series, cum_pct, n_to_80

# -----------------------------------------------------------------------------
# 📊 Visuals Configuration (Plotly)
# -----------------------------------------------------------------------------
COLOR_PALETTE = ["#e94560", "#F4A261", "#2A9D8F", "#E9C46A", "#264653", "#8D99AE"]
GRADIENT_COLOR = "#e94560"

def get_chart_layout(title=""):
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", color="#e0e0e0"),
        title=dict(text=title, font=dict(size=18, color="#fff")),
        margin=dict(l=50, r=40, t=60, b=40),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(255,255,255,0.08)"),
        hovermode="x unified",
        colorway=COLOR_PALETTE,
    )

PLOTLY_CONFIG = {
    "displayModeBar": False,
    "scrollZoom": False,
}

# -----------------------------------------------------------------------------
# 🏠 Main App Logic
# -----------------------------------------------------------------------------

# Load data
df_raw = load_data()
min_date = df_raw["Order Date"].min().date()
max_date = df_raw["Order Date"].max().date()

# --- Sidebar Filters ---
with st.sidebar:
    st.markdown("### 🛠️ settings")
    date_range = st.date_input("📅 Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    date_range_tuple = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
    
    st.markdown("---")
    st.markdown("### 🏢 Divisions")
    all_divisions = sorted(df_raw["Division"].dropna().unique().tolist())
    divisions = st.multiselect("Select Divisions", options=all_divisions, default=all_divisions)
    
    st.markdown("---")
    st.markdown("### 🔎 Product Details")
    margin_threshold = st.slider("Min Gross Margin (%)", 0.0, 100.0, 0.0, 1.0)
    product_search = st.text_input("Product Search (Name/ID)", "")
    
    st.markdown("---")
    st.caption("v2.0 • Nassau Candy Analysis\n\nDesigned & Developed by Pranav PV")

# Apply filters
df = apply_filters(df_raw, date_range_tuple, divisions, margin_threshold, product_search)

# --- Header ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("Distributor Analytics Dashboard")
    st.caption("Real-time Profitability & Operational Insights")

# --- KPI Section (Top Level) ---
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

total_sales = df['Sales'].sum()
total_profit = df['Gross Profit'].sum()
overall_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
order_count = len(df)

with col1:
    st.metric("Total Revenue", f"${total_sales:,.0f}", delta="Gross Sales")
with col2:
    st.metric("Total Gross Profit", f"${total_profit:,.0f}", delta=f"{overall_margin:.1f}% Margin")
with col3:
    st.metric("Avg Order Value", f"${total_sales/order_count:,.0f}" if order_count else "$0")
with col4:
    st.metric("Total Orders", f"{order_count:,}", delta="Filtered")

st.markdown("---")

# Precompute metrics
prod_metrics = compute_product_metrics(df)
prod_metrics = prod_metrics[prod_metrics["Gross_Margin_Pct"] >= margin_threshold]
div_metrics = compute_division_metrics(df)

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈 Product Performance", 
    "🏭 Division Analysis", 
    "⚠️ Risk Diagnostics", 
    "🎯 Pareto Analysis",
    "📉 Forecasting & Trends",
    "🤖 Smart AI Insights"
])

# -----------------------------------------------------------------------------
# TAB 1: Product Performance
# -----------------------------------------------------------------------------
with tab1:
    col_t1_1, col_t1_2 = st.columns([2, 1])
    
    with col_t1_1:
        st.subheader("Top Products by Gross Margin")
        top_margin = prod_metrics.nlargest(15, "Gross_Margin_Pct").sort_values("Gross_Margin_Pct", ascending=True)
        
        fig_bar = px.bar(
            top_margin, 
            y="Product Name", x="Gross_Margin_Pct", 
            color="Division",
            orientation="h",
            color_discrete_sequence=COLOR_PALETTE
        )
        fig_bar.update_layout(
            **get_chart_layout("Gross Margin % Leaderboard"),
            height=500,
            xaxis_title="Gross Margin (%)",
            yaxis_title=None,
            legend=dict(orientation="h", y=1.1)
        )
        fig_bar.update_traces(hovertemplate="<b>%{y}</b><br>Margin: %{x:.1f}%<extra></extra>")
        st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG)

    with col_t1_2:
        st.subheader("Profit Composition")
        top_profit = prod_metrics.nlargest(8, "Gross_Profit")
        
        fig_pie = px.pie(
            top_profit, 
            values="Gross_Profit", 
            names="Product Name",
            color_discrete_sequence=COLOR_PALETTE,
            hole=0.6
        )
        layout_pie = get_chart_layout("Profit Share (Top 8)")
        layout_pie.update(dict(
            height=500,
            showlegend=False,
            margin=dict(t=80, b=40, l=20, r=20)
        ))
        fig_pie.update_layout(layout_pie)
        fig_pie.update_traces(textinfo="percent+label", textposition="inside")
        st.plotly_chart(fig_pie, use_container_width=True, config=PLOTLY_CONFIG)

    # Detailed Data Table
    with st.expander("📝 View Detailed Product Metrics", expanded=False):
        st.dataframe(
            prod_metrics.sort_values("Sales", ascending=False).style.format({
                "Sales": "${:,.0f}", "Gross_Profit": "${:,.0f}", 
                "Cost": "${:,.0f}", "Gross_Margin_Pct": "{:.1f}%", 
                "Profit_Per_Unit": "${:.2f}"
            }),
            use_container_width=True,
            height=400
        )

# -----------------------------------------------------------------------------
# TAB 2: Division Analysis
# -----------------------------------------------------------------------------
with tab2:
    col_t2_1, col_t2_2 = st.columns(2)
    
    with col_t2_1:
        fig_combo = go.Figure()
        fig_combo.add_trace(go.Bar(
            name="Revenue", 
            x=div_metrics["Division"], 
            y=div_metrics["Sales"],
            marker_color=COLOR_PALETTE[0],
            marker_line_width=0
        ))
        fig_combo.add_trace(go.Scatter(
            name="Margin %",
            x=div_metrics["Division"],
            y=div_metrics["Gross_Margin_Pct"],
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#ffffff", width=3, shape='spline'),
            marker=dict(size=8, color="#ffffff")
        ))
        
        layout = get_chart_layout("Revenue vs Margin Efficiency")
        layout.update(
            yaxis2=dict(
                title="Margin %", 
                overlaying="y", 
                side="right", 
                showgrid=False,
                range=[0, 100]
            ),
            yaxis=dict(title="Revenue ($)", showgrid=True),
            legend=dict(orientation="h", y=1.1),
            height=450
        )
        fig_combo.update_layout(layout)
        st.plotly_chart(fig_combo, use_container_width=True, config=PLOTLY_CONFIG)

    with col_t2_2:
        if len(df) > 0:
            # Aggregate top products per division for cleaner view
            df_tree = df.groupby(["Division", "Product Name"])["Gross Profit"].sum().reset_index()
            # Only take top 50 products overall to avoid lag
            df_tree = df_tree.nlargest(50, "Gross Profit")
            
            fig_sun = px.sunburst(
                df_tree, 
                path=["Division", "Product Name"], 
                values="Gross Profit",
                color="Division",
                color_discrete_sequence=COLOR_PALETTE
            )
            fig_sun.update_layout(
                **get_chart_layout("Profit Distribution Hierarchy (Sunburst)"),
                height=500
            )
            st.plotly_chart(fig_sun, use_container_width=True, config=PLOTLY_CONFIG)

# -----------------------------------------------------------------------------
# TAB 3: Risk Diagnostics
# -----------------------------------------------------------------------------
# ... (existing code)



# -----------------------------------------------------------------------------
# TAB 3: Risk Diagnostics
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("### 🚦 Margin outlier detection")
    
    col_t3_1, col_t3_2 = st.columns([3, 1])
    with col_t3_1:
        # Scatter Plot
        fig_scat = px.scatter(
            prod_metrics, 
            x="Cost", 
            y="Sales", 
            size="Gross_Profit",
            color="Gross_Margin_Pct",
            color_continuous_scale="RdYlGn",
            hover_name="Product Name",
            hover_data={
                "Gross_Margin_Pct": ":.1f",
                "Division": True
            },
            size_max=50
        )
        
        # Add break-even line
        max_val = max(prod_metrics["Cost"].max(), prod_metrics["Sales"].max())
        fig_scat.add_shape(
            type="line", line=dict(dash="dash", color="white", width=1),
            x0=0, y0=0, x1=max_val, y1=max_val
        )
        fig_scat.add_annotation(
            x=max_val*0.9, y=max_val*0.95, text="Break-even", 
            showarrow=False, font=dict(color="white")
        )
        
        fig_scat.update_layout(
            **get_chart_layout("Cost vs Revenue Efficiency"),
            height=500,
            xaxis_title="Total Cost ($)",
            yaxis_title="Total Revenue ($)"
        )
        st.plotly_chart(fig_scat, use_container_width=True, config=PLOTLY_CONFIG)
        
    with col_t3_2:
        st.info("💡 **Bubble Size** = Total Profit\n\n**Color** = Margin %\n\nProducts below the dashed diagonal line are selling at a loss.")
        
        # Risk Table: Show All Products
        st.markdown("### 📋 All Products Margin Analysis")
        all_prods_sorted = prod_metrics.sort_values("Gross_Margin_Pct", ascending=True)
        
        st.dataframe(
            all_prods_sorted[["Product Name", "Division", "Gross_Margin_Pct", "Gross_Profit", "Sales"]].style.format({
                "Gross_Margin_Pct": "{:.1f}%",
                "Gross_Profit": "${:,.0f}",
                "Sales": "${:,.0f}"
            }),
            use_container_width=True,
            height=400
        )

# -----------------------------------------------------------------------------
# TAB 4: Pareto Analysis
# -----------------------------------------------------------------------------
with tab4:
    prod_sorted = prod_metrics.sort_values("Gross_Profit", ascending=False)
    profit_series = prod_sorted["Gross_Profit"]
    _, cum_pct, n_80 = pareto_data(profit_series, 0.8)
    
    col_p1, col_p2 = st.columns([3, 1])
    
    with col_p1:
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_pareto.add_trace(
            go.Bar(
                x=list(range(len(profit_series))), 
                y=profit_series.values, 
                name="Profit", 
                marker_color=COLOR_PALETTE[0]
            ), 
            secondary_y=False
        )
        
        fig_pareto.add_trace(
            go.Scatter(
                x=list(range(len(cum_pct))), 
                y=cum_pct.values, 
                name="Cumulative %", 
                mode="lines", 
                line=dict(color="white", width=2)
            ), 
            secondary_y=True
        )
        
        # Add 80% line
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="rgba(255,255,255,0.5)", secondary_y=True)
        
        layout = get_chart_layout(f"Profit Concentration: Top {n_80} Products = 80% Profit")
        layout.update(height=500, showlegend=True)
        fig_pareto.update_layout(layout)
        fig_pareto.update_yaxes(title_text="Profit ($)", secondary_y=False, showgrid=True, gridcolor="rgba(255,255,255,0.05)")
        fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True, showgrid=False)
        
        st.plotly_chart(fig_pareto, use_container_width=True, config=PLOTLY_CONFIG)
        
        st.markdown("### 🧠 Insight")
        st.metric("80/20 Rule", f"{n_80} Products", delta="Generate 80% Profit")
        
        st.markdown("#### 🌟 Top Contributors")
        top_performers = prod_sorted.head(n_80)
        st.dataframe(
            top_performers[["Product Name", "Gross_Profit", "Gross_Margin_Pct"]].style.format({
                "Gross_Profit": "${:,.0f}",
                "Gross_Margin_Pct": "{:.1f}%"
            }),
            use_container_width=True,
            height=300
        )

# -----------------------------------------------------------------------------
# TAB 5: Forecasting & Trends
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("### 📅 Monthly Trends & Forecast")
    
    # Prepare time-series data
    df_trend = df.copy()
    df_trend['YearMonth'] = df_trend['Order Date'].dt.to_period('M').astype(str)
    
    monthly_agg = df_trend.groupby('YearMonth').agg(
             Sales=('Sales', 'sum'),
             Profit=('Gross Profit', 'sum')
    ).reset_index()
    
    # Simple Linear Forecast (Next 3 Months)
    if len(monthly_agg) > 1:
        x = np.arange(len(monthly_agg))
        y_sales = monthly_agg['Sales'].values
        
        # Fit polynomial (degree 1 = linear)
        z = np.polyfit(x, y_sales, 1)
        p = np.poly1d(z)
        
        # Future points
        x_future = np.arange(len(monthly_agg) + 3)
        y_future = p(x_future)
        
        # Plot
        fig_trend = go.Figure()
        
        # Historical Data
        fig_trend.add_trace(go.Bar(
            x=monthly_agg['YearMonth'], 
            y=monthly_agg['Sales'], 
            name="Actual Sales",
            marker_color=COLOR_PALETTE[0]
        ))
        
        # Trend Line
        # Need accurate future dates for x-axis
        last_date = pd.to_datetime(monthly_agg['YearMonth'].max(), format='%Y-%m')
        future_dates = [monthly_agg['YearMonth'].tolist()[-1]]
        for i in range(1, 4):
            next_date = last_date + pd.DateOffset(months=i)
            future_dates.append(next_date.strftime('%Y-%m'))
            
        fig_trend.add_trace(go.Scatter(
            x=monthly_agg['YearMonth'].tolist() + future_dates[1:],
            y=y_future,
            name="Trend Forecast",
            line=dict(color="#F4A261", width=3, dash="dot")
        ))
        
        layout = get_chart_layout("Sales Trend + 3 Month Forecast")
        layout.update(height=450)
        fig_trend.update_layout(layout)
        st.plotly_chart(fig_trend, use_container_width=True, config=PLOTLY_CONFIG)
        
        # Growth Metric
        slope = z[0]
        st.metric("Estimated Monthly Growth", f"${slope:,.0f}", delta="Trend Slope")

    else:
        st.info("Not enough data points for trend analysis (need > 1 month).")

# -----------------------------------------------------------------------------
# TAB 6: Smart AI Insights
# -----------------------------------------------------------------------------
with tab6:
    st.markdown("### 🤖 Automated Business Insights")
    
    insights = []
    
    # 1. Margin Decline Check
    if len(monthly_agg) >= 2:
        last_month_margin = monthly_agg.iloc[-1]['Profit'] / monthly_agg.iloc[-1]['Sales'] * 100
        prev_month_margin = monthly_agg.iloc[-2]['Profit'] / monthly_agg.iloc[-2]['Sales'] * 100
        change = last_month_margin - prev_month_margin
        
        if change < -2:
            insights.append(f"⚠️ **Margin Alert:** Gross margin dropped by **{abs(change):.1f}%** compared to last month.")
        elif change > 2:
            insights.append(f"🚀 **Positive Trend:** Margins improved by **{change:.1f}%** month-over-month.")

    # 2. Pareto Concentration Risk
    if n_80 < 5:
        insights.append(f"⚠️ **High Concentration Risk:** Only **{n_80}** products generate 80% of your profit. Diversification is recommended.")

    # 3. Cost Flag
    high_cost_prods = prod_metrics[prod_metrics['Cost'] > prod_metrics['Sales'] * 0.95]
    if not high_cost_prods.empty:
        insights.append(f"🛑 **Loss Leaders:** Found **{len(high_cost_prods)}** products selling at near-zero or negative margin.")

    # 4. Division Dominance
    if not div_metrics.empty:
        top_div = div_metrics.sort_values('Gross_Profit', ascending=False).iloc[0]
        share = (top_div['Gross_Profit'] / total_profit) * 100
        insights.append(f"🏆 **Star Performer:** The **{top_div['Division']}** division drives **{share:.0f}%** of total company profit.")
        
        # 5. Efficiency Champion (Highest Margin Division)
        efficient_div = div_metrics.sort_values('Gross_Margin_Pct', ascending=False).iloc[0]
        insights.append(f"⚡ **Efficiency Champion:** **{efficient_div['Division']}** leads with the highest margins at **{efficient_div['Gross_Margin_Pct']:.1f}%**.")

    # 6. Top Product Contribution
    if not prod_metrics.empty:
        top_prod = prod_metrics.sort_values('Gross_Profit', ascending=False).iloc[0]
        prod_share = (top_prod['Gross_Profit'] / total_profit) * 100
        insights.append(f"💎 **Crown Jewel:** The product **{top_prod['Product Name']}** alone contributes **{prod_share:.1f}%** to your total bottom line.")

    # Display
    if insights:
        for i, text in enumerate(insights):
            st.info(text)
    else:
        st.success("✅ Operations look stable. No critical anomalies detected by AI logic.")

