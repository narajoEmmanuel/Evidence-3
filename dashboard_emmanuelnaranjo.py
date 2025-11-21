# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Whirlpool Pricing Analytics",
    page_icon="🌀",
    layout="wide"
)

# =========================================================
# Dummy data generation
# =========================================================


@st.cache_data
def generate_dummy_data():
    np.random.seed(42)
    weeks = pd.date_range("2024-01-01", periods=24, freq="W")
    skus = ["WFR5200D", "AFR2110G", "8MWTW2023WPM"]
    partners = ["WALMART", "LIVERPOOL", "CHEDRAUI"]

    rows = []
    for sku in skus:
        base_demand = np.random.randint(400, 900)
        base_price = np.random.randint(6500, 9500)
        for partner in partners:
            partner_factor = np.random.uniform(0.8, 1.2)
            for d in weeks:
                price = base_price * np.random.uniform(0.9, 1.1)
                promo_flag = np.random.choice([0, 1], p=[0.7, 0.3])
                promo_effect = 1.2 if promo_flag == 1 else 1.0
                noise = np.random.normal(0, 60)
                sell_out = max(
                    0,
                    base_demand * partner_factor * promo_effect
                    * (9500 / price)
                    + noise
                )
                margin = np.random.uniform(0.12, 0.28)
                category_share = np.random.uniform(0.02, 0.15)

                rows.append(
                    {
                        "week": d,
                        "sku": sku,
                        "partner": partner,
                        "price": round(price, 0),
                        "promo_flag": promo_flag,
                        "sell_out": round(sell_out, 0),
                        "margin": margin,
                        "category_share": category_share,
                    }
                )
    df = pd.DataFrame(rows)
    return df


data = generate_dummy_data()

# Global filters options
all_skus = sorted(data["sku"].unique())
all_partners = sorted(data["partner"].unique())
min_date = data["week"].min()
max_date = data["week"].max()

# =========================================================
# Sidebar
# =========================================================
st.sidebar.markdown("## Whirlpool Pricing Analytics")

page = st.sidebar.radio(
    "Select view",
    ["Internal Pricing Dashboard", "Trading Partner Performance", "ML Price Engine"],
)

st.sidebar.markdown("---")

selected_sku = st.sidebar.selectbox("SKU", all_skus, index=0)
selected_partner = st.sidebar.selectbox(
    "Trading Partner", ["ALL"] + all_partners, index=0)

date_range = st.sidebar.slider(
    "Date range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    format="YYYY MMM",
)

st.sidebar.markdown("---")
st.sidebar.caption("Dummy prototype for academic purposes")

# Filter data
mask = (data["week"] >= pd.to_datetime(date_range[0])) & (
    data["week"] <= pd.to_datetime(date_range[1])
)
filtered = data[mask]

if selected_partner != "ALL":
    filtered = filtered[filtered["partner"] == selected_partner]

filtered = filtered[filtered["sku"] == selected_sku]

if filtered.empty:
    st.warning(
        "No data available for the selected filters. Please adjust filters.")
    st.stop()

# =========================================================
# Helper functions
# =========================================================


def compute_kpis(df):
    df_sorted = df.sort_values("week")
    total_sell_out = df_sorted["sell_out"].sum()

    if df_sorted["week"].nunique() >= 2:
        last_week = df_sorted["week"].max()
        prev_week = df_sorted["week"].nlargest(2).iloc[-1]

        last_val = df_sorted.loc[df_sorted["week"]
                                 == last_week, "sell_out"].sum()
        prev_val = df_sorted.loc[df_sorted["week"]
                                 == prev_week, "sell_out"].sum()

        if prev_val > 0:
            wow = (last_val - prev_val) / prev_val
        else:
            wow = 0.0
    else:
        wow = 0.0

    forecast_accuracy = np.random.uniform(0.86, 0.96)

    return total_sell_out, wow, forecast_accuracy


def format_pct(x):
    return f"{x * 100:.1f}%"


def format_int(x):
    return f"{int(x):,}"


# =========================================================
# Internal Pricing Dashboard
# =========================================================
if page == "Internal Pricing Dashboard":
    st.title("Internal Pricing Dashboard")
    st.caption("Weekly performance overview and price driven insights")

    kpi_sell_out, kpi_wow, kpi_acc = compute_kpis(filtered)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Weekly Sell Out (units)",
            value=format_int(kpi_sell_out),
        )
    with col2:
        st.metric(
            "Week over Week Growth",
            value=format_pct(kpi_wow),
        )
    with col3:
        st.metric(
            "Forecast Accuracy (dummy)",
            value=format_pct(kpi_acc),
        )

    st.markdown("---")

    col_main, col_side = st.columns([2.4, 1.6])

    with col_main:
        st.subheader("Sales history for selected SKU")

        fig_sales = px.line(
            filtered,
            x="week",
            y="sell_out",
            title="Weekly Sell Out",
            markers=True,
            labels={"week": "Week", "sell_out": "Units sold"},
        )
        fig_sales.update_layout(margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_sales, use_container_width=True)

        st.subheader("Price and promotion timeline")
        fig_price = go.Figure()

        fig_price.add_trace(
            go.Bar(
                x=filtered["week"],
                y=filtered["price"],
                name="Price",
                marker_color="#C69214",
            )
        )

        promo_weeks = filtered[filtered["promo_flag"] == 1]["week"]
        promo_prices = filtered[filtered["promo_flag"] == 1]["price"]

        fig_price.add_trace(
            go.Scatter(
                x=promo_weeks,
                y=promo_prices,
                mode="markers",
                name="Promotion",
                marker=dict(size=10, color="#555555"),
            )
        )

        fig_price.update_layout(
            title="Price and promotion activity",
            xaxis_title="Week",
            yaxis_title="Price (MXN)",
            margin=dict(l=0, r=0, t=40, b=0),
        )

        st.plotly_chart(fig_price, use_container_width=True)

    with col_side:
        st.subheader("Elasticity snapshot (dummy)")

        elasticity_df = filtered.copy()
        elasticity_df["price_normalized"] = (
            elasticity_df["price"] / elasticity_df["price"].mean()
        )
        elasticity_df["sell_out_normalized"] = (
            elasticity_df["sell_out"] / elasticity_df["sell_out"].mean()
        )

        fig_el = px.scatter(
            elasticity_df,
            x="price_normalized",
            y="sell_out_normalized",
            trendline="ols",
            labels={
                "price_normalized": "Price (normalized)",
                "sell_out_normalized": "Demand (normalized)",
            },
        )
        fig_el.update_layout(
            title="Price vs demand (normalized, dummy)",
            margin=dict(l=0, r=0, t=40, b=0),
        )
        st.plotly_chart(fig_el, use_container_width=True)

        elasticity_dummy = -1.25
        if elasticity_dummy <= -1:
            elasticity_label = "Elastic"
        else:
            elasticity_label = "Inelastic"

        st.markdown(
            f"**Elasticity estimate (dummy):** {elasticity_dummy:.2f} ({elasticity_label})"
        )

        st.markdown("---")
        st.subheader("ML price recommendation (dummy)")

        avg_price = filtered["price"].mean()
        rec_price = avg_price * np.random.uniform(0.96, 1.04)
        lower = rec_price * 0.96
        upper = rec_price * 1.04

        st.markdown(f"**Recommended price:** {rec_price:,.0f} MXN")
        st.markdown(f"Optimal range: {lower:,.0f} to {upper:,.0f} MXN")
        st.caption(
            "Recommendation based on historical patterns, price sensitivity and promotions (dummy logic)."
        )

    st.markdown("---")
    st.subheader("Analyst notes (example interpretation)")
    st.write(
        """
- Recent promotions increased sell out but also increased price volatility.
- The current price is close to the upper bound of the recommended range, so aggressive discounts might not be necessary.
- Elasticity appears moderately elastic, so large price changes are likely to affect demand significantly.
"""
    )

# =========================================================
# Trading Partner Performance
# =========================================================
elif page == "Trading Partner Performance":
    st.title("Trading Partner Performance Dashboard")
    st.caption("Partner specific view of SKU performance")

    partner_filter_df = data[(data["sku"] == selected_sku)]
    partner_agg = (
        partner_filter_df.groupby("partner")
        .agg(
            sell_out=("sell_out", "sum"),
            avg_price=("price", "mean"),
            avg_margin=("margin", "mean"),
            avg_share=("category_share", "mean"),
        )
        .reset_index()
    )

    total_sell_out_all = partner_agg["sell_out"].sum()
    partner_agg["sell_out_share"] = partner_agg["sell_out"] / \
        total_sell_out_all

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Sell Out (all partners)",
                  format_int(total_sell_out_all))
    with col2:
        if selected_partner != "ALL":
            sel_row = partner_agg[partner_agg["partner"] == selected_partner]
            if not sel_row.empty:
                st.metric(
                    f"Sell Out for {selected_partner}",
                    format_int(sel_row["sell_out"].iloc[0]),
                )
            else:
                st.metric("Sell Out for selected partner", "n.a.")
        else:
            st.metric("Unique partners", partner_agg["partner"].nunique())
    with col3:
        st.metric(
            "Average category share (all partners)",
            format_pct(partner_agg["avg_share"].mean()),
        )

    st.markdown("---")

    st.subheader("Week over week comparison")

    if selected_partner == "ALL":
        wo_df = partner_filter_df.copy()
    else:
        wo_df = partner_filter_df[
            partner_filter_df["partner"].isin([selected_partner])
        ].copy()

    wo_agg = (
        wo_df.groupby(["week", "partner"])
        .agg(sell_out=("sell_out", "sum"))
        .reset_index()
    )

    fig = px.line(
        wo_agg,
        x="week",
        y="sell_out",
        color="partner",
        markers=True,
        labels={"week": "Week", "sell_out": "Units sold", "partner": "Partner"},
        title="Week over week sell out by partner",
    )
    fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Partner comparison table (dummy)")

    comparison_df = partner_agg.copy()
    comparison_df["status"] = np.where(
        comparison_df["sell_out_share"] > comparison_df["sell_out_share"].mean(),
        "🟢",
        "🟡",
    )

    comparison_df_display = comparison_df[
        ["partner", "sell_out", "avg_price",
            "avg_margin", "sell_out_share", "status"]
    ].copy()
    comparison_df_display.rename(
        columns={
            "partner": "Partner",
            "sell_out": "Sell Out (units)",
            "avg_price": "Average price (MXN)",
            "avg_margin": "Average margin",
            "sell_out_share": "Sell out share",
            "status": "Status",
        },
        inplace=True,
    )

    comparison_df_display["Sell Out (units)"] = comparison_df_display[
        "Sell Out (units)"
    ].apply(format_int)
    comparison_df_display["Average price (MXN)"] = comparison_df_display[
        "Average price (MXN)"
    ].apply(lambda x: f"{x:,.0f}")
    comparison_df_display["Average margin"] = comparison_df_display[
        "Average margin"
    ].apply(format_pct)
    comparison_df_display["Sell out share"] = comparison_df_display[
        "Sell out share"
    ].apply(format_pct)

    st.dataframe(comparison_df_display, use_container_width=True)

    st.markdown("---")
    st.subheader("Analyst notes (example interpretation)")
    st.write(
        """
- Green status indicates partners with above average contribution to total sell out.
- Price levels and margins vary by partner, which may justify differentiated promotional intensity.
- Category share can be used to identify where to prioritize future campaigns for the selected SKU.
"""
    )

# =========================================================
# ML Price Engine
# =========================================================
else:
    st.title("ML Price Recommendation Engine")
    st.caption("Dummy version for pricing decision support")

    col_left, col_right = st.columns([1.1, 1.9])

    with col_left:
        st.subheader("Inputs")

        sku_input = st.selectbox("SKU", all_skus, index=0, key="ml_sku")
        partner_input = st.selectbox(
            "Trading Partner", all_partners, index=0, key="ml_partner"
        )
        time_window = st.selectbox(
            "Time window",
            ["Last 4 weeks", "Last 8 weeks", "Last 12 weeks"],
            index=1,
        )
        promo_type = st.selectbox(
            "Promotion type",
            ["None", "Discount", "Bundle", "Cashback"],
            index=1,
        )

        inventory_condition = st.selectbox(
            "Inventory condition",
            ["Normal", "Low inventory", "Overstock"],
            index=0,
        )

        run_model = st.button("Run ML model (dummy)")

    with col_right:
        st.subheader("Recommended price (dummy output)")

        if run_model:
            sku_df = data[(data["sku"] == sku_input) & (
                data["partner"] == partner_input)]
            base_price = sku_df["price"].mean() if not sku_df.empty else 8000

            factor = 1.0
            if promo_type != "None":
                factor -= 0.05
            if inventory_condition == "Overstock":
                factor -= 0.04
            if inventory_condition == "Low inventory":
                factor += 0.03

            rec_price = base_price * factor
            lower = rec_price * 0.95
            upper = rec_price * 1.05
            confidence = np.random.uniform(0.82, 0.93)

            st.markdown(f"**Recommended price:** {rec_price:,.0f} MXN")
            st.markdown(f"Suggested range: {lower:,.0f} to {upper:,.0f} MXN")
            st.markdown(f"Model confidence (dummy): {confidence * 100:.1f}%")

            st.markdown("---")

            st.subheader("Revenue vs price curve (dummy)")

            price_grid = np.linspace(rec_price * 0.8, rec_price * 1.2, 25)
            revenue = []
            for p in price_grid:
                demand = max(0, 1200 * (rec_price / p) +
                             np.random.normal(0, 40))
                revenue.append(p * demand)

            fig_rev = go.Figure()
            fig_rev.add_trace(
                go.Scatter(
                    x=price_grid,
                    y=revenue,
                    mode="lines",
                    name="Revenue",
                    line=dict(color="#C69214"),
                )
            )
            fig_rev.add_trace(
                go.Scatter(
                    x=[rec_price],
                    y=[np.interp(rec_price, price_grid, revenue)],
                    mode="markers",
                    name="Recommended price",
                    marker=dict(size=10, color="#333333"),
                )
            )
            fig_rev.update_layout(
                title="Revenue curve for different price points (dummy)",
                xaxis_title="Price (MXN)",
                yaxis_title="Revenue (dummy units)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_rev, use_container_width=True)

            st.subheader("Historical forecast error (dummy)")

            periods = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
            error_vals = np.random.uniform(0.05, 0.18, size=len(periods))

            fig_err = px.bar(
                x=periods,
                y=error_vals,
                labels={"x": "Period", "y": "Mean absolute error"},
                title="Forecast error by period (dummy)",
            )
            fig_err.update_yaxes(tickformat=".0%")
            fig_err.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_err, use_container_width=True)

            st.subheader("Analyst interpretation")
            st.write(
                """
- The recommended price sits near the top of the revenue curve, so it should be evaluated together with elasticity and partner strategy.
- Error levels are not negligible, so the ML output should be complemented with recent campaign performance and business judgement.
- This tool is intended as decision support, not an automatic override of analyst experience.
"""
            )
        else:
            st.info(
                "Configure the inputs on the left and click the button to generate a dummy recommendation.")
