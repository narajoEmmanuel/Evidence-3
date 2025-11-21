import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
#                     STYLE (Whirlpool Gold + Gray)
# ---------------------------------------------------------------------
st.set_page_config(page_title="Whirlpool Pricing Analytics", layout="wide")

st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #f7f7f7;
    border-right: 1px solid #e0e0e0;
}
.whirlpool-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #2c2c2c;
    text-align: center;
    margin-bottom: 1rem;
}

/* Page Title */
.page-title {
    font-size: 1.7rem;
    font-weight: 700;
    color: #2c2c2c;
}
.page-subtitle {
    font-size: 0.95rem;
    color: #666666;
    margin-bottom: 1rem;
}

/* KPI number */
.big-kpi {
    font-size: 2.2rem;
    font-weight: 700;
    color: #2c2c2c;
}

/* Gold highlight */
.gold {
    color: #c69214;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
#                    GENERATE DUMMY DATA
# ---------------------------------------------------------------------
def generate_dummy_data():
    np.random.seed(42)

    weeks = pd.date_range("2024-01-01", periods=30, freq="W-MON")
    skus = {
        "WFR5200D": "Washing Machines",
        "AFR2110G": "Refrigerators",
        "8MWTW2023WPM": "Washers & Dryers"
    }
    partners = ["WALMART", "LIVERPOOL", "CHEDRAUI"]

    rows = []
    for sku, cat in skus.items():
        base_demand = np.random.randint(400, 900)
        base_price = np.random.randint(6500, 9500)
        base_inventory = np.random.randint(600, 1600)

        for partner in partners:
            inventory = base_inventory
            factor = np.random.uniform(0.8, 1.2)

            for w in weeks:
                price = base_price * np.random.uniform(0.9, 1.1)
                real_price = price * np.random.uniform(0.97, 1.03)
                promo = np.random.choice([0, 1], p=[0.7, 0.3])
                promo_effect = 1.25 if promo == 1 else 1.0
                noise = np.random.normal(0, 60)

                demand = max(0, base_demand * factor *
                             promo_effect * (9500 / price) + noise)
                sell_out = min(inventory, demand)

                inventory = max(0, inventory - sell_out +
                                np.random.randint(200, 500))

                rows.append({
                    "week": w,
                    "week_of_year": w.isocalendar().week,
                    "sku": sku,
                    "category": cat,
                    "partner": partner,
                    "price": round(price),
                    "real_price": round(real_price),
                    "promo_flag": promo,
                    "sell_out": round(sell_out),
                    "inventory": round(inventory)
                })

    return pd.DataFrame(rows)


data = generate_dummy_data()


# ---------------------------------------------------------------------
#                            SIDEBAR
# ---------------------------------------------------------------------
with st.sidebar:

    # -----------------------------------------
    # LOGO
    # -----------------------------------------
    st.image("whirlpool_logo.png", use_container_width=True)

    # TITLE
    st.markdown("## **Pricing Analytics Suite**")
    st.caption("Smart insights for strategic pricing")

    # Slim line
    st.markdown("<hr style='margin-top:0.3rem; margin-bottom:0.3rem;'>",
                unsafe_allow_html=True)

    # -----------------------------------------
    # NAVIGATION
    # -----------------------------------------
    st.markdown("#### Navigation")
    page = st.radio(
        "",
        ["Internal Pricing Performance",
            "Trading Partner Performance", "ML Price Calculator"],
    )

    # Slim divider
    st.markdown("<hr style='margin-top:0.3rem; margin-bottom:0.3rem;'>",
                unsafe_allow_html=True)

    # -----------------------------------------
    # USER PROFILE (Compact)
    # -----------------------------------------
    st.markdown("#### User Profile")
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])

        with col1:
            st.write("👨🏻‍💼")
        with col2:
            st.write("**Emmanuel Naranjo**")
            st.caption("Pricing Data Analyst")
            st.markdown("🟡 **Internal Tool**")

    # Tight spacing
    st.write("")

    # -----------------------------------------
    # SYSTEM DETAILS (Compact)
    # -----------------------------------------
    st.markdown("#### System Details")
    with st.container(border=True):
        st.write("**Version:** 1.0.0")
        st.write("**Last Update:** Nov 20, 2025")
        st.write("**Status:** 🟢 Online")

    st.write("")

    # -----------------------------------------
    # TOOLS SECTION (Compact)
    # -----------------------------------------
    st.markdown("#### Tools")

    if st.button("📄 Download PDF Report"):
        st.success("Report generation started…")

    if st.button("📘 Documentation"):
        st.info("Opening documentation…")

    if st.button("❓ Help & Support"):
        st.warning("Contact: analytics-support@whirlpool.com")

    # -----------------------------------------
    # ABOUT (Compact)
    # -----------------------------------------
    st.markdown("#### About This Dashboard")
    with st.container(border=True):
        st.caption(
            "Internal Whirlpool tool for SKU monitoring, "
            "partner analytics and ML-based pricing insights."
        )

    st.write("")

    # Minimal footer spacing
    st.write("")
    st.caption("© Whirlpool Corporation — Internal Use Only")


# ---------------------------------------------------------------------
#                 PAGE 1 – INTERNAL PRICING DASHBOARD
# ---------------------------------------------------------------------
def internal_page(df):

    # -------------------------------------------------------------
    # TITLE & STORYTELLING INTRO – IMPROVED STYLE
    # -------------------------------------------------------------
    st.markdown("## **Internal Pricing Performance Overview**")
    st.markdown(
        """
        <div style='font-size:1.05rem; font-weight:400; line-height:1.45; margin-bottom:0.6rem;'>
            Explore the full picture of Whirlpool’s SKU portfolio across all retail partners. 
            This section provides a system-level understanding of demand behavior, pricing evolution 
            and inventory stability before drilling down into specific SKUs or partners.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write(" ")
    # -------------------------------------------------------------
    # KPI SECTION (no date filter, fixed values + dynamic SellOut)
    # -------------------------------------------------------------

    def kpi_card(title, value):
        st.markdown(
            f"""
            <div style='text-align:center; padding:0.6rem 0;'>
                <div style='font-size:1rem; font-weight:600; color:#444;'>{title}</div>
                <div style='font-size:2.4rem; font-weight:800; color:#c69214;'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    k1, k2, k3 = st.columns(3)

    with k1:
        kpi_card("Total Registered SKUs", "360")

    with k2:
        kpi_card("Active Trade Partners", "8")

    with k3:
        total_sell_out = df["sell_out"].sum()
        kpi_card("Total Sell Out (QTY)", f"{total_sell_out:,}")

    st.markdown(
        """
        <div style='text-align:center; font-size:0.92rem; color:#555; margin-top:0.5rem; margin-bottom:0.8rem;'>
            These metrics provide a high-level snapshot of Whirlpool’s commercial footprint across all partners.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # TOP SKU SUMMARY TABLE
    # -------------------------------------------------------------
    st.markdown("### Top SKUs Summary")
    st.markdown(
        """
    <div style='font-size:1rem; color:#555; line-height:1.4; margin-bottom:1rem;'>
        This section highlights the strongest performers across Whirlpool’s full SKU portfolio. 
        It compares total Sell Out, average Real Price and inventory stability, offering a clear view 
        of which products consistently drive commercial performance.
    </div>
    """,
        unsafe_allow_html=True
    )

    top_df = (
        df.groupby(["sku", "category"])
        .agg(
            total_qty=("sell_out", "sum"),
            avg_price=("real_price", "mean"),
            avg_inventory=("inventory", "mean"),
        )
        .reset_index()
        .sort_values("total_qty", ascending=False)
        .head(5)
    )

    top_df["avg_price"] = top_df["avg_price"].astype(int)
    top_df["avg_inventory"] = top_df["avg_inventory"].astype(int)

    top_df_display = top_df.rename(columns={
        "sku": "SKU Code",
        "category": "Product Category",
        "total_qty": "Total Units Sold",
        "avg_price": "Average Real Price (MXN)",
        "avg_inventory": "Average Inventory (Units)"
    })

    st.dataframe(top_df_display, use_container_width=True)

# -------------------------------------------------------------
    # FILTERS FOR DETAILED VIEW (SKU + Partner)
    # -------------------------------------------------------------
    st.markdown(" ")
    st.markdown("### **SKU and Partner Selection**")
    st.markdown(
        """
    <div style='font-size:1rem; color:#555; line-height:1.4; margin-bottom:1rem;'>
        Select a specific SKU and retail partner to explore how pricing, demand and inventory evolve 
        over time. This view allows you to identify trends, compare retailer behavior and detect 
        early signals that may impact commercial performance.
    </div>
    """,
        unsafe_allow_html=True
    )

    colf1, colf2 = st.columns(2)

    with colf1:
        selected_sku = st.selectbox("SKU", sorted(df["sku"].unique()))

    with colf2:
        partner_options = ["All Partners"] + sorted(df["partner"].unique())
        selected_partner = st.selectbox("Trade Partner", partner_options)

    filtered = df[df["sku"] == selected_sku]

    if selected_partner != "All Partners":
        filtered = filtered[filtered["partner"] == selected_partner]

    # -------------------------------------------------------------
    # TIME SERIES SECTION — WITH COLOR PER TRADE PARTNER
    # -------------------------------------------------------------
    st.markdown(" ")
    st.markdown("### Time Series Analysis")
    st.markdown(
        """
    <div style='font-size:1rem; color:#555; line-height:1.4; margin-bottom:1rem;'>
        These visual timelines reveal how the selected SKU evolves week by week in terms of demand, 
        inventory availability and pricing behavior. Each retail partner is represented by a different 
        color, making it easier to compare performance patterns and detect differences in strategy 
        or consumer response.
    </div>
    """,
        unsafe_allow_html=True
    )

    st.write("")

    # Assign partner colors
    partner_colors = {
        "WALMART": "#c69214",      # gold
        "LIVERPOOL": "#8a6d2f",    # dark gold
        "CHEDRAUI": "#2c2c2c",     # charcoal
    }

    # Sell Out timeline
    tab1, tab2, tab3 = st.tabs(
        ["Sell Out Over Time", "Inventory Over Time", "Real Price Over Time"])

    with tab1:
        st.markdown("#### Sell Out (Units Sold per Week)")
        st.caption(
            "Shows weekly demand for the SKU. Peaks may indicate promotions or seasonal effects.")
        fig_so = px.line(
            filtered,
            x="week",
            y="sell_out",
            color="partner",
            color_discrete_map=partner_colors,
            labels={"week": "Week", "sell_out": "Units Sold"},
        )
        st.plotly_chart(fig_so, use_container_width=True)

    with tab2:
        st.markdown("#### Inventory Levels Over Time")
        st.caption(
            "Tracks available stock week by week. Drops can indicate high demand or insufficient replenishment.")
        fig_inv = px.line(
            filtered,
            x="week",
            y="inventory",
            color="partner",
            color_discrete_map=partner_colors,
            labels={"week": "Week", "inventory": "Units in Stock"},
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    with tab3:
        st.markdown("#### Real Price Evolution")
        st.caption(
            "Represents the final price to consumer after adjustments, promotions or partner price updates.")
        fig_rp = px.line(
            filtered,
            x="week",
            y="real_price",
            color="partner",
            color_discrete_map=partner_colors,
            labels={"week": "Week", "real_price": "Real Price (MXN)"},
        )
        st.plotly_chart(fig_rp, use_container_width=True)

    st.divider()

    # -------------------------------------------------------------
    # GLOSSARY – MINIMALIST STYLE
    # -------------------------------------------------------------
    st.markdown(
        """
        <div style='font-size:0.95rem; color:#555; margin-bottom:0.8rem;'>
            Use this glossary as a quick reference to interpret pricing, demand and inventory metrics 
            throughout the dashboard.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ---- Two-column layout with minimalistic note boxes ----
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style='background:#f7f7f7; padding:0.8rem 1rem; border-left:4px solid #c69214; border-radius:4px; margin-bottom:1rem;'>
                <strong>Sell Out</strong><br>
                Total units sold during a specific week. Indicates true product movement and demand.
            </div>

            <div style='background:#f7f7f7; padding:0.8rem 1rem; border-left:4px solid #c69214; border-radius:4px; margin-bottom:1rem;'>
                <strong>Inventory</strong><br>
                Units available at the retailer’s network for the selected SKU. Critical for replenishment planning.
            </div>

            <div style='background:#f7f7f7; padding:0.8rem 1rem; border-left:4px solid #c69214; border-radius:4px; margin-bottom:1rem;'>
                <strong>Real Price</strong><br>
                Effective selling price after applying discounts, promotions and partner-specific adjustments.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style='background:#f7f7f7; padding:0.8rem 1rem; border-left:4px solid #c69214; border-radius:4px; margin-bottom:1rem;'>
                <strong>Average Price</strong><br>
                Mean Real Price observed over the analyzed period for the selected SKU.
            </div>

            <div style='background:#f7f7f7; padding:0.8rem 1rem; border-left:4px solid #c69214; border-radius:4px; margin-bottom:1rem;'>
                <strong>Average Inventory</strong><br>
                Mean inventory level held throughout the selected time window.
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------------------------------------------------------------------
#             PAGE 2 – TRADING PARTNER PERFORMANCE
# ---------------------------------------------------------------------


def partner_page(df):

    # -------------------------------------------------------------
    # HEADER: TITLE + FILTER
    # -------------------------------------------------------------
    header_left, header_right = st.columns([2, 1])

    with header_left:
        st.markdown("## **Trading Partner Performance**")
        st.markdown(
            """
            <div style='font-size:1rem; font-weight:400; line-height:1.4; margin-bottom:0.8rem;'>
                Explore how each retailer contributes to Whirlpool’s commercial results. 
                This view focuses on SKU-level contribution, partner behavior and long-term 
                demand patterns to support strategic, data-driven conversations.
            </div>
            """,
            unsafe_allow_html=True
        )

    with header_right:
        st.markdown("<div style='padding-top:18px;'>", unsafe_allow_html=True)
        partner = st.selectbox(
            "Select Partner", sorted(df["partner"].unique()))
        st.markdown("</div>", unsafe_allow_html=True)

    st.write(" ")

    # -------------------------------------------------------------
    # PARTNER TITLE (CENTERED)
    # -------------------------------------------------------------
    st.markdown(
        f"""
        <div style='text-align:center; margin-top:0.5rem;'>
            <span style='font-size:2.3rem; font-weight:800; color:#c69214;'>
                {partner}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    partner_df = df[df["partner"] == partner]

    # -------------------------------------------------------------
    # PARTNER KPIs (CENTERED)
    # -------------------------------------------------------------
    total_skus = partner_df["sku"].nunique()
    total_qty = partner_df["sell_out"].sum()
    total_inv = int(partner_df["inventory"].mean())

    def centered_kpi(label, value):
        st.markdown(
            f"""
            <div style='text-align:center; padding:0.5rem 0;'>
                <div style='font-size:1rem; font-weight:600; color:#444;'>{label}</div>
                <div style='font-size:2.4rem; font-weight:800; color:#2c2c2c;'>{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    k1, k2, k3 = st.columns(3)
    with k1:
        centered_kpi("Active SKUs", total_skus)
    with k2:
        centered_kpi("Total Sell Out (QTY)", f"{total_qty:,}")
    with k3:
        centered_kpi("Average Inventory", f"{total_inv:,}")

    st.markdown(
        """
        <div style='text-align:center; font-size:0.95rem; color:#555; margin-top:0.5rem;'>
            These indicators provide a high-level snapshot of the partner’s SKU variety, 
            demand intensity and inventory stability.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # TOP 5 SKUs BY VOLUME
    # -------------------------------------------------------------
    st.markdown("### **Top SKUs by Sales Volume**")
    st.markdown(
        """
        <div style='font-size:0.95rem; color:#555; line-height:1.45; margin-bottom:1rem;'>
            These SKUs represent the strongest contributors to this partner’s Sell Out performance. 
            Understanding their pricing, demand and inventory relationship helps prioritize 
            promotional strategy and replenishment planning.
        </div>
        """,
        unsafe_allow_html=True,
    )

    top5 = (
        partner_df.groupby(["sku", "category"])
        .agg(
            Total_QTY=("sell_out", "sum"),
            Avg_Inventory=("inventory", "mean"),
            Avg_Price=("real_price", "mean"),
        )
        .reset_index()
        .sort_values("Total_QTY", ascending=False)
        .head(5)
    )

    top5["Avg_Inventory"] = top5["Avg_Inventory"].astype(int)
    top5["Avg_Price"] = top5["Avg_Price"].astype(int)

    # Rename columns for more meaningful display
    top5_display = top5.rename(columns={
        "sku": "SKU Code",
        "category": "Product Category",
        "Total_QTY": "Total Units Sold",
        "Avg_Inventory": "Average Inventory (Units)",
        "Avg_Price": "Average Real Price (MXN)"
    })

    st.dataframe(top5_display, use_container_width=True)

    fig = px.bar(
        top5,
        x="sku",
        y="Total_QTY",
        color="category",
        color_discrete_sequence=["#c69214", "#e0b65a", "#8a6d2f"],
        title=f"Top SKUs by Quantity Sold – {partner}",
        labels={"Total_QTY": "Units Sold", "sku": "SKU"},
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        title_font=dict(size=18, color="#2c2c2c"),
        xaxis_title="SKU",
        yaxis_title="Units Sold",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"""
        The SKUs above represent a major share of **{partner}’s** demand profile.

        • High volume may reflect strong price elasticity or brand preference  
        • Low inventory combined with strong demand may signal stock risk  
        • Category concentration reveals if the partner is diversified or dependent on a small product group  
        """
    )

    st.markdown("---")

    # -------------------------------------------------------------
    # TIME SERIES TRENDS
    # -------------------------------------------------------------
    st.markdown("### **Weekly Behavior Across SKUs**")
    st.caption(
        "These time series reveal pricing, demand and inventory patterns averaged across all SKUs for this partner."
    )

    temporal_df = (
        partner_df.groupby("week")
        .agg(
            avg_qty=("sell_out", "mean"),
            avg_price=("real_price", "mean"),
            avg_inventory=("inventory", "mean"),
        )
        .reset_index()
    )

    tabs = st.tabs(
        ["Sell Out Timeline", "Price Timeline", "Inventory Timeline"])

    with tabs[0]:
        fig_qty = px.line(
            temporal_df, x="week", y="avg_qty", markers=True,
            title="Average Weekly Sell Out",
            labels={"week": "Week", "avg_qty": "Units Sold (Avg)"}
        )
        st.plotly_chart(fig_qty, use_container_width=True)

    with tabs[1]:
        fig_price = px.line(
            temporal_df, x="week", y="avg_price", markers=True,
            title="Average Weekly Real Price",
            labels={"week": "Week", "avg_price": "Real Price (MXN)"}
        )
        st.plotly_chart(fig_price, use_container_width=True)

    with tabs[2]:
        fig_inv = px.line(
            temporal_df, x="week", y="avg_inventory", markers=True,
            title="Average Weekly Inventory",
            labels={"week": "Week", "avg_inventory": "Stock Level (Avg)"}
        )
        st.plotly_chart(fig_inv, use_container_width=True)

    st.markdown("---")

    # -------------------------------------------------------------
    # CLOSING STORYTELLING
    # -------------------------------------------------------------
    st.markdown(
        f"""
        <div style='
            margin-top:1.2rem; 
            padding:0.8rem 1rem; 
            background-color:#f7f7f7; 
            border-left:4px solid #c69214; 
            font-size:0.9rem; 
            color:#444;
            border-radius:4px;
        '>
            <strong>Note:</strong> The weekly trends and SKU rankings shown above provide a 
            consolidated snapshot of <strong>{partner}</strong>’s commercial behavior. 
            Use these insights as guidance for discussions on pricing, stock planning 
            and promotional coordination with this retail partner.
        </div>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------------------------
#              PAGE 3 – ML PRICE CALCULATOR (ELEGANT)
# ---------------------------------------------------------------------


def ml_engine_page(df):
    # -------------------------------------------------------------
    # TITLE & STORYTELLING INTRO
    # -------------------------------------------------------------
    st.markdown("## **ML Price Recommendation Engine**")

    # ---------------------------------------------------------
    #                     MODEL EXPLANATION
    # ---------------------------------------------------------
    st.info(
        """
        The ML model behind this calculator uses historical Real Price, promotional activity, Sell Out, 
        inventory flow, SKU seasonality and partner specific dynamics to generate a recommended price for a specific week.  
        
        It considers:  
        • Real Price and discount history  
        • Inventory cycles and consumption patterns  
        • Week of the year and seasonality  
        • Partner behavior and sensitivity  
        • Lagged demand variables and elasticity-like effects  
        """
    )

    # ---------------------------------------------------------
    # MODEL PERFORMANCE PANEL (XGBoost)
    # ---------------------------------------------------------

    # Simulated performance metrics for XGBoost model
    model_rmse = 182.4
    model_mape = 7.8
    model_r2 = 0.86

    model_params = {
        "n_estimators": 300,
        "learning_rate": 0.06,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "train_size": "80%",
        "test_size": "20%"
    }

    st.markdown("### Model Performance Summary")
    st.markdown(
        """
    <div style='font-size:1rem; color:#555; line-height:1.4; margin-bottom:1rem;'>
        This section summarizes the predictive accuracy and calibration behavior of the 
        XGBoost model powering the weekly price recommendations. These metrics help determine 
        how reliable the model is under different partner dynamics, SKU patterns and seasonal conditions.
    </div>
    """,
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("**RMSE**")
        st.markdown(
            f"<div class='big-kpi gold'>{model_rmse}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("**MAPE percent**")
        st.markdown(
            f"<div class='big-kpi'>{model_mape}%</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("**R2 Score**")
        st.markdown(
            f"<div class='big-kpi'>{model_r2}</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("**Train Split**")
        st.markdown(
            f"<div class='big-kpi'>{model_params['train_size']}</div>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------------------------------------------------------
    #                      INPUT PANEL (LEFT) + INSTRUCTIONS (RIGHT)
    # ---------------------------------------------------------

    st.markdown("### Configuration Panel")
    st.markdown(
        """
    <div style='font-size:1rem; color:#555; line-height:1.4; margin-bottom:1rem;'>
        Configure the conditions you want to analyze. The selected week, SKU and 
        trading partner define the scenario the model evaluates to generate a 
        data-driven price recommendation.
    </div>
    """,
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1.2, 1])   # left wider

    # ---------------- LEFT COLUMN: INPUTS ----------------
    with col_left:
        st.markdown("#### Input Parameters")

        c1, c2 = st.columns(2)
        with c1:
            week = st.number_input(
                "Week of the year", min_value=1, max_value=52, value=10)
        with c2:
            partner = st.selectbox(
                "Trade Partner", sorted(df["partner"].unique()))

        sku = st.selectbox("SKU", sorted(df["sku"].unique()))
        category = df[df["sku"] == sku]["category"].iloc[0]
        st.text_input("Category", value=category, disabled=True)

    # ---------------- RIGHT COLUMN: INSTRUCTIONS ----------------
    with col_right:
        st.markdown("#### How to Use This Tool")
        st.info(
            """
            **1.** Select week, SKU and trading partner.  
            **2.** The system analyzes historical pricing and demand patterns.  
            **3.** You will obtain:  
                • recommended price  
                • expected stock usage  
                • comparison vs current price  
                • historical supporting time series 
                
            **4.** Use the charts to validate or adjust model output.
            """
        )

    st.markdown(" ")

    # ---------------------------------------------------------
    #                     RUN BUTTON
    # ---------------------------------------------------------
    if st.button("Run Price Recommendation", type="primary"):

        filtered = df[(df["sku"] == sku) & (df["partner"] == partner)]

        # ---------------------------------------------------------
        #               ML PREDICTION (SIMULATED)
        # ---------------------------------------------------------
        base_price = filtered["real_price"].mean()
        recommended_price = int(base_price * np.random.uniform(0.92, 1.08))
        current_price = int(filtered["real_price"].iloc[-1])
        expected_stock = int(
            filtered["inventory"].mean() * np.random.uniform(0.75, 1.25))

        # ---------------------------------------------------------
        #                  OUTPUTS – CARDS
        # ---------------------------------------------------------
        st.markdown("### Results")

        k1, k2, k3 = st.columns(3)

        with k1:
            st.markdown("**Recommended Price (MXN)**")
            st.markdown(
                f"<div class='big-kpi gold'>{recommended_price:,}</div>", unsafe_allow_html=True)

        with k2:
            st.markdown("**Current Price (MXN)**")
            st.markdown(
                f"<div class='big-kpi'>{current_price:,}</div>", unsafe_allow_html=True)

        with k3:
            st.markdown("**Expected Stock Needed (units)**")
            st.markdown(
                f"<div class='big-kpi'>{expected_stock:,}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # ---------------------------------------------------------
        #             SUPPORTING TIME SERIES TABS
        # ---------------------------------------------------------

        st.markdown("### Historical Behavior of the SKU")

        st.caption(
            "These charts provide essential context to validate whether the recommended price "
            "aligns with realistic demand, pricing stability and inventory patterns."
        )

        tab1, tab2, tab3 = st.tabs([
            "Inventory timeline",
            "Price (Real Price) timeline",
            "Sell Out timeline"
        ])

        # --- Inventory Time Series ---
        with tab1:
            fig_inv = px.line(
                filtered.sort_values("week"),
                x="week",
                y="inventory",
                markers=True,
                title=f"Inventory Over Time – {sku}",
                labels={"week": "Week", "inventory": "Inventory level"},
            )
            fig_inv.update_traces(line=dict(color="#c69214"))
            st.plotly_chart(fig_inv, use_container_width=True)

        # --- Real Price Time Series ---
        with tab2:
            fig_price = px.line(
                filtered.sort_values("week"),
                x="week",
                y="real_price",
                markers=True,
                title=f"Real Price Over Time – {sku}",
                labels={"week": "Week", "real_price": "Real price (MXN)"},
            )
            fig_price.update_traces(line=dict(color="#8a6d2f"))
            st.plotly_chart(fig_price, use_container_width=True)

        # --- Sell Out Time Series ---
        with tab3:
            fig_out = px.line(
                filtered.sort_values("week"),
                x="week",
                y="sell_out",
                markers=True,
                title=f"Sell Out Over Time – {sku}",
                labels={"week": "Week", "sell_out": "Units sold"},
            )
            fig_out.update_traces(line=dict(color="#2c2c2c"))
            st.plotly_chart(fig_out, use_container_width=True)

        # ---------------------------------------------------------
        #           FINAL INTERPRETATION BOX (Human in the Loop)
        # ---------------------------------------------------------

        st.markdown("### Interpretation")

        interpretation_text = f"""
        The model suggests a price of **{recommended_price:,} MXN**, compared to the current price of **{current_price:,} MXN**.  
        
        Historical demand for *{sku}* at *{partner}* shows that:
        - Inventory tends to stabilize around **{int(filtered['inventory'].mean()):,} units**  
        - Real Price usually fluctuates between **{int(filtered['real_price'].min()):,}** and **{int(filtered['real_price'].max()):,} MXN**  
        - Sell Out experiences week to week variation linked to price sensitivity and replenishment cycles  

        Based on these patterns, the recommended price aligns with the SKU's past performance and expected demand for week {week}.

        Use the graphs above to verify if promotional events, sudden price changes or low inventory levels should adjust the recommendation.
        """

        st.info(interpretation_text)


# ---------------------------------------------------------------------
#                        PAGE ROUTING
# ---------------------------------------------------------------------
if page == "Internal Pricing Performance":
    internal_page(data)
elif page == "Trading Partner Performance":
    partner_page(data)
else:
    ml_engine_page(data)
