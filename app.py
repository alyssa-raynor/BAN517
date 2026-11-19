"""
Supply Chain Analytics - Inventory Decision Agent Application
Dual-Mode:
1. Streamlit Interactive Dashboard (Run via: `streamlit run app.py`)
2. Command-Line Interface (Run via: `python3 app.py "question"` or `python3 app.py --cli`)
"""

import sys
import argparse
import pandas as pd
from inventory_agent import InventoryAgent


# ===========================================================================
# CLI Execution Mode
# ===========================================================================
def run_cli_interactive():
    agent = InventoryAgent()
    print("=" * 75)
    print("  SUPPLY CHAIN ANALYTICS - INVENTORY DECISION AGENT (CLI MODE)")
    print("  Supported: EOQ | Quantity Discount | Safety Stock & ROP | Fixed-Period | Newsvendor")
    print("  Type 'quit' or 'exit' to exit.")
    print("=" * 75)

    sample_questions = [
        "1. EOQ: 'What is the EOQ for annual demand 10,000, order cost $50, holding cost $2?'",
        "2. Discount: 'Demand 5000, order cost $49, discount tiers 0-499: $20, 500-999: $18, 1000+: $16'",
        "3. ROP: 'Average demand 100, lead time 5 days, std dev demand 15, service level 95%'",
        "4. Periodic: 'Periodic review every 14 days, lead time 4 days, daily demand 50, service level 95%'",
        "5. Newsvendor: 'Newsvendor model: price $100, cost $40, salvage $15, demand mean 500, std dev 100'",
    ]
    print("\nQuick Examples:")
    for sq in sample_questions:
        print(f"  {sq}")
    print("\n" + "-" * 75)

    while True:
        try:
            user_input = input("\n[Enter inventory question] > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Exiting Supply Chain Inventory Agent. Goodbye!")
                break

            run_and_display_cli(agent, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


def run_and_display_cli(agent: InventoryAgent, question: str):
    print("\n" + "=" * 75)
    print("  AGENT PROCESSING PIPELINE: 7-STAGE EXECUTION")
    print("=" * 75)

    print(f"\n[STAGE 1: Customer Question]\n  \"{question}\"")

    # Run Pipeline
    pipeline_out = agent.run_pipeline(question)
    id_info = pipeline_out["identification"]
    inputs = pipeline_out["inputs"]
    tool_name = pipeline_out["tool_name"]
    res = pipeline_out["result"]
    verif = pipeline_out["verification"]
    explanation = pipeline_out["explanation"]

    print(f"\n[STAGE 2: Identify Model]\n  Model: {id_info['model_name']}\n  Confidence: {int(id_info['confidence']*100)}%\n  Matched Signals: {', '.join(id_info['reasons'])}")
    print(f"\n[STAGE 3: Extract Inputs]")
    for k, v in inputs.items():
        print(f"  • {k}: {v}")
    print(f"\n[STAGE 4: Select Python Tool]\n  Selected: `inventory_tools.{tool_name}`")
    print(f"\n[STAGE 5: Calculate]\n  Optimal Target: {res.get('recommended_order_quantity') or res.get('recommended_reorder_point')} units")
    print(f"\n[STAGE 6: Verify]\n  Status: {verif['status']}")
    for c in verif["checks"]:
        print(f"  ✓ {c}")
    print(f"\n[STAGE 7: Explain]\n")
    print(explanation)
    print("=" * 75)


# ===========================================================================
# Streamlit Interactive Web Application Mode
# ===========================================================================
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
        }
        .pipeline-step {
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 0.88rem;
            font-weight: 600;
            display: inline-block;
            margin-right: 5px;
            background-color: #EEF2FF;
            color: #3730A3;
        }
        </style>
    """, unsafe_allow_html=True)

    agent = InventoryAgent()

    st.markdown('<div class="main-header">📦 Supply Chain Inventory Decision Agent</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Automated Decision Architecture: '
        '<b>Question ➔ Identify Model ➔ Extract Inputs ➔ Select Python Tool ➔ Calculate ➔ Verify ➔ Explain</b></div>',
        unsafe_allow_html=True
    )

    # Sidebar: Presets and Configuration
    st.sidebar.header("🎯 Preset Scenarios")
    preset_choice = st.sidebar.selectbox(
        "Choose an industry benchmark scenario:",
        [
            "1. Classic EOQ (Consumer Goods)",
            "2. Quantity Discount (Industrial Components)",
            "3. Safety Stock & ROP (Retail Distribution)",
            "4. Fixed-Period Review (Grocery Restocking)",
            "5. Newsvendor Model (Fashion & Seasonal)",
            "Custom Query",
        ]
    )

    # Sidebar: Data Upload Section (Last section in sidebar)
    st.sidebar.markdown("---")
    st.sidebar.header("📂 Data Upload")
    uploaded_csv = st.sidebar.file_uploader("Upload CSV File", type=["csv"], help="Upload an inventory CSV dataset containing Product, Category, Units, and Unit Cost.")

    scenario_prompts = {
        "1. Classic EOQ (Consumer Goods)": (
            "We have an annual demand of 12,000 units of standard widgets. "
            "Our order placement and setup cost is $65 per order, and holding cost is $2.50 per unit per year. "
            "What is our economic order quantity, how many orders per year should we place, and what is our total cost?"
        ),
        "2. Quantity Discount (Industrial Components)": (
            "Our supplier provides an all-units quantity discount for electric motors with annual demand of 6,000 units and order cost $50. "
            "Holding rate is 20% of purchase price. "
            "Price tiers: 0 to 499 units: $25.00, 500 to 999 units: $22.50, 1000+ units: $20.00. "
            "Which quantity should we order to minimize total annual cost?"
        ),
        "3. Safety Stock & ROP (Retail Distribution)": (
            "Determine the safety stock and reorder point for a high-turnover product. "
            "Daily demand averages 120 units with a standard deviation of 18 units. "
            "Supplier lead time is 6 days with a standard deviation of 1.2 days. "
            "Our management requires a 95% cycle service level."
        ),
        "4. Fixed-Period Review (Grocery Restocking)": (
            "We operate a periodic review inventory policy every 14 days for dairy items. "
            "Supplier replenishment lead time is 3 days. Average daily demand is 80 units with a standard deviation of 15. "
            "We target a 98% service level. Currently we have 220 units on hand, 100 on order, and 10 backorders. "
            "What is our target order-up-to level and how many units should we order right now?"
        ),
        "5. Newsvendor Model (Fashion & Seasonal)": (
            "We are ordering seasonal ski jackets for the upcoming winter. Selling price is $180, purchase cost is $75, "
            "and any unsold jackets can be salvaged for $30 at the end of the season. "
            "Forecasted demand is normally distributed with mean 850 units and standard deviation 160 units. "
            "What is the optimal order quantity, critical fractile, and expected profit?"
        )
    }

    default_query = scenario_prompts.get(preset_choice, "")

    query = st.text_area(
        "💬 Enter your natural-language inventory question:",
        value=default_query,
        height=110,
        placeholder="e.g. Calculate EOQ for demand 10,000, order cost $50, holding cost $2"
    )

    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_clicked = st.button("🚀 Analyze & Solve", type="primary", use_container_width=True)

    if run_clicked or query:
        if not query.strip():
            st.warning("Please enter a question or select a preset scenario.")
            return

        with st.spinner("Processing through 7-stage inventory decision architecture..."):
            pipeline_data = agent.run_pipeline(query)

        id_info = pipeline_data["identification"]
        inputs = pipeline_data["inputs"]
        tool_name = pipeline_data["tool_name"]
        res = pipeline_data["result"]
        verif = pipeline_data["verification"]
        explanation = pipeline_data["explanation"]

        # Display Pipeline Stage Pills
        st.markdown("---")
        st.markdown("""
        <div style="margin-bottom: 15px;">
            <span class="pipeline-step">1. Identify Model ✅</span>
            <span class="pipeline-step">2. Extract Inputs ✅</span>
            <span class="pipeline-step">3. Select Tool ✅</span>
            <span class="pipeline-step">4. Calculate ✅</span>
            <span class="pipeline-step">5. Verify ✅</span>
            <span class="pipeline-step">6. Explain ✅</span>
        </div>
        """, unsafe_allow_html=True)

        # Tabs for full inspection
        tab1, tab2, tab3, tab4 = st.tabs(["🎓 Student Step-by-Step Report", "🧮 Numerical Details & Invariants", "📊 Sensitivity & Visuals", "⚙️ Extracted Pipeline Trace"])

        with tab1:
            st.markdown(explanation)

        with tab2:
            st.subheader("Numerical Calculation & Invariant Verification")
            c1, c2, c3, c4 = st.columns(4)

            model_type = id_info["model_type"]
            if model_type == InventoryAgent.MODEL_EOQ:
                c1.metric("Optimal EOQ (Q*)", f"{res['recommended_order_quantity']:,} units")
                c2.metric("Orders Per Year", f"{res['orders_per_year']} / yr")
                c3.metric("Cycle Time", f"{res['cycle_time_days']} days")
                c4.metric("Total Inventory Cost", f"${res['total_inventory_cost']:,.2f}")
            elif model_type == InventoryAgent.MODEL_QUANTITY_DISCOUNT:
                c1.metric("Optimal Tier", f"Tier {res['optimal_tier']}")
                c2.metric("Optimal Order Qty", f"{res['recommended_order_quantity']:,} units")
                c3.metric("Effective Unit Price", f"${res['optimal_unit_price']:.2f}")
                c4.metric("Total Annual Cost", f"${res['optimal_total_cost']:,.2f}")
            elif model_type == InventoryAgent.MODEL_SAFETY_STOCK_ROP:
                c1.metric("Safety Stock", f"{res['recommended_safety_stock']:,} units")
                c2.metric("Reorder Point (ROP)", f"{res['recommended_reorder_point']:,} units")
                c3.metric("Lead Time Demand", f"{res['expected_lead_time_demand']:,.1f} units")
                c4.metric("Stockout Risk / Cycle", f"{res['stockout_risk_per_cycle']}%")
            elif model_type == InventoryAgent.MODEL_FIXED_PERIOD_REVIEW:
                c1.metric("Order Quantity (Now)", f"{res['recommended_order_quantity']:,} units")
                c2.metric("Order-Up-To Target (S)", f"{res['recommended_order_up_to_level']:,} units")
                c3.metric("Inventory Position", f"{res['current_inventory_position']:,} units")
                protection_h = res["inputs"].get("protection_period_R_plus_L", inputs.get("review_period", 0) + inputs.get("lead_time", 0))
                c4.metric("Protection Horizon", f"{protection_h} {inputs.get('time_unit', 'days')}")
            elif model_type == InventoryAgent.MODEL_NEWSVENDOR:
                c1.metric("Optimal Order (Q*)", f"{res['recommended_order_quantity']:,} units")
                c2.metric("Critical Fractile (CR)", res['critical_ratio_percentage'])
                c3.metric("Expected Profit", f"${res['expected_profit']:,.2f}")
                c4.metric("Expected Sales", f"{res['expected_sales']:.1f} units")

            st.markdown("#### Mathematical Invariant Verification")
            st.success(f"{verif['status']}")
            for chk in verif["checks"]:
                st.write(f"✓ {chk}")

            with st.expander("Raw Calculation Result JSON"):
                st.json(res)

        with tab3:
            st.subheader("Visual Analytics & Sensitivity")
            if model_type == InventoryAgent.MODEL_EOQ:
                q_star = res["optimal_order_quantity"]
                q_range = [q_star * factor for factor in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]]
                d = inputs["annual_demand"]
                s = inputs["order_cost"]
                h = inputs["holding_cost"]
                chart_data = pd.DataFrame({
                    "Order Quantity (Q)": [int(q) for q in q_range],
                    "Ordering Cost": [(d / q) * s for q in q_range],
                    "Holding Cost": [(q / 2) * h for q in q_range],
                    "Total Inventory Cost": [((d / q) * s) + ((q / 2) * h) for q in q_range],
                }).set_index("Order Quantity (Q)")
                st.line_chart(chart_data)
                st.caption("EOQ Cost Trade-off: Total cost is minimized where Ordering Cost intersects Holding Cost.")

            elif model_type == InventoryAgent.MODEL_QUANTITY_DISCOUNT:
                tier_df = pd.DataFrame(res["evaluated_tiers"])
                st.dataframe(
                    tier_df[["tier_index", "min_qty", "max_qty", "unit_price", "evaluated_qty", "total_annual_cost", "is_candidate"]],
                    use_container_width=True
                )

            elif model_type == InventoryAgent.MODEL_NEWSVENDOR:
                mean_d = inputs["mean_demand"]
                std_d = inputs["std_demand"]
                q_opt = res["optimal_order_quantity"]
                p = inputs["unit_price"]
                c = inputs["unit_cost"]
                s = inputs["salvage_value"]

                import inventory_tools
                test_qs = [mean_d + (factor * std_d) for factor in [-1.5, -1.0, -0.5, 0.0, (q_opt - mean_d)/std_d, 0.5, 1.0, 1.5, 2.0]]
                test_qs = sorted(list(set(test_qs)))
                profits = []
                for tq in test_qs:
                    z = (tq - mean_d) / std_d
                    lost = std_d * inventory_tools._unit_normal_loss(z)
                    sales = mean_d - lost
                    left = tq - sales
                    prof = (sales * p) + (left * s) - (tq * c)
                    profits.append(round(prof, 2))

                nv_chart = pd.DataFrame({
                    "Order Quantity": [int(round(q)) for q in test_qs],
                    "Expected Profit ($)": profits
                }).set_index("Order Quantity")
                st.line_chart(nv_chart)
                st.caption(f"Newsvendor Profit Concavity: Maximum profit occurs at Q* = {int(round(q_opt))} units.")
            else:
                st.info("Interactive chart is available for EOQ, Quantity Discount, and Newsvendor models.")

        with tab4:
            st.subheader("Architecture Pipeline Execution Trace")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**1. Identified Model:**")
                st.write(f"- Name: `{id_info['model_name']}`")
                st.write(f"- Confidence: `{int(id_info['confidence']*100)}%`")
                st.write(f"- Diagnostic signals: {id_info['reasons']}")
                st.markdown("**2. Selected Python Tool:**")
                st.code(f"inventory_tools.{tool_name}(**inputs)")
            with col_t2:
                st.markdown("**3. Extracted Model Inputs:**")
                st.json(inputs)

    # =======================================================================
    # CSV Data Upload & Inventory Portfolio Analysis Section
    # =======================================================================
    if uploaded_csv is not None:
        st.markdown("---")
        st.header("📊 Inventory Dataset Processing & Portfolio Analysis")

        try:
            # 1. Read CSV into a pandas dataframe
            df = pd.read_csv(uploaded_csv)

            # 2. Validate required columns
            required_cols = ["Product", "Category", "Units", "Unit Cost"]
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                # 3. Clear error message identifying missing columns
                st.error(
                    f"⚠️ Missing Required Column(s): **{', '.join(missing_cols)}**.\n\n"
                    f"The uploaded CSV must contain the following columns: **{', '.join(required_cols)}**."
                )
            else:
                # 4 & 5. Display dataset profile (rows, columns, column names)
                st.success("✅ CSV uploaded and validated successfully!")
                
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                col_meta1.metric("Number of Rows", f"{df.shape[0]:,}")
                col_meta2.metric("Number of Columns", f"{df.shape[1]:,}")
                col_meta3.write(f"**Column Names:**\n`{', '.join(df.columns)}`")

                # DATA ANALYSIS using pandas
                # Ensure numerical types for Units and Unit Cost
                df["Units"] = pd.to_numeric(df["Units"], errors="coerce").fillna(0)
                df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce").fillna(0.0)

                # 1. Calculate total Units
                total_units = df["Units"].sum()

                # 2. Calculate average Unit Cost
                avg_unit_cost = df["Unit Cost"].mean() if len(df) > 0 else 0.0

                # 3. Identify Product with the highest Units
                if len(df) > 0:
                    highest_idx = df["Units"].idxmax()
                    highest_product = df.loc[highest_idx, "Product"]
                    highest_units = df.loc[highest_idx, "Units"]
                    highest_display = f"{highest_product} ({highest_units:,.0f} units)"
                else:
                    highest_display = "N/A (Empty dataset)"

                # 4. Group data by Category and calculate total Units for each Category
                category_units = df.groupby("Category")["Units"].sum().reset_index()
                category_units.columns = ["Category", "Total Units"]
                category_units = category_units.sort_values(by="Total Units", ascending=False)

                # 5. Create calculated column: Inventory Value = Units * Unit Cost
                df["Inventory Value"] = df["Units"] * df["Unit Cost"]

                # 6. Calculate total Inventory Value across all products
                total_inventory_val = df["Inventory Value"].sum()

                # RESULTS DISPLAY
                st.subheader("📈 Summary Analysis Metrics")
                res_col1, res_col2, res_col3, res_col4 = st.columns(4)
                res_col1.metric("Total Units", f"{total_units:,.0f}")
                res_col2.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
                res_col3.metric("Product with Highest Units", highest_display)
                res_col4.metric("Total Inventory Value", f"${total_inventory_val:,.2f}")

                # Total Units by Category
                st.subheader("🏷️ Total Units by Category")
                cat_col1, cat_col2 = st.columns([1, 1])
                with cat_col1:
                    st.dataframe(
                        category_units.style.format({"Total Units": "{:,.0f}"}),
                        use_container_width=True
                    )
                with cat_col2:
                    st.bar_chart(category_units.set_index("Category")["Total Units"])

                # Display the complete dataframe including the new Inventory Value column
                st.subheader("📋 Enriched Inventory Dataset (with Inventory Value)")
                display_df = df.copy()
                formatted_df = display_df.style.format({
                    "Units": "{:,.0f}",
                    "Unit Cost": "${:,.2f}",
                    "Inventory Value": "${:,.2f}",
                })
                st.dataframe(formatted_df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error processing uploaded CSV file: {str(e)}")


def is_running_in_streamlit() -> bool:
    """Accurately detect if the script is running under Streamlit server."""
    try:
        import streamlit as st
        if hasattr(st, "runtime") and hasattr(st.runtime, "exists"):
            return bool(st.runtime.exists())
    except Exception:
        pass
    return "streamlit" in sys.argv[0] or any("streamlit" in arg for arg in sys.argv)


# ===========================================================================
# Main entrypoint: Dispatches to CLI or Streamlit
# ===========================================================================
def main():
    if is_running_in_streamlit():
        run_streamlit_app()
        return

    parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
    parser.add_argument("query", nargs="*", help="Natural language inventory question")
    parser.add_argument("--cli", action="store_true", help="Launch interactive CLI REPL mode")

    args = parser.parse_args()

    if args.cli:
        run_cli_interactive()
    elif args.query:
        question = " ".join(args.query)
        agent = InventoryAgent()
        run_and_display_cli(agent, question)
    else:
        # Default banner and instructions
        print("Supply Chain Analytics Inventory Decision Agent")
        print("Usage:")
        print("  1. Launch Streamlit Web UI:   python3 -m streamlit run app.py")
        print("  2. Interactive CLI Mode:       python3 app.py --cli")
        print("  3. Direct Question Execution:  python3 app.py \"What is the EOQ for demand 10000, order cost 50, holding cost 2?\"")
        print("\nStarting Interactive CLI Mode by default...\n")
        run_cli_interactive()


if __name__ == "__main__":
    main()
