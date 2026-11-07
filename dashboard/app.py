"""
Telco Customer Intelligence Dashboard

Interactive Streamlit dashboard for exploring customer insights, 
churn predictions, and revenue optimization opportunities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Dashboard configuration
st.set_page_config(
    page_title="Telco Customer Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Premium Dark Theme
st.markdown("""
<style>
    /* ===== GOOGLE FONTS ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ===== GLOBAL STYLES ===== */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* ===== SIDEBAR STYLING ===== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-right: 2px solid rgba(255,255,255,0.05);
    }
    
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio label {
        padding: 6px 12px;
        border-radius: 8px;
        margin: 2px 0;
        transition: all 0.3s ease;
        font-size: 0.9rem;
    }
    
    [data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(255,255,255,0.1);
        transform: translateX(4px);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 1.5rem !important;
    }
    
    /* ===== HEADERS ===== */
    h1 {
        background: linear-gradient(135deg, #a78bfa 0%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }
    
    h2 {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
        border-bottom: 3px solid;
        border-image: linear-gradient(135deg, #667eea, #764ba2) 1;
        padding-bottom: 8px;
    }
    
    h3 {
        color: #cbd5e1 !important;
        font-weight: 600 !important;
    }
    
    /* ===== METRIC CARDS ===== */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(167, 139, 250, 0.2);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(167, 139, 250, 0.15);
        border-color: rgba(167, 139, 250, 0.4);
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #a78bfa !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 800 !important;
        color: #f1f5f9 !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* ===== DATAFRAMES ===== */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(17, 153, 142, 0.3);
    }
    
    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 10px;
        margin-top: 0.75rem;
        border-left-width: 5px;
    }
    
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    
    /* ===== SELECTBOX / SLIDERS ===== */
    .stSelectbox, .stSlider {
        font-weight: 500;
    }
    
    /* ===== PLOTLY CHARTS ===== */
    [data-testid="stPlotlyChart"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
        margin: 1.5rem 0;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }
    
    /* ===== EXPANDERS ===== */
    [data-testid="stExpander"] {
        border-radius: 10px;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    /* ===== FOOTER HIDE ===== */
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the telecom customer dataset."""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'master_dataset.csv')
        df = pd.read_csv(data_path)
        return df
    except FileNotFoundError:
        st.error("Dataset not found. Please run `python generate_data.py` first to download and process the Kaggle dataset.")
        return None

def calculate_business_metrics(df):
    """Calculate key business metrics."""
    if df is None:
        return {}
    
    metrics = {
        'total_customers': len(df),
        'total_revenue': df['arpu'].sum(),
        'avg_arpu': df['arpu'].mean(),
        'churn_rate': df['churned'].mean(),
        'avg_satisfaction': df['satisfaction_score'].mean(),
        'revenue_at_risk': df[df['churned'] == 1]['arpu'].sum(),
        'high_value_customers': (df['arpu'] > 75).sum(),
        'campaign_responsive': (df['total_conversions'] > 0).sum()
    }
    return metrics

def create_revenue_dashboard(df):
    """Create revenue analysis dashboard."""
    st.header("💰 Revenue Analysis")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_revenue = df['arpu'].sum()
        st.metric("Total Monthly Revenue", f"${total_revenue:,.2f}")
    
    with col2:
        avg_arpu = df['arpu'].mean()
        st.metric("Average ARPU", f"${avg_arpu:.2f}")
    
    with col3:
        high_value = (df['arpu'] > 75).sum()
        st.metric("High-Value Customers", f"{high_value:,}")
    
    with col4:
        revenue_at_risk = df[df['churned'] == 1]['arpu'].sum()
        st.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}")
    
    # Revenue visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # ARPU distribution
        fig_arpu = px.histogram(df, x='arpu', nbins=50, title='ARPU Distribution')
        fig_arpu.add_vline(x=avg_arpu, line_dash="dash", line_color="red", 
                          annotation_text=f"Mean: ${avg_arpu:.2f}")
        st.plotly_chart(fig_arpu, use_container_width=True)
        pct_below_avg = (df['arpu'] < avg_arpu).mean() * 100
        st.caption(f"💡 {pct_below_avg:.0f}% of customers fall below the average ARPU of ${avg_arpu:.2f}. Upselling opportunities exist in this segment.")
    
    with col2:
        # Revenue by plan type
        plan_revenue = df.groupby('plan_type').agg({
            'arpu': 'mean',
            'customer_id': 'count'
        }).reset_index()
        plan_revenue.columns = ['Plan Type', 'Avg ARPU', 'Customer Count']
        
        fig_plan = px.bar(plan_revenue, x='Plan Type', y='Avg ARPU', 
                         title='Average ARPU by Plan Type')
        st.plotly_chart(fig_plan, use_container_width=True)
        best_plan = plan_revenue.loc[plan_revenue['Avg ARPU'].idxmax()]
        st.caption(f"💡 {best_plan['Plan Type']} plan generates the highest ARPU at ${best_plan['Avg ARPU']:.2f} — a {((best_plan['Avg ARPU']/avg_arpu - 1)*100):.0f}% premium over the average.")
    
    # Geographic revenue analysis
    st.subheader("Geographic Revenue Distribution")
    city_revenue = df.groupby('city').agg({
        'arpu': ['mean', 'sum'],
        'customer_id': 'count'
    }).round(2)
    city_revenue.columns = ['Avg ARPU', 'Total Revenue', 'Customer Count']
    city_revenue = city_revenue.sort_values('Total Revenue', ascending=False)
    
    fig_geo = px.bar(city_revenue.head(10), x=city_revenue.head(10).index, 
                     y='Total Revenue', title='Top 10 Cities by Total Revenue')
    st.plotly_chart(fig_geo, use_container_width=True)
    top_city = city_revenue.index[0]
    top_pct = city_revenue.iloc[0]['Total Revenue'] / city_revenue['Total Revenue'].sum() * 100
    st.caption(f"💡 {top_city} leads with ${city_revenue.iloc[0]['Total Revenue']:,.0f} in revenue ({top_pct:.1f}% of total). Focus retention efforts on top-performing cities to protect this revenue.")

def create_churn_dashboard(df):
    """Create churn analysis dashboard."""
    st.header("🔄 Churn Analysis")
    
    # Churn metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = df['churned'].mean()
        st.metric("Overall Churn Rate", f"{churn_rate:.2%}")
    
    with col2:
        churned_customers = df['churned'].sum()
        st.metric("Churned Customers", f"{churned_customers:,}")
    
    with col3:
        churned_customers = df[df['churned'] == 1]
        if len(churned_customers) > 0:
            avg_churn_arpu = churned_customers['arpu'].mean()
            st.metric("Avg ARPU (Churned)", f"${avg_churn_arpu:.2f}")
        else:
            st.metric("Avg ARPU (Churned)", "N/A")
    
    with col4:
        high_risk_churn = ((df['arpu'] > 75) & (df['churned'] == 1)).sum()
        st.metric("High-Value Churned", f"{high_risk_churn:,}")
    
    # Churn analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Churn by satisfaction score - handle empty bins
        try:
            satisfaction_bins = pd.cut(df['satisfaction_score'], bins=5)
            churn_by_satisfaction = df.groupby(satisfaction_bins)['churned'].mean()
            
            fig_satisfaction = px.bar(
                x=[str(x) for x in churn_by_satisfaction.index],
                y=churn_by_satisfaction.values * 100,
                title='Churn Rate by Satisfaction Score Range',
                labels={'x': 'Satisfaction Score Range', 'y': 'Churn Rate (%)'}
            )
            st.plotly_chart(fig_satisfaction, use_container_width=True)
            low_sat_churn = df[df['satisfaction_score'] <= 4]['churned'].mean() * 100
            high_sat_churn = df[df['satisfaction_score'] >= 8]['churned'].mean() * 100
            st.caption(f"💡 Low-satisfaction customers (≤4) churn at {low_sat_churn:.0f}% vs {high_sat_churn:.0f}% for satisfied ones (≥8) — a {low_sat_churn/max(high_sat_churn,1):.1f}x gap.")
        except ValueError as e:
            st.warning("Unable to create satisfaction score chart: " + str(e))
    
    with col2:
        # Churn by tenure - handle empty bins
        try:
            tenure_bins = pd.cut(df['tenure_months'], bins=5)
            churn_by_tenure = df.groupby(tenure_bins)['churned'].mean()
            
            fig_tenure = px.bar(
                x=[str(x) for x in churn_by_tenure.index],
                y=churn_by_tenure.values * 100,
                title='Churn Rate by Tenure Range',
                labels={'x': 'Tenure Range (months)', 'y': 'Churn Rate (%)'}
            )
            st.plotly_chart(fig_tenure, use_container_width=True)
            new_churn = df[df['tenure_months'] <= 12]['churned'].mean() * 100
            loyal_churn = df[df['tenure_months'] >= 48]['churned'].mean() * 100
            st.caption(f"💡 New customers (≤12 months) churn at {new_churn:.0f}% vs {loyal_churn:.0f}% for long-term (≥48 months). First-year retention is critical.")
        except ValueError as e:
            st.warning("Unable to create tenure chart: " + str(e))
    
    # Risk segmentation
    st.subheader("Customer Risk Segmentation")
    
    # Create risk scores safely
    satisfaction_component = (10 - df['satisfaction_score']) * 0.4
    complaints_component = df['num_complaints_12m'] * 0.3
    late_payments_component = df['late_payments_12m'] * 0.3
    
    df['risk_score'] = satisfaction_component + complaints_component + late_payments_component
    
    # Risk categories
    df['risk_category'] = pd.cut(df['risk_score'], 
                                bins=[0, 2, 4, 6, float('inf')],
                                labels=['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk'])
    
    risk_analysis = df.groupby('risk_category').agg({
        'customer_id': 'count',
        'churned': 'mean',
        'arpu': 'mean'
    }).round(3)
    risk_analysis.columns = ['Customer Count', 'Churn Rate', 'Avg ARPU']
    
    st.dataframe(risk_analysis)

def create_campaign_dashboard(df):
    """Create campaign effectiveness dashboard."""
    st.header("📢 Campaign Effectiveness")
    
    # Campaign metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_campaigns = df['campaigns_exposed'].mean()
        st.metric("Avg Campaigns/Customer", f"{avg_campaigns:.1f}")
    
    with col2:
        total_impressions = df['total_impressions'].sum()
        total_clicks = df['total_clicks'].sum()
        if total_impressions > 0:
            overall_ctr = total_clicks / total_impressions
            st.metric("Overall CTR", f"{overall_ctr:.2%}")
        else:
            st.metric("Overall CTR", "N/A")
    
    with col3:
        total_clicks = df['total_clicks'].sum()
        total_conversions = df['total_conversions'].sum()
        if total_clicks > 0:
            conversion_rate = total_conversions / total_clicks
            st.metric("Conversion Rate", f"{conversion_rate:.2%}")
        else:
            st.metric("Conversion Rate", "N/A")
    
    with col4:
        responsive_customers = (df['total_conversions'] > 0).sum()
        st.metric("Responsive Customers", f"{responsive_customers:,}")
    
    # Campaign performance by channel
    st.subheader("Channel Performance Analysis")
    
    channel_performance = df.groupby('primary_channel').agg({
        'total_impressions': 'sum',
        'total_clicks': 'sum',
        'total_conversions': 'sum'
    }).reset_index()
    
    # Safely calculate CTR and Conversion Rate
    channel_performance['CTR'] = np.where(
        channel_performance['total_impressions'] > 0,
        channel_performance['total_clicks'] / channel_performance['total_impressions'],
        0
    )
    channel_performance['Conversion_Rate'] = np.where(
        channel_performance['total_clicks'] > 0,
        channel_performance['total_conversions'] / channel_performance['total_clicks'],
        0
    )
    
    # Remove channels with no data
    channel_performance = channel_performance[channel_performance['total_impressions'] > 0].copy()
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_ctr = px.bar(channel_performance.sort_values('CTR', ascending=False),
                        x='primary_channel', y='CTR',
                        title='Click-Through Rate by Channel')
        fig_ctr.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig_ctr, use_container_width=True)
        best_ctr = channel_performance.loc[channel_performance['CTR'].idxmax()]
        st.caption(f"💡 {best_ctr['primary_channel']} has the highest CTR at {best_ctr['CTR']:.1%} — prioritize this channel for awareness campaigns.")
    
    with col2:
        fig_conv = px.bar(channel_performance.sort_values('Conversion_Rate', ascending=False),
                         x='primary_channel', y='Conversion_Rate',
                         title='Conversion Rate by Channel')
        fig_conv.update_layout(yaxis_tickformat='.2%')
        st.plotly_chart(fig_conv, use_container_width=True)
        best_conv = channel_performance.loc[channel_performance['Conversion_Rate'].idxmax()]
        st.caption(f"💡 {best_conv['primary_channel']} converts at {best_conv['Conversion_Rate']:.1%} — allocate retention budget here for maximum ROI.")

def create_customer_insights(df):
    """Create customer insights dashboard."""
    st.header("👥 Customer Insights")
    
    # Customer segmentation
    st.subheader("Customer Segmentation")
    
    # Value-based segmentation
    def assign_value_segment(row):
        if row['arpu'] >= 80 and row['satisfaction_score'] >= 8:
            return 'Champions'
        elif row['arpu'] >= 80 and row['satisfaction_score'] >= 6:
            return 'Loyal Customers'
        elif row['arpu'] >= 60 and row['satisfaction_score'] >= 7:
            return 'Potential Loyalists'
        elif row['arpu'] >= 40 and row['satisfaction_score'] >= 6:
            return 'New Customers'
        elif row['arpu'] >= 40 and row['satisfaction_score'] < 6:
            return 'At Risk'
        else:
            return 'Need Attention'
    
    df['customer_segment'] = df.apply(assign_value_segment, axis=1)
    
    # Segment distribution
    segment_counts = df['customer_segment'].value_counts()
    fig_segments = px.pie(values=segment_counts.values, names=segment_counts.index,
                         title='Customer Segment Distribution')
    st.plotly_chart(fig_segments, use_container_width=True)
    champions_pct = segment_counts.get('Champions', 0) / len(df) * 100
    at_risk_pct = (segment_counts.get('At Risk', 0) + segment_counts.get('Need Attention', 0)) / len(df) * 100
    st.caption(f"💡 Champions make up {champions_pct:.1f}% of customers while {at_risk_pct:.1f}% need attention — targeted engagement can shift at-risk customers towards loyalty.")
    
    # Segment analysis table
    segment_analysis = df.groupby('customer_segment').agg({
        'customer_id': 'count',
        'arpu': 'mean',
        'churned': 'mean',
        'satisfaction_score': 'mean',
        'total_conversions': 'mean'
    }).round(2)
    segment_analysis.columns = ['Count', 'Avg ARPU', 'Churn Rate', 'Avg Satisfaction', 'Avg Conversions']
    
    st.subheader("Segment Performance Metrics")
    st.dataframe(segment_analysis)
    
    # Usage patterns
    st.subheader("Usage Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data usage by age group
        df['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 60, 100], 
                                labels=['18-30', '31-45', '46-60', '60+'])
        age_usage = df.groupby('age_group')['monthly_data_gb'].mean()
        
        fig_age_usage = px.bar(x=age_usage.index, y=age_usage.values,
                              title='Average Data Usage by Age Group',
                              labels={'x': 'Age Group', 'y': 'Data Usage (GB)'})
        st.plotly_chart(fig_age_usage, use_container_width=True)
    
    with col2:
        # Digital engagement
        df['total_sessions'] = df['monthly_web_sessions'] + df['monthly_app_sessions']
        engagement_income = df.groupby(pd.qcut(df['annual_income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4']))['total_sessions'].mean()
        
        fig_engagement = px.bar(x=engagement_income.index, y=engagement_income.values,
                               title='Digital Engagement by Income Quartile',
                               labels={'x': 'Income Quartile', 'y': 'Monthly Sessions'})
        st.plotly_chart(fig_engagement, use_container_width=True)

def create_predictive_insights(df):
    """Create predictive insights dashboard."""
    st.header("🔮 Predictive Insights")
    
    # Score customers with the trained churn pipeline (train_model.py)
    try:
        from churn_pipeline import load_churn_artifact
        artifact = load_churn_artifact()
    except FileNotFoundError:
        st.warning("Churn model not found. Run `python train_model.py` first.")
        return
    df['churn_probability'] = artifact['pipeline'].predict_proba(df[artifact['feature_columns']])[:, 1]
    st.caption(f"Scores from the trained churn model (held-out ROC-AUC {artifact['metrics']['roc_auc']:.3f}).")
    
    # High-risk customers
    high_risk_threshold = 0.7
    high_risk_customers = df[df['churn_probability'] > high_risk_threshold]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("High-Risk Customers", f"{len(high_risk_customers):,}")
    
    with col2:
        risk_revenue = high_risk_customers['arpu'].sum()
        st.metric("Revenue at Risk", f"${risk_revenue:,.2f}")
    
    with col3:
        if len(high_risk_customers) > 0:
            avg_risk_arpu = high_risk_customers['arpu'].mean()
            st.metric("Avg ARPU (High-Risk)", f"${avg_risk_arpu:.2f}")
    
    # Churn probability distribution
    fig_churn_dist = px.histogram(df, x='churn_probability', nbins=50,
                                 title='Churn Probability Distribution')
    fig_churn_dist.add_vline(x=high_risk_threshold, line_dash="dash", line_color="red",
                            annotation_text="High Risk Threshold")
    st.plotly_chart(fig_churn_dist, use_container_width=True)
    
    # Top at-risk customers
    st.subheader("Top 10 At-Risk High-Value Customers")
    at_risk_high_value = df[(df['churn_probability'] > 0.5) & (df['arpu'] > 60)].nlargest(10, 'churn_probability')
    
    display_cols = ['customer_id', 'arpu', 'satisfaction_score', 'tenure_months', 'churn_probability']
    st.dataframe(at_risk_high_value[display_cols])

def create_retention_engine_dashboard(df):
    """Smart Retention Engine — the crown jewel of the project."""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    st.header("🎯 Smart Retention Engine")
    st.markdown("**From Prediction → Prescription → Revenue Impact**")
    st.markdown("This engine goes beyond churn prediction to recommend **specific retention actions** with ROI for each customer.")
    
    # Check if retention analysis exists, otherwise run it
    retention_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'retention_analysis.csv')
    
    if os.path.exists(retention_path):
        df_ret = pd.read_csv(retention_path)
    else:
        st.warning("Retention analysis not found. Running engine now (this takes ~2 minutes)...")
        try:
            from retention_engine import run_full_retention_analysis
            data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'master_dataset.csv')
            df_full, summary, metrics = run_full_retention_analysis(data_path)
            output_cols = [
                'customer_id', 'age', 'gender', 'city', 'plan_type', 'tenure_months',
                'arpu', 'churned', 'customer_health_score', 'health_category',
                'churn_driver_label', 'churn_driver_score', 'ml_churn_probability',
                'strategy_name', 'intervention_cost', 'expected_revenue_saved',
                'expected_roi', 'retention_priority_rank'
            ]
            available_cols = [c for c in output_cols if c in df_full.columns]
            os.makedirs(os.path.dirname(retention_path), exist_ok=True)
            df_full[available_cols].to_csv(retention_path, index=False)
            df_ret = df_full[available_cols]
        except Exception as e:
            st.error(f"Failed to run retention engine: {e}")
            st.info("Run `python -c \"from src.retention_engine import run_full_retention_analysis; run_full_retention_analysis()\"` first.")
            return
    
    # ---------- KEY METRICS ROW ----------
    at_risk = df_ret[df_ret['strategy_name'] != 'No Intervention Needed'] if 'strategy_name' in df_ret.columns else df_ret
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        avg_health = df_ret['customer_health_score'].mean() if 'customer_health_score' in df_ret.columns else 0
        st.metric("Avg Health Score", f"{avg_health:.1f}/100")
    with col2:
        st.metric("At-Risk Customers", f"{len(at_risk):,}")
    with col3:
        total_cost = at_risk['intervention_cost'].sum() if 'intervention_cost' in at_risk.columns else 0
        st.metric("Total Intervention Cost", f"${total_cost:,.0f}")
    with col4:
        total_saved = at_risk['expected_revenue_saved'].sum() if 'expected_revenue_saved' in at_risk.columns else 0
        st.metric("Expected Revenue Saved", f"${total_saved:,.0f}")
    with col5:
        roi = (total_saved - total_cost) / max(total_cost, 1)
        st.metric("Overall ROI", f"{roi:.1f}x")
    
    st.markdown("---")
    
    # ---------- TWO-COLUMN LAYOUT ----------
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Health Category Distribution
        st.subheader("Customer Health Distribution")
        if 'health_category' in df_ret.columns:
            health_counts = df_ret['health_category'].value_counts()
            colors_map = {'Critical': '#d32f2f', 'At Risk': '#ff9800', 'Stable': '#4caf50', 'Healthy': '#2196f3'}
            colors = [colors_map.get(cat, '#999') for cat in health_counts.index]
            fig_health = px.pie(values=health_counts.values, names=health_counts.index,
                               color=health_counts.index,
                               color_discrete_map=colors_map,
                               title='Health Category Breakdown')
            st.plotly_chart(fig_health, use_container_width=True)
    
    with col_right:
        # Churn Driver Distribution
        st.subheader("Why Customers Leave")
        if 'churn_driver_label' in df_ret.columns:
            driver_counts = df_ret['churn_driver_label'].value_counts().head(7)
            fig_drivers = px.bar(x=driver_counts.values, y=driver_counts.index,
                                orientation='h', title='Primary Churn Drivers',
                                labels={'x': 'Customers', 'y': 'Driver'},
                                color=driver_counts.values,
                                color_continuous_scale='Reds')
            fig_drivers.update_layout(showlegend=False)
            st.plotly_chart(fig_drivers, use_container_width=True)
    
    # ---------- STRATEGY ANALYSIS ----------
    st.subheader("🏷️ Retention Strategy Allocation")
    
    if 'strategy_name' in df_ret.columns:
        strat_summary = df_ret.groupby('strategy_name').agg({
            'customer_id': 'count',
            'intervention_cost': 'sum',
            'expected_revenue_saved': 'sum',
            'expected_roi': 'mean'
        }).reset_index()
        strat_summary.columns = ['Strategy', 'Customers', 'Total Cost ($)', 'Revenue Saved ($)', 'Avg ROI (x)']
        strat_summary = strat_summary.sort_values('Revenue Saved ($)', ascending=False)
        st.dataframe(strat_summary.style.format({
            'Total Cost ($)': '${:,.0f}',
            'Revenue Saved ($)': '${:,.0f}',
            'Avg ROI (x)': '{:.1f}x'
        }), use_container_width=True)
    
    # ---------- TOP PRIORITY CUSTOMERS ----------
    st.subheader("🔥 Top 20 Priority Customers")
    st.markdown("*Ranked by expected revenue impact (ARPU × churn probability × retention boost)*")
    
    if 'retention_priority_rank' in df_ret.columns:
        top20 = df_ret.nsmallest(20, 'retention_priority_rank')
        display_cols_ret = ['customer_id', 'arpu', 'ml_churn_probability', 
                           'customer_health_score', 'churn_driver_label',
                           'strategy_name', 'expected_revenue_saved', 'expected_roi']
        available = [c for c in display_cols_ret if c in top20.columns]
        st.dataframe(top20[available].style.format({
            'arpu': '${:.2f}',
            'ml_churn_probability': '{:.1%}',
            'customer_health_score': '{:.1f}',
            'expected_revenue_saved': '${:.0f}',
            'expected_roi': '{:.1f}x'
        }), use_container_width=True)
    
    # ---------- BUSINESS IMPACT ----------
    st.subheader("💰 Business Impact Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"💡 **Net Benefit**: ${total_saved - total_cost:,.0f} (annual revenue protected minus costs)")
        st.info(f"📊 **Model AUC-ROC**: High confidence in churn predictions")
    with col2:
        st.warning(f"🎯 **Actionable**: {len(at_risk):,} customers have specific retention strategies assigned")
        st.info(f"⚡ **Prioritized**: Customers ranked by expected revenue impact, not just churn risk")

def create_revenue_simulator(df):
    """Interactive Revenue Impact Simulator with what-if sliders."""
    import os
    
    st.header("💰 Revenue Impact Simulator")
    st.markdown("**What-If Analysis: Simulate the financial impact of retention and growth strategies**")
    st.markdown("Drag the sliders to see how different scenarios affect your bottom line.")
    
    # Load retention data for baseline metrics
    retention_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'retention_analysis.csv')
    if os.path.exists(retention_path):
        df_ret = pd.read_csv(retention_path)
    else:
        df_ret = df.copy()
        # No model scores available: fall back to the portfolio churn rate, never the label itself
        df_ret['ml_churn_probability'] = df['churned'].mean() if 'churned' in df.columns else 0.26
    
    # ===== BASELINE METRICS =====
    total_customers = len(df)
    current_churn_rate = df['churned'].mean() if 'churned' in df.columns else 0.26
    current_arpu = df['arpu'].mean()
    monthly_revenue = df['arpu'].sum()
    annual_revenue = monthly_revenue * 12
    churned_customers = int(total_customers * current_churn_rate)
    monthly_revenue_at_risk = df[df['churned'] == 1]['arpu'].sum() if 'churned' in df.columns else monthly_revenue * current_churn_rate
    
    st.markdown("---")
    
    # ===== BASELINE DISPLAY =====
    st.subheader("📊 Current Baseline (Real Data)")
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.metric("Monthly Revenue", f"${monthly_revenue:,.0f}")
    with bcol2:
        st.metric("Annual Revenue", f"${annual_revenue:,.0f}")
    with bcol3:
        st.metric("Churn Rate", f"{current_churn_rate:.1%}")
    with bcol4:
        st.metric("Avg ARPU", f"${current_arpu:.2f}")
    
    st.markdown("---")
    
    # ===== SCENARIO SLIDERS =====
    st.subheader("🎛️ Scenario Controls")
    st.markdown("*Adjust these sliders to simulate different business strategies:*")
    
    slider_col1, slider_col2, slider_col3 = st.columns(3)
    
    with slider_col1:
        churn_reduction = st.slider(
            "🔄 Churn Reduction (%)",
            min_value=0, max_value=50, value=15, step=5,
            help="How much churn can we reduce through retention programs?"
        )
    
    with slider_col2:
        arpu_increase = st.slider(
            "📈 ARPU Increase (%)",
            min_value=0, max_value=30, value=5, step=1,
            help="Expected ARPU growth from upselling and plan upgrades"
        )
    
    with slider_col3:
        retention_budget = st.slider(
            "💵 Retention Budget ($K/month)",
            min_value=0, max_value=100, value=20, step=5,
            help="Monthly investment in retention programs"
        )
    
    # ===== SCENARIO CALCULATIONS =====
    new_churn_rate = current_churn_rate * (1 - churn_reduction / 100)
    new_arpu = current_arpu * (1 + arpu_increase / 100)
    
    # Customers retained by reducing churn
    customers_saved = int(churned_customers * (churn_reduction / 100))
    
    # New monthly revenue
    retained_customers = total_customers - int(total_customers * new_churn_rate)
    new_monthly_revenue = retained_customers * new_arpu
    monthly_budget = retention_budget * 1000
    
    # Revenue impact
    revenue_from_retained = customers_saved * current_arpu
    revenue_from_arpu_lift = (total_customers - churned_customers + customers_saved) * (new_arpu - current_arpu)
    total_monthly_impact = revenue_from_retained + revenue_from_arpu_lift - monthly_budget
    total_annual_impact = total_monthly_impact * 12
    
    # ROI
    annual_investment = monthly_budget * 12
    annual_revenue_gain = (revenue_from_retained + revenue_from_arpu_lift) * 12
    roi = (annual_revenue_gain - annual_investment) / max(annual_investment, 1)
    
    st.markdown("---")
    
    # ===== SCENARIO RESULTS =====
    st.subheader("📈 Projected Impact")
    
    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
    
    with rcol1:
        st.metric("New Churn Rate", f"{new_churn_rate:.1%}",
                 delta=f"{-(current_churn_rate - new_churn_rate)*100:.1f}pp")
    with rcol2:
        st.metric("Customers Saved", f"{customers_saved:,}",
                 delta=f"+{customers_saved:,}")
    with rcol3:
        st.metric("New Avg ARPU", f"${new_arpu:.2f}",
                 delta=f"+${new_arpu - current_arpu:.2f}")
    with rcol4:
        st.metric("Monthly Revenue Gain", f"${total_monthly_impact:,.0f}",
                 delta=f"${total_monthly_impact:+,.0f}")
    with rcol5:
        delta_color = "normal" if roi > 0 else "inverse"
        st.metric("Annual ROI", f"{roi:.1f}x")
    
    # ===== 12-MONTH PROJECTION CHART =====
    st.subheader("📊 12-Month Revenue Projection")
    
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Baseline: declining revenue due to churn
    baseline_revenue = []
    scenario_revenue = []
    cumulative_saved = []
    cumulative_total = 0
    
    for m in months:
        # Baseline: lose customers each month at current churn rate
        remaining_base = total_customers * (1 - current_churn_rate / 12 * m)
        baseline_rev = remaining_base * current_arpu
        baseline_revenue.append(baseline_rev)
        
        # Scenario: reduced churn + higher ARPU (gradual ramp)
        ramp = min(m / 3, 1.0)  # Full effect after 3 months
        effective_churn_reduction = churn_reduction * ramp
        effective_arpu_increase = arpu_increase * ramp
        
        new_churn_m = current_churn_rate * (1 - effective_churn_reduction / 100)
        remaining_scenario = total_customers * (1 - new_churn_m / 12 * m)
        new_arpu_m = current_arpu * (1 + effective_arpu_increase / 100)
        scenario_rev = remaining_scenario * new_arpu_m - monthly_budget
        scenario_revenue.append(scenario_rev)
        
        cumulative_total += (scenario_rev - baseline_rev)
        cumulative_saved.append(cumulative_total)
    
    fig_projection = go.Figure()
    
    fig_projection.add_trace(go.Scatter(
        x=month_names, y=baseline_revenue,
        name='Baseline (No Action)',
        line=dict(color='#ef5350', width=3, dash='dash'),
        fill=None
    ))
    
    fig_projection.add_trace(go.Scatter(
        x=month_names, y=scenario_revenue,
        name='With Retention Strategy',
        line=dict(color='#4caf50', width=3),
        fill='tonexty',
        fillcolor='rgba(76, 175, 80, 0.15)'
    ))
    
    fig_projection.update_layout(
        title='Monthly Revenue: Baseline vs Strategy',
        xaxis_title='Month',
        yaxis_title='Monthly Revenue ($)',
        yaxis_tickformat='$,.0f',
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_projection, use_container_width=True)
    
    # ===== SCENARIO COMPARISON TABLE =====
    st.subheader("📋 Scenario Comparison")
    
    comparison_data = {
        'Metric': [
            'Monthly Revenue',
            'Annual Revenue',
            'Churn Rate',
            'Customers Lost/Year',
            'Average ARPU',
            'Annual Retention Cost',
            'Net Annual Impact'
        ],
        'Current (Baseline)': [
            f'${monthly_revenue:,.0f}',
            f'${annual_revenue:,.0f}',
            f'{current_churn_rate:.1%}',
            f'{churned_customers:,}',
            f'${current_arpu:.2f}',
            '$0',
            '$0'
        ],
        'With Strategy': [
            f'${new_monthly_revenue:,.0f}',
            f'${new_monthly_revenue * 12:,.0f}',
            f'{new_churn_rate:.1%}',
            f'{int(total_customers * new_churn_rate):,}',
            f'${new_arpu:.2f}',
            f'${annual_investment:,.0f}',
            f'${total_annual_impact:,.0f}'
        ],
        'Difference': [
            f'+${new_monthly_revenue - monthly_revenue:,.0f}',
            f'+${(new_monthly_revenue * 12) - annual_revenue:,.0f}',
            f'{(new_churn_rate - current_churn_rate)*100:+.1f}pp',
            f'{int(total_customers * new_churn_rate) - churned_customers:+,}',
            f'+${new_arpu - current_arpu:.2f}',
            f'-${annual_investment:,.0f}',
            f'+${total_annual_impact:,.0f}'
        ]
    }
    
    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
    
    # ===== ROI SUMMARY BOX =====
    st.markdown("---")
    st.subheader("🏆 ROI Summary")
    
    roi_col1, roi_col2, roi_col3 = st.columns(3)
    
    with roi_col1:
        if total_annual_impact > 0:
            st.success(f"✅ **Profitable Strategy!** Investing ${annual_investment:,.0f}/year generates ${total_annual_impact:,.0f} in net revenue impact.")
        else:
            st.error(f"⚠️ **Negative ROI** at current settings. Try reducing budget or increasing churn reduction target.")
    
    with roi_col2:
        break_even_months = annual_investment / max(total_monthly_impact, 1) if total_monthly_impact > 0 else float('inf')
        if break_even_months < 12:
            st.info(f"📅 **Break-even**: {break_even_months:.1f} months")
        else:
            st.warning(f"📅 **Break-even**: >12 months")
    
    with roi_col3:
        cost_per_saved = annual_investment / max(customers_saved * 12, 1)
        st.info(f"💰 **Cost per saved customer**: ${cost_per_saved:.2f}/month")

def create_nbo_dashboard(df):
    """Next Best Offer prediction dashboard."""
    import os
    
    st.header("🔮 Next Best Offer Predictions")
    st.markdown("**ML-powered recommendations: Which product/service should we offer each customer?**")
    st.markdown("Each customer gets personalized offer recommendations ranked by expected revenue potential.")
    
    # Load pre-computed NBO data
    nbo_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'nbo_recommendations.csv')
    nbo_all_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'nbo_all_offers.csv')
    
    if not os.path.exists(nbo_path):
        st.warning("⚠️ NBO analysis not found. Run the NBO engine first.")
        st.code('python -c "from src.nbo_engine import run_full_nbo_analysis; run_full_nbo_analysis()"', language='bash')
        return
    
    best_offers = pd.read_csv(nbo_path)
    
    # ===== KEY METRICS =====
    total_revenue_potential = best_offers['expected_annual_value'].sum()
    avg_conversion = best_offers['conversion_prob'].mean()
    customers_with_offers = best_offers['customer_id'].nunique()
    avg_uplift = best_offers['monthly_uplift'].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Revenue Opportunity", f"${total_revenue_potential:,.0f}/yr")
    with col2:
        st.metric("🎯 Avg Conversion Rate", f"{avg_conversion:.1%}")
    with col3:
        st.metric("👥 Customers with Offers", f"{customers_with_offers:,}")
    with col4:
        st.metric("⚡ Avg Monthly Uplift", f"${avg_uplift:.0f}")
    
    st.markdown("---")
    
    # ===== TWO COLUMN CHARTS =====
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Best Offer Distribution")
        offer_counts = best_offers['offer_name'].value_counts()
        fig_dist = px.pie(values=offer_counts.values, names=offer_counts.index,
                         title='Top Recommended Offers',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col_right:
        st.subheader("Revenue Potential by Offer")
        rev_by_offer = best_offers.groupby('offer_name')['expected_annual_value'].sum().sort_values(ascending=True)
        fig_rev = px.bar(x=rev_by_offer.values, y=rev_by_offer.index,
                        orientation='h', title='Expected Annual Revenue by Offer Type',
                        labels={'x': 'Expected Annual Revenue ($)', 'y': 'Offer'},
                        color=rev_by_offer.values,
                        color_continuous_scale='Viridis')
        fig_rev.update_layout(showlegend=False, yaxis_tickfont_size=9)
        st.plotly_chart(fig_rev, use_container_width=True)
    
    # ===== OFFER SUMMARY TABLE =====
    st.subheader("📋 Offer Performance Summary")
    offer_summary = best_offers.groupby('offer_name').agg({
        'customer_id': 'count',
        'conversion_prob': 'mean',
        'monthly_uplift': 'first',
        'expected_annual_value': ['sum', 'mean']
    }).reset_index()
    offer_summary.columns = ['Offer', 'Customers', 'Avg Conv. Rate', 
                            'Monthly Uplift ($)', 'Total Revenue ($)', 'Avg Revenue/Customer ($)']
    offer_summary = offer_summary.sort_values('Total Revenue ($)', ascending=False)
    st.dataframe(offer_summary.style.format({
        'Avg Conv. Rate': '{:.1%}',
        'Monthly Uplift ($)': '${:.0f}',
        'Total Revenue ($)': '${:,.0f}',
        'Avg Revenue/Customer ($)': '${:.2f}'
    }), use_container_width=True)
    
    # ===== TOP 20 OPPORTUNITIES =====
    st.subheader("🔥 Top 20 Revenue Opportunities")
    st.markdown("*Customers with the highest expected value from their recommended offer*")
    
    top20 = best_offers.nlargest(20, 'expected_annual_value')
    display_cols = ['customer_id', 'offer_name', 'conversion_prob', 
                   'monthly_uplift', 'expected_annual_value']
    available = [c for c in display_cols if c in top20.columns]
    st.dataframe(top20[available].style.format({
        'conversion_prob': '{:.1%}',
        'monthly_uplift': '${:.0f}/mo',
        'expected_annual_value': '${:,.0f}/yr'
    }), use_container_width=True)
    
    # ===== IMPACT SUMMARY =====
    st.markdown("---")
    st.subheader("🏆 Growth Impact Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success(f"💡 **{customers_with_offers:,}** customers have personalized offer recommendations")
    with col2:
        st.info(f"💰 **${total_revenue_potential:,.0f}** total expected annual revenue from NBO")
    with col3:
        st.warning(f"🎯 Average **{avg_conversion:.1%}** predicted acceptance rate across all offers")

def create_explainability_dashboard(df):
    """🧠 SHAP Model Explainability Dashboard."""
    import os
    
    st.header("🧠 Model Explainability (SHAP)")
    st.markdown("**Why does the model predict churn? Which features matter most?**")
    st.markdown("This answers the critical interview question: *'How do you explain your model to business stakeholders?'*")
    
    # Load SHAP data
    importance_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'shap_importance.csv')
    shap_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'shap_values.csv')
    
    if not os.path.exists(importance_path):
        st.warning("⚠️ SHAP analysis not found. Run the explainability engine first.")
        st.code('python -c "from src.explainability import run_explainability_analysis; run_explainability_analysis()"', language='bash')
        return
    
    importance = pd.read_csv(importance_path, index_col=0, header=None)
    importance.columns = ['importance']
    importance = importance.sort_values('importance', ascending=True)
    
    # ===== GLOBAL FEATURE IMPORTANCE =====
    st.subheader("🎯 Global Feature Importance")
    st.markdown("*Mean absolute SHAP value — how much each feature contributes to predictions on average*")
    
    top_n = min(15, len(importance))
    top_features = importance.tail(top_n)
    
    fig_imp = px.bar(x=top_features['importance'].values, y=top_features.index,
                    orientation='h', title=f'Top {top_n} Features Driving Churn Prediction',
                    labels={'x': 'Mean |SHAP Value| (Impact on Prediction)', 'y': 'Feature'},
                    color=top_features['importance'].values,
                    color_continuous_scale='Reds')
    fig_imp.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig_imp, use_container_width=True)
    
    # ===== SHAP BEESWARM (scatter) =====
    if os.path.exists(shap_path):
        shap_df = pd.read_csv(shap_path)
        
        st.subheader("🔍 Feature Impact Deep Dive")
        
        top_features_list = importance.tail(top_n).index.tolist()[::-1]
        selected_feature = st.selectbox("Select a feature to analyze:", top_features_list)
        
        if selected_feature in shap_df.columns and selected_feature in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                fig_dep = px.scatter(
                    x=df[selected_feature].head(len(shap_df)),
                    y=shap_df[selected_feature],
                    title=f'Impact of {selected_feature} on Churn Prediction',
                    labels={'x': selected_feature, 'y': 'SHAP Value (churn impact)'},
                    color=shap_df[selected_feature],
                    color_continuous_scale='RdBu_r',
                    opacity=0.5
                )
                fig_dep.add_hline(y=0, line_dash='dash', line_color='gray')
                st.plotly_chart(fig_dep, use_container_width=True)
            with col2:
                fig_box = px.box(
                    x=df['churned'].head(len(shap_df)).map({0: 'Retained', 1: 'Churned'}),
                    y=df[selected_feature].head(len(shap_df)),
                    title=f'{selected_feature} by Churn Status',
                    labels={'x': 'Customer Status', 'y': selected_feature},
                    color=df['churned'].head(len(shap_df)).map({0: 'Retained', 1: 'Churned'})
                )
                st.plotly_chart(fig_box, use_container_width=True)
    
    # ===== KEY INSIGHT =====
    st.markdown("---")
    st.subheader("💡 Key Insights")
    top3 = importance.tail(3).index.tolist()[::-1]
    st.success(f"The **top 3 churn drivers** are: **{top3[0]}**, **{top3[1]}**, and **{top3[2]}**")
    st.info("SHAP values show both **direction** (positive = increases churn) and **magnitude** (how much it matters).")

def create_cohort_dashboard(df):
    """📅 Customer Cohort Analysis Dashboard."""
    import os
    
    st.header("📅 Customer Cohort Analysis")
    st.markdown("**How do different customer groups behave over their lifecycle?**")
    
    # Load cohort data
    cohort_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cohort_retention.csv')
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cohort_metrics.csv')
    heatmap_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'cohort_heatmap.csv')
    
    if not os.path.exists(cohort_path):
        st.warning("⚠️ Cohort analysis not found. Run the cohort engine first.")
        st.code('python -c "from src.cohort_analysis import run_cohort_analysis; run_cohort_analysis()"', language='bash')
        return
    
    cohort_ret = pd.read_csv(cohort_path)
    
    # ===== KEY METRICS =====
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Cohorts", f"{len(cohort_ret)}")
    with col2:
        st.metric("Overall Retention", f"{cohort_ret['retention_rate'].mean():.1%}")
    with col3:
        best = cohort_ret.loc[cohort_ret['retention_rate'].idxmax()]
        st.metric("Best Cohort", f"{best['cohort']}", delta=f"{best['retention_rate']:.0%} ret.")
    with col4:
        worst = cohort_ret.loc[cohort_ret['retention_rate'].idxmin()]
        st.metric("Worst Cohort", f"{worst['cohort']}", delta=f"{worst['retention_rate']:.0%} ret.")
    
    st.markdown("---")
    
    # ===== RETENTION BY COHORT =====
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Retention Rate by Cohort")
        # Show top 20 cohorts by size
        top_cohorts = cohort_ret.nlargest(20, 'total_customers')
        fig_ret = px.bar(top_cohorts, x='cohort', y='retention_rate',
                        title='Retention Rate by Join Cohort (Top 20 by Size)',
                        labels={'cohort': 'Cohort', 'retention_rate': 'Retention Rate'},
                        color='retention_rate',
                        color_continuous_scale='RdYlGn')
        fig_ret.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_ret, use_container_width=True)
    
    with col_right:
        st.subheader("ARPU by Cohort")
        fig_arpu = px.bar(top_cohorts, x='cohort', y='avg_arpu',
                         title='Average ARPU by Cohort (Top 20 by Size)',
                         labels={'cohort': 'Cohort', 'avg_arpu': 'Avg ARPU ($)'},
                         color='avg_arpu',
                         color_continuous_scale='Viridis')
        fig_arpu.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_arpu, use_container_width=True)
    
    # ===== HEATMAP =====
    if os.path.exists(heatmap_path):
        heatmap_data = pd.read_csv(heatmap_path)
        
        if len(heatmap_data) > 0:
            st.subheader("🗺️ Retention Heatmap")
            
            pivot = heatmap_data.pivot_table(
                index='cohort', columns='tenure_band', 
                values='retention_rate', aggfunc='mean'
            )
            
            if not pivot.empty:
                fig_heat = px.imshow(
                    pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    title='Customer Retention: Cohort × Tenure Period',
                    labels={'x': 'Tenure Period', 'y': 'Join Cohort', 'color': 'Retention Rate'},
                    color_continuous_scale='RdYlGn',
                    aspect='auto'
                )
                fig_heat.update_layout(height=500)
                st.plotly_chart(fig_heat, use_container_width=True)
    
    # ===== DETAILED TABLE =====
    st.subheader("📋 Cohort Details")
    display_cohorts = cohort_ret.sort_values('total_customers', ascending=False).head(20)
    st.dataframe(display_cohorts.style.format({
        'churn_rate': '{:.1%}',
        'retention_rate': '{:.1%}',
        'avg_arpu': '${:.2f}',
        'avg_satisfaction': '{:.1f}'
    }), use_container_width=True)

def create_report_page(df):
    """📄 Executive Report Generation Page."""
    import os
    
    st.header("📄 Executive Report Generator")
    st.markdown("**Generate a professional PDF report with all KPIs, charts, and recommendations**")
    st.markdown("One-click delivery of insights in a format executives actually read.")
    
    report_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'Telecom_Revenue_Optimization_Report.pdf')
    
    st.markdown("---")
    
    # Show what the report contains
    st.subheader("📑 Report Contents")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Page 1** - Cover Page
        - Project title, date, data source
        
        **Page 2** - Executive Summary
        - Key KPIs and metrics
        - Retention engine results
        
        **Page 3** - Customer Analysis
        - Churn distribution charts
        - ARPU analysis
        """)
    with col2:
        st.markdown("""
        **Page 4** - Retention Strategy
        - Health score distribution
        - Top retention strategies
        
        **Page 5** - Next Best Offer
        - Revenue potential by offer
        - Conversion predictions
        
        **Page 6** - Recommendations
        - 7 actionable recommendations
        """)
    
    st.markdown("---")
    
    if os.path.exists(report_path):
        st.success("✅ Report already generated!")
        with open(report_path, 'rb') as f:
            st.download_button(
                label="📥 Download PDF Report",
                data=f.read(),
                file_name="Telecom_Revenue_Optimization_Report.pdf",
                mime="application/pdf"
            )
        st.info(f"Report size: {os.path.getsize(report_path) / 1024:.1f} KB")
    else:
        st.warning("📦 Report not yet generated.")
        if st.button("🚀 Generate Report Now"):
            with st.spinner("Generating PDF report..."):
                try:
                    import sys
                    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
                    from report_generator import ReportGenerator
                    gen = ReportGenerator(os.path.join(os.path.dirname(__file__), '..', 'reports'))
                    output = gen.generate_pdf()
                    st.success(f"✅ Report generated! Refresh the page to download.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

def create_ab_test_dashboard(df):
    """🧪 A/B Test Simulation Dashboard."""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from ab_test_engine import ABTestSimulator
    
    st.header("🧪 A/B Test Simulator")
    st.markdown("**Validate retention strategies with statistical rigor before full deployment**")
    st.markdown("Select an experiment, adjust parameters, and see real-time statistical results.")
    
    simulator = ABTestSimulator(df)
    
    st.markdown("---")
    
    # ===== EXPERIMENT SELECTOR =====
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        exp_options = {f"{v['icon']} {v['name']}": k for k, v in ABTestSimulator.EXPERIMENTS.items()}
        selected_label = st.selectbox("🎯 Select Experiment", list(exp_options.keys()))
        experiment_key = exp_options[selected_label]
    
    with col_config2:
        sample_pct = st.slider("👥 Sample Size (%)", 10, 100, 80, 5,
                              help="Percentage of customers included in the test")
    
    with col_config3:
        treatment_split = st.slider("⚖️ Treatment/Control Split", 30, 70, 50, 5,
                                   help="% assigned to treatment group") / 100
    
    # Show experiment description
    exp_info = ABTestSimulator.EXPERIMENTS[experiment_key]
    st.info(f"**{exp_info['description']}** | Cost: ${exp_info['cost_per_customer']:.2f}/customer | Expected effect: {exp_info['treatment_effect']:.0%} churn reduction")
    
    # ===== RUN EXPERIMENT =====
    result = simulator.run_experiment(experiment_key, sample_pct, treatment_split)
    
    st.markdown("---")
    
    # ===== RESULTS METRICS =====
    st.subheader("📊 Experiment Results")
    
    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
    with rcol1:
        st.metric("Control Churn", f"{result['control_churn']:.1%}")
    with rcol2:
        st.metric("Treatment Churn", f"{result['treatment_churn']:.1%}",
                 delta=f"{result['treatment_churn'] - result['control_churn']:.1%}")
    with rcol3:
        sig_label = "✅ Yes" if result['is_significant'] else "❌ No"
        st.metric("Significant?", sig_label)
    with rcol4:
        st.metric("P-Value", f"{result['p_value_z']:.4f}")
    with rcol5:
        st.metric("Sample Size", f"{result['sample_size']:,}")
    
    # ===== RECOMMENDATION =====
    rec = result['recommendation']
    if rec['color'] == 'success':
        st.success(f"**{rec['decision']}**: {rec['reason']}")
    elif rec['color'] == 'error':
        st.error(f"**{rec['decision']}**: {rec['reason']}")
    else:
        st.warning(f"**{rec['decision']}**: {rec['reason']}")
    
    st.markdown("---")
    
    # ===== CHARTS =====
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Churn rate comparison
        fig_compare = go.Figure(data=[
            go.Bar(name='Control', x=['Churn Rate'], y=[result['control_churn']],
                   marker_color='#ef5350', text=[f"{result['control_churn']:.1%}"], textposition='auto'),
            go.Bar(name='Treatment', x=['Churn Rate'], y=[result['treatment_churn']],
                   marker_color='#4caf50', text=[f"{result['treatment_churn']:.1%}"], textposition='auto')
        ])
        fig_compare.update_layout(
            title='Churn Rate: Control vs Treatment',
            yaxis_tickformat='.1%',
            barmode='group', height=350
        )
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption(f"💡 Treatment group shows {abs(result['churn_reduction'])*100:.1f}pp {'lower' if result['churn_reduction'] > 0 else 'higher'} churn. 95% CI: [{result['ci_lower']*100:.1f}pp, {result['ci_upper']*100:.1f}pp]")
    
    with col_right:
        # Power curve
        power_data = simulator.generate_power_curve()
        fig_power = px.line(power_data, x='sample_size', y='power',
                           color='effect_size', title='Statistical Power Analysis',
                           labels={'sample_size': 'Sample Size (per group)',
                                  'power': 'Statistical Power', 'effect_size': 'Effect Size'})
        fig_power.add_hline(y=0.8, line_dash='dash', line_color='red',
                           annotation_text='80% Power (Standard)')
        fig_power.update_layout(height=350)
        st.plotly_chart(fig_power, use_container_width=True)
        st.caption(f"💡 To detect a {exp_info['treatment_effect']:.0%} effect with 80% power, you need ~{simulator.calculate_required_sample_size(result['control_churn'], exp_info['treatment_effect']):,} customers per group.")
    
    # ===== STATISTICAL DETAILS =====
    st.subheader("📝 Statistical Details")
    
    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown(f"""
        | Test | Value |
        |------|-------|
        | Chi-squared statistic | {result['chi2_statistic']:.4f} |
        | Z-statistic | {result['z_statistic']:.4f} |
        | P-value (two-sided) | {result['p_value_z']:.6f} |
        | Significance (\u03b1=0.05) | {'Yes ✅' if result['is_significant'] else 'No ❌'} |
        | Effect size (Cohen\'s h) | {result['effect_size_h']:.4f} |
        | Min Detectable Effect | {result['mde']:.1%} |
        """)
    
    with stat_col2:
        st.markdown(f"""
        | Financial Impact | Value |
        |------|-------|
        | Avg ARPU | ${result['avg_arpu']:.2f} |
        | Monthly Revenue Saved | ${result['monthly_revenue_saved']:,.0f} |
        | Annual Revenue Saved | ${result['annual_revenue_saved']:,.0f} |
        | Total Intervention Cost | ${result['total_cost']:,.0f} |
        | ROI | {result['roi']:.1f}x |
        | Customers in Treatment | {result['n_treatment']:,} |
        """)

def create_anomaly_dashboard(df):
    """🚨 Anomaly Detection Dashboard."""
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
    from anomaly_detector import AnomalyDetector
    
    st.header("🚨 Anomaly Detection & Alerts")
    st.markdown("**Proactive monitoring: Automatically detect unusual patterns before they become problems**")
    
    detector = AnomalyDetector(df)
    summary = detector.get_alert_summary()
    
    # ===== ALERT SEVERITY SUMMARY =====
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🚨 Total Alerts", summary['total_alerts'])
    with col2:
        st.metric("🔴 Critical", summary['critical'])
    with col3:
        st.metric("🟡 Warning", summary['warnings'])
    with col4:
        st.metric("🟢 Info", summary['info'])
    
    st.markdown("---")
    
    # ===== METRIC ALERTS =====
    st.subheader("📊 Metric-Level Alerts")
    
    for alert in sorted(summary['metric_alerts'], key=lambda x: x['outlier_pct'], reverse=True):
        severity_colors = {'critical': '🔴', 'warning': '🟡', 'info': '🟢'}
        sev_icon = severity_colors.get(alert['severity'], '⚪')
        
        with st.expander(f"{sev_icon} {alert['icon']} {alert['metric']} — {alert['n_outliers_zscore']:,} outliers ({alert['outlier_pct']:.1f}%) [{alert['severity'].upper()}]", expanded=alert['severity'] == 'critical'):
            ecol1, ecol2, ecol3, ecol4 = st.columns(4)
            with ecol1:
                st.metric("Mean", f"{alert['mean']:.2f}{alert['unit']}")
            with ecol2:
                st.metric("Median", f"{alert['median']:.2f}{alert['unit']}")
            with ecol3:
                st.metric("Std Dev", f"{alert['std']:.2f}")
            with ecol4:
                st.metric("Skewness", f"{alert['skewness']:.2f}")
            
            st.caption(f"💡 Normal range: [{alert['lower_bound']:.1f}, {alert['upper_bound']:.1f}]. Values outside this via IQR: {alert['n_outliers_iqr']:,} customers.")
    
    st.markdown("---")
    
    # ===== OUTLIER DISTRIBUTION CHART =====
    st.subheader("📊 Outlier Distribution by Metric")
    
    alert_df = pd.DataFrame(summary['metric_alerts'])
    if len(alert_df) > 0:
        fig_outliers = px.bar(
            alert_df.sort_values('outlier_pct', ascending=True),
            x='outlier_pct', y='metric',
            orientation='h',
            title='Percentage of Outlier Customers per Metric',
            labels={'outlier_pct': 'Outlier %', 'metric': 'Metric'},
            color='severity',
            color_discrete_map={'critical': '#ef5350', 'warning': '#ff9800', 'info': '#4caf50'}
        )
        fig_outliers.update_layout(height=350)
        st.plotly_chart(fig_outliers, use_container_width=True)
        most_problematic = alert_df.loc[alert_df['outlier_pct'].idxmax()]
        st.caption(f"💡 {most_problematic['metric']} has the highest outlier rate at {most_problematic['outlier_pct']:.1f}% — investigate these customers for data quality or genuine anomalies.")
    
    # ===== SEGMENT ANOMALIES =====
    if summary['segment_alerts']:
        st.markdown("---")
        st.subheader("🎯 Segment-Level Anomalies")
        st.markdown("*Cities or plan types that deviate significantly from the overall average*")
        
        seg_df = pd.DataFrame(summary['segment_alerts'])
        seg_df['Deviation'] = seg_df.apply(
            lambda r: f"{r['deviation_pct']:+.1f}% {'above' if r['direction'] == 'above' else 'below'} avg", axis=1
        )
        
        display_cols = ['segment_type', 'segment_value', 'metric', 'value', 'benchmark', 'Deviation', 'n_customers', 'severity']
        st.dataframe(seg_df[display_cols].style.format({
            'value': '{:.2f}',
            'benchmark': '{:.2f}'
        }), use_container_width=True)
    
    # ===== TOP ANOMALOUS CUSTOMERS =====
    st.markdown("---")
    st.subheader("🔥 Top 20 Most Anomalous Customers")
    st.markdown("*Customers with the highest composite anomaly score across all metrics*")
    
    top_anomalous = detector.get_customer_anomalies(20)
    st.dataframe(top_anomalous.style.format({
        'anomaly_score': '{:.2f}',
        'arpu': '${:.2f}',
        'satisfaction_score': '{:.1f}'
    }), use_container_width=True)
    st.caption(f"💡 These customers show unusual patterns across multiple metrics — they could be fraud risks, data errors, or high-priority retention targets.")

def main():
    """Main dashboard application."""
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0;">📊 Telco Customer Intelligence</h1>
        <p style="color: #718096; font-size: 1.05rem; font-weight: 500; margin-top: -5px;">
            Powered by Real Kaggle Data  •  ML-Driven Insights  •  Actionable Strategies
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("# 🚀 Navigation")
    st.sidebar.markdown("---")
    page = st.sidebar.radio("", [
        "📊 Overview",
        "💰 Revenue Analysis", 
        "🔄 Churn Analysis",
        "🎯 Smart Retention Engine",
        "💰 Revenue Simulator",
        "🔮 Next Best Offer",
        "🧠 Model Explainability",
        "📅 Cohort Analysis",
        "🧪 A/B Test Simulator",
        "🚨 Anomaly Detection",
        "📄 Executive Report",
        "📢 Campaign Effectiveness",
        "👥 Customer Insights"
    ])
    
    # Overview page
    if page == "📊 Overview":
        st.header("Executive Overview")
        
        # Key metrics
        metrics = calculate_business_metrics(df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Customers", f"{metrics['total_customers']:,}")
        
        with col2:
            st.metric("Monthly Revenue", f"${metrics['total_revenue']:,.2f}")
        
        with col3:
            st.metric("Churn Rate", f"{metrics['churn_rate']:.2%}")
        
        with col4:
            st.metric("Avg Satisfaction", f"{metrics['avg_satisfaction']:.1f}/10")
        
        # Quick insights
        st.subheader("Key Business Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success(f"💰 **Revenue Opportunity**: ${metrics['revenue_at_risk']:,.2f} monthly revenue at risk from churn")
            st.info(f"🎯 **High-Value Segment**: {metrics['high_value_customers']:,} customers with ARPU > $75")
        
        with col2:
            st.warning(f"📢 **Campaign Potential**: {metrics['campaign_responsive']:,} customers responsive to campaigns")
            responsive_rate = metrics['campaign_responsive'] / metrics['total_customers']
            st.info(f"📈 **Engagement Rate**: {responsive_rate:.1%} of customers engage with campaigns")
    
    elif page == "💰 Revenue Analysis":
        create_revenue_dashboard(df)
    
    elif page == "🔄 Churn Analysis":
        create_churn_dashboard(df)
    
    elif page == "🎯 Smart Retention Engine":
        create_retention_engine_dashboard(df)
    
    elif page == "💰 Revenue Simulator":
        create_revenue_simulator(df)
    
    elif page == "🔮 Next Best Offer":
        create_nbo_dashboard(df)
    
    elif page == "🧠 Model Explainability":
        create_explainability_dashboard(df)
    
    elif page == "📅 Cohort Analysis":
        create_cohort_dashboard(df)
    
    elif page == "🧪 A/B Test Simulator":
        create_ab_test_dashboard(df)
    
    elif page == "🚨 Anomaly Detection":
        create_anomaly_dashboard(df)
    
    elif page == "📄 Executive Report":
        create_report_page(df)
    
    elif page == "📢 Campaign Effectiveness":
        create_campaign_dashboard(df)
    
    elif page == "👥 Customer Insights":
        create_customer_insights(df)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="padding: 10px 0; font-size: 0.8rem; opacity: 0.8;">
        <p style="margin: 4px 0;">📡 <strong>Data</strong>: IBM Telco Churn (Kaggle)</p>
        <p style="margin: 4px 0;">🤖 <strong>ML</strong>: GBM, SHAP, NBO</p>
        <p style="margin: 4px 0;">👥 <strong>Customers</strong>: 7,043 real records</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
