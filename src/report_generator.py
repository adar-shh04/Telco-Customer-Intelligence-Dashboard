"""
Automated Executive PDF Report Generator

One-click generation of a professional PDF report with:
- Executive summary with KPIs
- Customer health analysis
- Churn predictions and retention strategies
- NBO recommendations
- Revenue impact projections

Shows you understand the "last mile" of data science:
delivering results in a format executives actually read.
"""

import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime


def _create_chart(fig_func, filepath, **kwargs):
    """Helper: create and save a chart image."""
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 4)))
    fig_func(ax, **kwargs)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return filepath


class ReportGenerator:
    """Generates professional executive PDF reports."""
    
    def __init__(self, output_dir='reports'):
        self.output_dir = output_dir
        self.chart_dir = os.path.join(output_dir, 'charts')
        os.makedirs(self.chart_dir, exist_ok=True)
        self.charts = {}
    
    def _load_data(self):
        """Load all available data sources."""
        data = {}
        
        # Master dataset
        master_path = 'data/raw/master_dataset.csv'
        if os.path.exists(master_path):
            data['master'] = pd.read_csv(master_path)
        
        # Retention analysis
        ret_path = 'data/processed/retention_analysis.csv'
        if os.path.exists(ret_path):
            data['retention'] = pd.read_csv(ret_path)
        
        # NBO recommendations
        nbo_path = 'data/processed/nbo_recommendations.csv'
        if os.path.exists(nbo_path):
            data['nbo'] = pd.read_csv(nbo_path)
        
        # SHAP importance
        shap_path = 'data/processed/shap_importance.csv'
        if os.path.exists(shap_path):
            data['shap'] = pd.read_csv(shap_path, index_col=0, header=None)
        
        return data
    
    def _generate_charts(self, data):
        """Generate all chart images for the report."""
        master = data.get('master')
        if master is None:
            return
        
        # 1. Churn Rate Pie
        def churn_pie(ax, **kw):
            churned = master['churned'].value_counts()
            labels = ['Retained', 'Churned']
            colors = ['#4caf50', '#ef5350']
            ax.pie(churned.values, labels=labels, autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12})
            ax.set_title('Customer Churn Distribution', fontsize=14, fontweight='bold')
        
        self.charts['churn_pie'] = _create_chart(churn_pie, 
            os.path.join(self.chart_dir, 'churn_pie.png'))
        
        # 2. ARPU Distribution
        def arpu_dist(ax, **kw):
            ax.hist(master['arpu'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
            ax.axvline(master['arpu'].mean(), color='red', linestyle='--',
                      label=f"Mean: ${master['arpu'].mean():.2f}")
            ax.set_title('ARPU Distribution', fontsize=14, fontweight='bold')
            ax.set_xlabel('Monthly ARPU ($)')
            ax.set_ylabel('Number of Customers')
            ax.legend()
        
        self.charts['arpu_dist'] = _create_chart(arpu_dist,
            os.path.join(self.chart_dir, 'arpu_dist.png'))
        
        # 3. Health Score Distribution (if available)
        retention = data.get('retention')
        if retention is not None and 'customer_health_score' in retention.columns:
            def health_dist(ax, **kw):
                ax.hist(retention['customer_health_score'], bins=40, 
                       color='#2196f3', edgecolor='black', alpha=0.7)
                ax.axvline(retention['customer_health_score'].mean(), color='red',
                          linestyle='--', label=f"Mean: {retention['customer_health_score'].mean():.1f}")
                ax.set_title('Customer Health Score Distribution', fontsize=14, fontweight='bold')
                ax.set_xlabel('Health Score (0-100)')
                ax.set_ylabel('Customers')
                ax.legend()
            
            self.charts['health_dist'] = _create_chart(health_dist,
                os.path.join(self.chart_dir, 'health_dist.png'))
        
        # 4. Churn by Plan Type
        def churn_by_plan(ax, **kw):
            churn_plan = master.groupby('plan_type')['churned'].mean() * 100
            colors = ['#ef5350', '#ff9800', '#4caf50']
            bars = ax.bar(churn_plan.index, churn_plan.values, color=colors, edgecolor='black')
            ax.set_title('Churn Rate by Plan Type', fontsize=14, fontweight='bold')
            ax.set_ylabel('Churn Rate (%)')
            for bar, val in zip(bars, churn_plan.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}%', ha='center', fontsize=11)
        
        self.charts['churn_by_plan'] = _create_chart(churn_by_plan,
            os.path.join(self.chart_dir, 'churn_by_plan.png'))
        
        # 5. Strategy Distribution (if available)
        if retention is not None and 'strategy_name' in retention.columns:
            def strategy_dist(ax, **kw):
                strat = retention['strategy_name'].value_counts().head(5)
                ax.barh(range(len(strat)), strat.values, color=plt.cm.Set2(np.linspace(0, 1, len(strat))))
                ax.set_yticks(range(len(strat)))
                ax.set_yticklabels([s[:35] for s in strat.index], fontsize=9)
                ax.set_title('Top Retention Strategies', fontsize=14, fontweight='bold')
                ax.set_xlabel('Number of Customers')
            
            self.charts['strategy_dist'] = _create_chart(strategy_dist,
                os.path.join(self.chart_dir, 'strategy_dist.png'),
                figsize=(8, 5))
        
        print(f"  Generated {len(self.charts)} charts")
    
    def generate_pdf(self, filename='Telecom_Revenue_Optimization_Report.pdf'):
        """Generate the complete executive PDF report."""
        print("=" * 60)
        print("EXECUTIVE REPORT GENERATOR")
        print("=" * 60)
        
        # Load data
        print("\n📥 Loading data...")
        data = self._load_data()
        master = data.get('master')
        if master is None:
            print("ERROR: Master dataset not found!")
            return None
        
        # Generate charts
        print("\n📊 Generating charts...")
        self._generate_charts(data)
        
        # Build PDF
        print("\n📄 Building PDF report...")
        try:
            from fpdf import FPDF
        except ImportError:
            print("  fpdf2 not installed. Installing...")
            os.system('pip install fpdf2 --quiet')
            from fpdf import FPDF
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # ===== COVER PAGE =====
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 28)
        pdf.ln(50)
        pdf.cell(0, 15, 'Telecom Revenue', new_x='LMARGIN', new_y='NEXT', align='C')
        pdf.cell(0, 15, 'Optimization Report', new_x='LMARGIN', new_y='NEXT', align='C')
        pdf.ln(10)
        pdf.set_font('Helvetica', '', 14)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%B %d, %Y")}', 
                new_x='LMARGIN', new_y='NEXT', align='C')
        pdf.cell(0, 10, f'Data Source: IBM Telco Customer Churn (Kaggle)', 
                new_x='LMARGIN', new_y='NEXT', align='C')
        pdf.cell(0, 10, f'Customers Analyzed: {len(master):,}', 
                new_x='LMARGIN', new_y='NEXT', align='C')
        pdf.ln(20)
        pdf.set_font('Helvetica', 'I', 10)
        pdf.cell(0, 8, 'Powered by Smart Retention Engine, NBO Predictions, and ML Explainability',
                new_x='LMARGIN', new_y='NEXT', align='C')
        
        # ===== EXECUTIVE SUMMARY =====
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 12, '1. Executive Summary', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(5)
        
        pdf.set_font('Helvetica', '', 11)
        
        churn_rate = master['churned'].mean()
        avg_arpu = master['arpu'].mean()
        total_revenue = master['arpu'].sum()
        revenue_at_risk = master[master['churned'] == 1]['arpu'].sum()
        
        kpis = [
            f"Total Customers: {len(master):,}",
            f"Monthly Revenue: ${total_revenue:,.2f}",
            f"Average ARPU: ${avg_arpu:.2f}",
            f"Current Churn Rate: {churn_rate:.1%}",
            f"Revenue at Risk (from churned): ${revenue_at_risk:,.2f}/month",
            f"Average Satisfaction: {master['satisfaction_score'].mean():.1f}/10",
        ]
        
        for kpi in kpis:
            pdf.cell(0, 7, f"  * {kpi}", new_x='LMARGIN', new_y='NEXT')
        
        # Retention engine stats
        retention = data.get('retention')
        if retention is not None:
            pdf.ln(5)
            pdf.set_font('Helvetica', 'B', 13)
            pdf.cell(0, 10, 'Retention Engine Results:', new_x='LMARGIN', new_y='NEXT')
            pdf.set_font('Helvetica', '', 11)
            
            at_risk = retention[retention.get('strategy_name', pd.Series()) != 'No Intervention Needed'] if 'strategy_name' in retention.columns else retention
            
            ret_stats = [
                f"At-risk customers: {len(at_risk):,} ({len(at_risk)/len(retention):.1%})",
                f"Average Health Score: {retention['customer_health_score'].mean():.1f}/100" if 'customer_health_score' in retention.columns else "",
                f"Total intervention cost: ${at_risk['intervention_cost'].sum():,.0f}" if 'intervention_cost' in at_risk.columns else "",
                f"Expected revenue saved: ${at_risk['expected_revenue_saved'].sum():,.0f}" if 'expected_revenue_saved' in at_risk.columns else "",
            ]
            for stat in ret_stats:
                if stat:
                    pdf.cell(0, 7, f"  * {stat}", new_x='LMARGIN', new_y='NEXT')
        
        # ===== CHARTS PAGES =====
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 12, '2. Customer Analysis', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(5)
        
        if 'churn_pie' in self.charts:
            pdf.image(self.charts['churn_pie'], x=15, w=85)
        if 'churn_by_plan' in self.charts:
            pdf.image(self.charts['churn_by_plan'], x=105, w=85)
        
        pdf.ln(5)
        if 'arpu_dist' in self.charts:
            pdf.image(self.charts['arpu_dist'], x=15, w=180)
        
        # ===== RETENTION STRATEGIES =====
        if retention is not None:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 20)
            pdf.cell(0, 12, '3. Retention Strategy Analysis', new_x='LMARGIN', new_y='NEXT')
            pdf.ln(5)
            
            if 'health_dist' in self.charts:
                pdf.image(self.charts['health_dist'], x=15, w=180)
            
            pdf.ln(5)
            if 'strategy_dist' in self.charts:
                pdf.image(self.charts['strategy_dist'], x=15, w=180)
        
        # ===== NBO SECTION =====
        nbo = data.get('nbo')
        if nbo is not None:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 20)
            pdf.cell(0, 12, '4. Next Best Offer Analysis', new_x='LMARGIN', new_y='NEXT')
            pdf.ln(5)
            pdf.set_font('Helvetica', '', 11)
            
            total_rev = nbo['expected_annual_value'].sum()
            avg_conv = nbo['conversion_prob'].mean()
            
            pdf.cell(0, 7, f"  * Total expected annual revenue from NBO: ${total_rev:,.0f}", 
                    new_x='LMARGIN', new_y='NEXT')
            pdf.cell(0, 7, f"  * Average conversion probability: {avg_conv:.1%}", 
                    new_x='LMARGIN', new_y='NEXT')
            pdf.cell(0, 7, f"  * Customers with offers: {nbo['customer_id'].nunique():,}", 
                    new_x='LMARGIN', new_y='NEXT')
            
            pdf.ln(5)
            pdf.set_font('Helvetica', 'B', 13)
            pdf.cell(0, 10, 'Top Offers by Revenue Potential:', new_x='LMARGIN', new_y='NEXT')
            pdf.set_font('Helvetica', '', 10)
            
            offer_rev = nbo.groupby('offer_name')['expected_annual_value'].sum().sort_values(ascending=False)
            for offer, rev in offer_rev.items():
                pdf.cell(0, 6, f"    {offer}: ${rev:,.0f}", new_x='LMARGIN', new_y='NEXT')
        
        # ===== RECOMMENDATIONS =====
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 20)
        pdf.cell(0, 12, '5. Recommendations', new_x='LMARGIN', new_y='NEXT')
        pdf.ln(5)
        pdf.set_font('Helvetica', '', 11)
        
        recommendations = [
            "1. Deploy the Smart Retention Engine to proactively identify and retain at-risk customers.",
            "2. Implement the Next Best Offer system to drive upsell/cross-sell revenue growth.",
            "3. Focus retention budget on high-value customers with the highest expected ROI.",
            "4. Address the primary churn drivers (low engagement, competitor risk, price sensitivity).",
            "5. Monitor the Customer Health Score dashboard weekly for early warning signals.",
            "6. Run A/B tests on retention strategies to optimize intervention effectiveness.",
            "7. Invest in plan upgrades for Basic-plan customers, who show the highest churn risk.",
        ]
        
        for rec in recommendations:
            pdf.multi_cell(0, 7, f"  {rec}")
            pdf.ln(2)
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        pdf.output(output_path)
        
        print(f"\n✅ Report saved to: {output_path}")
        print(f"   Pages: {pdf.pages_count}")
        return output_path


def generate_report():
    """Generate the executive report."""
    generator = ReportGenerator()
    return generator.generate_pdf()


if __name__ == "__main__":
    generate_report()
