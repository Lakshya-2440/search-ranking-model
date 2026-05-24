import gradio as gr
import os

# Create a beautifully styled Gradio dashboard to showcase the Search Ranking Model

def get_plot(plot_name):
    return os.path.join("plots", plot_name)

css = """
h1 {
    text-align: center;
    color: #2ECC71;
}
h2, h3 {
    color: #3498DB;
}
.stat-box {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin: 10px;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stat-number {
    font-size: 2em;
    font-weight: bold;
    color: black;
}
.stat-label {
    font-size: 1.1em;
    color: #black;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Soft(primary_hue="green", secondary_hue="blue")) as demo:
    gr.Markdown("# LambdaMART Search Ranking Model")
    gr.Markdown("### A Production-Grade Learning-to-Rank System")
    
    with gr.Row():
        gr.HTML('''
        <div class="stat-box">
            <div class="stat-number">2M+</div>
            <div class="stat-label">Query-Doc Pairs Trained</div>
        </div>
        ''')
        gr.HTML('''
        <div class="stat-box">
            <div class="stat-number">+56.5%</div>
            <div class="stat-label">NDCG@10 Improvement</div>
        </div>
        ''')
        gr.HTML('''
        <div class="stat-box">
            <div class="stat-number">18</div>
            <div class="stat-label">Engineered Features</div>
        </div>
        ''')
        gr.HTML('''
        <div class="stat-box">
            <div class="stat-number">p &lt; 0.0001</div>
            <div class="stat-label">A/B Test Significance</div>
        </div>
        ''')

    with gr.Tabs():
        with gr.TabItem("Offline Evaluation"):
            gr.Markdown("## NDCG & Metrics Comparison")
            gr.Markdown("LambdaMART significantly outperforms the BM25 keyword-matching baseline across all metrics by incorporating user interaction signals and structural document features.")
            
            with gr.Row():
                gr.Image(value=get_plot("ndcg_comparison.png"), label="NDCG Cutoffs")
                gr.Image(value=get_plot("all_metrics_comparison.png"), label="All Metrics")
                
            with gr.Row():
                gr.Image(value=get_plot("per_query_ndcg.png"), label="Per-Query Distribution")

        with gr.TabItem("Feature Importance"):
            gr.Markdown("## What Drives Relevance?")
            gr.Markdown("Interaction features (dwell time, CTR, skip rate) dominate the model's decision making, proving that user signals are much stronger indicators of relevance than keyword overlap alone.")
            
            with gr.Row():
                gr.Image(value=get_plot("feature_importance.png"), label="XGBoost Feature Gain")
                gr.Image(value=get_plot("position_bias.png"), label="Position Bias Analysis")

        with gr.TabItem("A/B Testing & Deployment"):
            gr.Markdown("## Shadow Deployment & A/B Test Simulation")
            gr.Markdown("The pipeline includes an end-to-end simulation of production deployment. The shadow deployment runs the model side-by-side with BM25, and the A/B test simulates a 5% traffic split over 14 days.")
            
            with gr.Row():
                gr.Image(value=get_plot("ab_test_results.png"), label="A/B Test Results")
                
        with gr.TabItem("Highlights"):
            gr.Markdown("## Key Project Achievements")
            gr.Markdown("""
            - 🏆 **Built a learning-to-rank model** using LambdaMART (XGBRanker) on 2M+ synthetic query-doc pairs.
            - 📈 **Improved NDCG@10 by 56.5%** over the BM25 baseline in offline evaluation.
            - 🏗️ **Engineered 18 robust features** across query, document, and interaction groups, including Inverse Propensity Weighting (IPW) for position bias correction.
            - 🔬 **Simulated A/B testing** demonstrating statistically significant improvements (p < 0.0001) in CTR and dwell time.
            - 🛠️ **Tech Stack**: Python, XGBoost, Pandas, Scikit-Learn, Gradio.
            """)


if __name__ == "__main__":
    demo.launch()
