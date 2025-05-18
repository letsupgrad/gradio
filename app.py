import gradio as gr
import pandas as pd
import numpy as np
import os
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Calculator Logic ---
def calculate_metrics(impressions, clicks, conversions, cost):
    ctr = round(clicks / impressions, 4)
    cvr = round(conversions / clicks, 4) if clicks > 0 else 0
    ecpm = round((cost / impressions) * 1000, 2)
    cpa = round(cost / conversions, 2) if conversions > 0 else 0
    return ctr, cvr, ecpm, cpa

# --- 2. Forecast Logic ---
def forecast_ecpm():
    np.random.seed(42)
    history = np.random.uniform(5, 25, 30)
    ts = pd.Series(history)
    model = ARIMA(ts, order=(1, 1, 1))
    fit = model.fit()
    forecast = fit.forecast(steps=7)
    return pd.DataFrame({'Day': range(1, 8), 'Forecasted eCPM': forecast.round(2)})

# --- 3. Stakeholder Insights ---
def stakeholder_insights():
    return """\
**Stakeholder Value from SSP AI Optimizer**

1. Marketing Teams: Auto-prioritize screens based on CTR, CVR, eCPM.
2. Media Buyers: Negotiate rates & forecast ROIs.
3. Sales: Show clients real-time data-backed results.
4. Analysts: Compare AI vs manual bidding efficiency.
5. Ops Teams: Monitor AI bidding and scaling in production.
"""

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# SSP AI Yield & Pricing Optimizer")

    with gr.Tab("Calculator"):
        gr.Markdown("### Calculate eCPM, CTR, CVR, CPA")
        with gr.Row():
            impressions = gr.Number(label="Impressions", value=10000)
            clicks = gr.Number(label="Clicks", value=500)
            conversions = gr.Number(label="Conversions", value=50)
            cost = gr.Number(label="Cost (MYR)", value=800)

        with gr.Row():
            btn = gr.Button("Calculate")
            ctr = gr.Textbox(label="CTR")
            cvr = gr.Textbox(label="CVR")
            ecpm = gr.Textbox(label="eCPM")
            cpa = gr.Textbox(label="CPA")

        btn.click(calculate_metrics, [impressions, clicks, conversions, cost], [ctr, cvr, ecpm, cpa])

    with gr.Tab("Forecast"):
        gr.Markdown("### 7-Day eCPM Forecast (ARIMA Model)")
        forecast_btn = gr.Button("Generate Forecast")
        forecast_table = gr.Dataframe(headers=["Day", "Forecasted eCPM"])
        forecast_btn.click(forecast_ecpm, outputs=forecast_table)

    with gr.Tab("AI Benefits"):
        gr.Markdown("### How AI Boosts Ad Tech Teams")
        insight_box = gr.Textbox(label="", lines=15)
        gr.Button("Show Benefits").click(stakeholder_insights, outputs=insight_box)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)
