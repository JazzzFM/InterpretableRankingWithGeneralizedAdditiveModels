import os, mlflow.pyfunc, pandas as pd
import dash
from dash import dcc, html, Input, Output

MODEL_URI = os.getenv("MODEL_URI", "models:/credit_gam/Production")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
if MLFLOW_TRACKING_URI:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.pyfunc.load_model(MODEL_URI)

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H3("Credit Ranking — GAM"),
    html.Div([
        "Age: ", dcc.Input(id="age", type="number", value=35),
        " CreditAmount: ", dcc.Input(id="amount", type="number", value=2500),
        " Duration: ", dcc.Input(id="duration", type="number", value=12),
        html.Button("Score", id="btn")
    ]),
    html.Pre(id="out")
])

@app.callback(Output("out","children"),
              Input("btn","n_clicks"),
              [Input("age","value"), Input("amount","value"), Input("duration","value")])
def score(_, age, amount, duration):
    df = pd.DataFrame([{"Age":age,"CreditAmount":amount,"Duration":duration}])
    p = float(model.predict(df)[0])
    return f"Prob(default): {p:.3f} → {'approve' if p<0.25 else 'review'}"

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050)
