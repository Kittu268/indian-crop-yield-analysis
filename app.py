import os
import pandas as pd
import pickle
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Step 1: Load test data
X_test = pd.read_csv("model_files/X_test.csv")
y_test = pd.read_csv("model_files/y_test.csv")

# Step 2: Predict using numeric-only data
X_numeric = X_test.copy()
with open("model_files/rice_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

predictions = model.predict(X_numeric)

# Step 3: Inject 'Crop' and 'State' AFTER prediction
X_test["Crop"] = "Rice"

state_cols = [c for c in X_test.columns if c.startswith("State Name_")]
X_test["State"] = (
    X_test[state_cols]
      .idxmax(axis=1)
      .str.replace("State Name_", "")
)

# Step 4: Build final DataFrame
data = X_test.copy()
data["Actual"] = y_test.values.flatten()
data["Predicted"] = predictions

# Step 5: Initialize Dash app
app = Dash(__name__)

# Step 6: Layout with dropdown and graph
app.layout = html.Div([
    html.H1("Crop Yield Prediction Dashboard", style={'textAlign': 'center'}),

    dcc.Dropdown(
        id="state-dropdown",
        options=[{"label": s, "value": s} for s in sorted(data["State"].unique())],
        value=sorted(data["State"].unique())[0],
        clearable=False,
        style={"width": "50%", "margin": "0 auto", "margin-bottom": "20px"}
    ),

    dcc.Graph(id="yield-graph")
])

# Step 7: Callback to update graph
@app.callback(
    Output("yield-graph", "figure"),
    Input("state-dropdown", "value")
)
def update_graph(selected_state):
    df = data[data["State"] == selected_state]
    fig = px.scatter(
        df,
        x="Actual",
        y="Predicted",
        trendline="ols",
        title=f"Rice Yield: Actual vs Predicted in {selected_state}",
        labels={"Actual": "Actual Yield", "Predicted": "Predicted Yield"}
    )
    return fig

# Step 8: Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
