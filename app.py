import os
import pandas as pd
import pickle
import plotly.express as px
from dash import Dash, dcc, html, Input, Output

# Load trained model
with open("model_files/rice_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test data and actual yields
X_test = pd.read_csv("model_files/X_test.csv")
y_test = pd.read_csv("model_files/y_test.csv")

# Merge into one DataFrame
data = X_test.copy()
data["Actual"] = y_test.values.flatten()
data["Predicted"] = model.predict(X_test)

# Initialize Dash app
app = Dash(__name__)

# App layout with a dropdown and graph
app.layout = html.Div([
    html.H1("Crop Yield Prediction Dashboard", style={"textAlign": "center"}),
    
    # Dropdown for crop selection
    dcc.Dropdown(
        id="crop-dropdown",
        options=[
            {"label": crop, "value": crop}
            for crop in sorted(data["Crop"].unique())
        ],
        value=sorted(data["Crop"].unique())[0],
        clearable=False,
        style={"width": "50%", "margin": "0 auto"}
    ),
    
    # Graph that will update based on dropdown
    dcc.Graph(id="yield-graph")
])

# Callback: when dropdown changes, update the scatter plot
@app.callback(
    Output("yield-graph", "figure"),
    Input("crop-dropdown", "value")
)
def update_graph(selected_crop):
    # Filter data by the chosen crop
    df = data[data["Crop"] == selected_crop]
    
    # Recreate scatter with trendline
    fig = px.scatter(
        df,
        x="Actual",
        y="Predicted",
        trendline="ols",
        title=f"Actual vs Predicted Yield for {selected_crop}",
        labels={"Actual": "Actual Yield", "Predicted": "Predicted Yield"}
    )
    return fig

# Bind to the Render‐assigned port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
