# Updated to use rice_yield_model.pkl instead of preprocessing_pipeline.pkl

import pandas as pd
import pickle
import plotly.express as px
from dash import Dash, dcc, html

# Load trained model
with open("model_files/rice_yield_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load test data
X_test = pd.read_csv("model_files/X_test.csv")
y_test = pd.read_csv("model_files/y_test.csv")

# Make predictions
predictions = model.predict(X_test)

# Create DataFrame for visualization
pred_df = pd.DataFrame({
    'Actual': y_test.values.flatten(),
    'Predicted': predictions
})

# Initialize Dash app
app = Dash(__name__)

# Create scatter plot
fig = px.scatter(
    pred_df,
    x='Actual',
    y='Predicted',
    trendline='ols',
    title="Actual vs Predicted Rice Yield",
    labels={'Actual': 'Actual Yield', 'Predicted': 'Predicted Yield'}
)

# Define layout
app.layout = html.Div([
    html.H1("Crop Yield Prediction Dashboard", style={'textAlign': 'center'}),
    dcc.Graph(figure=fig)
])

# Run the app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host="0.0.0.0", port=port)
