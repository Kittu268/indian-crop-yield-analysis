import pandas as pd
import pickle
import plotly.express as px
from dash import Dash, dcc, html

# Load model and pipeline
with open("D:/crop-yield-project/indian-crop-yield-analysis/model_files/preprocessing_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Load test data
X_test = pd.read_csv("D:/crop-yield-project/indian-crop-yield-analysis/model_files/X_test.csv")
y_test = pd.read_csv("D:/crop-yield-project/indian-crop-yield-analysis/model_files/y_test.csv")

# Predict
predictions = pipeline.predict(X_test)

# Create DataFrame for visualization
pred_df = pd.DataFrame({
    'Actual': y_test.values.flatten(),
    'Predicted': predictions
})

# Initialize Dash app
app = Dash(__name__)

# Create scatter plot
fig = px.scatter(pred_df, x='Actual', y='Predicted', trendline='ols',
                 title="Actual vs Predicted Rice Yield",
                 labels={'Actual': 'Actual Yield', 'Predicted': 'Predicted Yield'})


# Layout
app.layout = html.Div([
    html.H1("Crop Yield Prediction Dashboard"),
    dcc.Graph(figure=fig)
])

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

