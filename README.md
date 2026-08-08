# Timber Value Bid Estimation System

A production-ready Flask web application serving an XGBoost machine learning model to estimate timber values based on wood characteristics.

## Project Structure
- `app.py`: Main Flask application.
- `models/`: Contains the serialized XGBoost model and preprocessor (e.g., encoders, scalers).
- `templates/`: HTML views for user interface.
- `static/`: CSS, JS, and image assets.
- `uploads/`: For file uploads (e.g., CSV imports for batch estimations).
- `logs/`: Application performance and debug logs.
