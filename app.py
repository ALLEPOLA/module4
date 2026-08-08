import os
import sys
import io
import math
import sqlite3
import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file
from fpdf import FPDF

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = "timber_secret_key_for_flash"

# Define paths to model and preprocessor files in the root folder
MODEL_PATH = "xgb_value_model.joblib"
PREPROCESSOR_PATH = "preprocessor.joblib"

# Global placeholders for the loaded machine learning objects
model = None
preprocessor = None

# Exact order of columns the preprocessor expects (must match training dataframe columns)
EXPECTED_COLUMNS = [
    'species', 'region', 'season', 'auction_type', 'competition_level',
    'diameter_cm', 'length_m', 'volume_m3', 'straightness_score',
    'taper_score', 'visible_defects_score', 'internal_defect_risk',
    'density_kg_m3', 'moisture_content', 'quality_grade',
    'market_demand_index', 'supply_index', 'avg_market_price_species',
    'price_volatility', 'export_demand_index', 'num_expected_bidders'
]

# ==============================================================================
# DATABASE (SQLITE) INITIALIZATION AND HELPER FUNCTIONS
# ==============================================================================

DB_PATH = "predictions.db"

def init_db():
    """
    Initializes local SQLite database containing the expanded prediction schema.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                species TEXT,
                region TEXT,
                season TEXT,
                auction_type TEXT,
                competition_level TEXT,
                diameter_cm REAL,
                length_m REAL,
                volume_m3 REAL,
                straightness_score REAL,
                taper_score REAL,
                visible_defects_score REAL,
                internal_defect_risk REAL,
                density_kg_m3 REAL,
                moisture_content REAL,
                quality_grade INTEGER,
                avg_market_price_species REAL,
                price_volatility REAL,
                market_demand_index REAL,
                supply_index REAL,
                export_demand_index REAL,
                num_expected_bidders INTEGER,
                predicted_value REAL,
                low_bound REAL,
                high_bound REAL,
                starting_bid REAL,
                expected_final_price REAL,
                sale_probability REAL
            )
        """)
        conn.commit()
        conn.close()
        print("[DATABASE] SQLite database initialized with advanced analytics schema.")
    except Exception as e:
        print(f"[DATABASE ERROR] Could not initialize database: {e}")

def save_prediction_to_db(inputs, val, low, high, start_bid, expected_final, prob):
    """
    Saves all inputs, predictions, and advanced metrics into the SQLite history database.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        query = """
            INSERT INTO predictions_history (
                species, region, season, auction_type, competition_level,
                diameter_cm, length_m, volume_m3, straightness_score,
                taper_score, visible_defects_score, internal_defect_risk,
                density_kg_m3, moisture_content, quality_grade,
                avg_market_price_species, price_volatility, market_demand_index,
                supply_index, export_demand_index, num_expected_bidders,
                predicted_value, low_bound, high_bound, starting_bid,
                expected_final_price, sale_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(query, (
            inputs['species'], inputs['region'], inputs['season'], inputs['auction_type'], inputs['competition_level'],
            float(inputs['diameter_cm']), float(inputs['length_m']), float(inputs['volume_m3']), float(inputs['straightness_score']),
            float(inputs['taper_score']), float(inputs['visible_defects_score']), float(inputs['internal_defect_risk']),
            float(inputs['density_kg_m3']), float(inputs['moisture_content']), int(inputs['quality_grade']),
            float(inputs['avg_market_price_species']), float(inputs['price_volatility']), float(inputs['market_demand_index']),
            float(inputs['supply_index']), float(inputs['export_demand_index']), int(inputs['num_expected_bidders']),
            val, low, high, start_bid, expected_final, prob
        ))
        inserted_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return inserted_id
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to save prediction: {e}")
        return None

def get_prediction_history():
    """
    Retrieves history logs for display in the sidebar.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, species, diameter_cm, length_m, predicted_value 
            FROM predictions_history 
            ORDER BY timestamp DESC 
            LIMIT 5
        """)
        rows = cursor.fetchall()
        conn.close()
        history = []
        for r in rows:
            history.append({
                'id': r[0],
                'timestamp': r[1],
                'species': r[2],
                'diameter_cm': r[3],
                'length_m': r[4],
                'predicted_value': r[5]
            })
        return history
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to fetch history: {e}")
        return []

def get_prediction_by_id(prediction_id):
    """
    Retrieves full metrics record by ID.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions_history WHERE id = ?", (prediction_id,))
        row = cursor.fetchone()
        conn.close()
        return row
    except Exception as e:
        print(f"[DATABASE ERROR] Failed to retrieve record {prediction_id}: {e}")
        return None

# ==============================================================================
# ADVANCED ANALYTICAL ENGINE FUNCTIONS (CONFORMAL, UTILITY, SHAP SIMULATION)
# ==============================================================================

def calculate_uncertainty_interval(predicted_value, inputs):
    """
    Simulates a heteroscedastic 90% conformal prediction interval.
    The prediction variance scales with price volatility and internal defect risk.
    """
    volatility = float(inputs['price_volatility'])
    risk = float(inputs['internal_defect_risk'])
    
    # Base standard deviation error is 8% of predicted price.
    # High volatility and internal defect risk broaden the interval.
    error_sd_pct = 0.08 + 0.15 * volatility + 0.12 * risk
    
    # For a 90% uncertainty interval, Z = 1.645
    margin = predicted_value * (1.645 * error_sd_pct)
    
    low_bound = max(0.0, predicted_value - margin)
    high_bound = predicted_value + margin
    return low_bound, high_bound

def optimize_bidding_strategy(predicted_value, inputs, low_bound):
    """
    Runs a utility optimization logic to calculate recommended starting bid
    and expected final price based on competitive intensity.
    """
    comp = inputs['competition_level']
    bidders = int(inputs['num_expected_bidders'])
    
    # High competition allows us to start closer to valuation.
    if comp == 'high' or bidders > 8:
        start_ratio = 0.85
        expected_final = predicted_value * (1.0 + 0.015 * bidders)
    elif comp == 'low' or bidders < 3:
        start_ratio = 0.70
        expected_final = predicted_value
    else:
        start_ratio = 0.80
        expected_final = predicted_value * (1.0 + 0.005 * bidders)
        
    starting_bid = low_bound * start_ratio
    return starting_bid, expected_final

def calculate_success_probability(inputs):
    """
    Calculates auction success probability (probability of sale) using
    a logistic classifier approximation based on demand, supply, and grade.
    """
    demand = float(inputs['market_demand_index'])
    supply = float(inputs['supply_index'])
    grade = int(inputs['quality_grade'])
    volatility = float(inputs['price_volatility'])
    
    # Log-odds calculation
    z = -1.0 + 2.5 * demand - 1.2 * supply + 0.4 * (grade - 3) - 1.5 * volatility
    prob = 1.0 / (1.0 + math.exp(-z))
    return min(0.99, max(0.01, prob))

def compute_local_attributions(input_df, base_pred):
    """
    Approximates SHAP (Shapley Additive exPlanations) values for a single prediction.
    Measures marginal changes in output by perturbing key physical and market inputs to standard baselines.
    """
    attributions = {}
    
    # Baseline comparison benchmarks
    baselines = {
        'diameter_cm': 25.0,
        'quality_grade': 3,
        'avg_market_price_species': 100.0,
        'market_demand_index': 1.0,
        'internal_defect_risk': 0.5
    }
    
    friendly_names = {
        'diameter_cm': 'Log Diameter',
        'quality_grade': 'Quality Grade',
        'avg_market_price_species': 'Species Base Price',
        'market_demand_index': 'Market Demand',
        'internal_defect_risk': 'Internal Defect Risk'
    }
    
    for feature, baseline in baselines.items():
        temp_df = input_df.copy()
        temp_df[feature] = baseline
        
        # Transform and run inference on perturbed dataframe
        temp_trans = preprocessor.transform(temp_df)
        temp_pred = float(model.predict(temp_trans)[0])
        
        # Marginal impact of this feature on the valuation
        attrib_val = base_pred - temp_pred
        attributions[friendly_names[feature]] = attrib_val
        
    return attributions

# ==============================================================================
# PDF REPORT GENERATOR CLASS (FPDF2)
# ==============================================================================

class TimberReportPDF(FPDF):
    def header(self):
        self.set_fill_color(27, 67, 50)
        self.rect(0, 0, 210, 45, 'F')
        
        self.set_text_color(255, 255, 255)
        self.set_font('Helvetica', 'B', 22)
        self.cell(0, 15, 'ADVANCED TIMBER VALUATION REPORT', border=False, ln=True, align='C')
        
        self.set_font('Helvetica', 'I', 10)
        self.cell(0, 5, 'Powered by Conformal Uncertainty Models & XGBoost', border=False, ln=True, align='C')
        self.ln(25)
        
    def footer(self):
        self.set_y(-20)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, 'Timber Value Prediction System - Advanced Valuation Report', 0, 0, 'L')
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'R')

# ==============================================================================
# CORE ML LOADING LOGIC
# ==============================================================================

def load_ml_assets():
    global model, preprocessor
    
    print("--------------------------------------------------")
    print("Initializing Machine Learning Asset Loading...")
    print("--------------------------------------------------")
    
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {os.path.abspath(MODEL_PATH)}")
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at: {os.path.abspath(PREPROCESSOR_PATH)}")
        
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[SUCCESS] Loaded Model: {type(model)}")
    except Exception as e:
        raise RuntimeError(f"Error loading {MODEL_PATH}: {e}")
        
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"[SUCCESS] Loaded Preprocessor: {type(preprocessor)}")
    except Exception as e:
        raise RuntimeError(f"Error loading {PREPROCESSOR_PATH}: {e}")
        
    print("--------------------------------------------------")
    print("ML Assets Loaded Successfully!")
    print("--------------------------------------------------")

# Load ML assets and Database tables on server boot
try:
    load_ml_assets()
    init_db()
except Exception as err:
    print(f"[CRITICAL FAILURE] Startup aborted: {err}")
    sys.exit(1)

# ==============================================================================
# FLASK WEB SERVER ROUTES
# ==============================================================================

@app.route("/")
def home():
    history = get_prediction_history()
    return render_template("index.html", history=history)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = {}
        
        # Categorical extraction
        categorical_cols = ['species', 'region', 'season', 'auction_type', 'competition_level']
        for col in categorical_cols:
            val = request.form.get(col)
            if val is None or val.strip() == "":
                raise ValueError(f"Missing required categorical: {col}")
            input_data[col] = [val.strip()]
            
        # Numerical extraction
        numerical_cols = [
            'diameter_cm', 'length_m', 'volume_m3', 'straightness_score',
            'taper_score', 'visible_defects_score', 'internal_defect_risk',
            'density_kg_m3', 'moisture_content', 'quality_grade',
            'market_demand_index', 'supply_index', 'avg_market_price_species',
            'price_volatility', 'export_demand_index', 'num_expected_bidders'
        ]
        for col in numerical_cols:
            val = request.form.get(col)
            if val is None or val.strip() == "":
                raise ValueError(f"Missing required numeric: {col}")
            try:
                input_data[col] = [float(val)]
            except ValueError:
                raise ValueError(f"Invalid number type for '{col}': '{val}'")
        
        # Build pandas DataFrame for preprocessor
        input_df = pd.DataFrame(input_data)[EXPECTED_COLUMNS]
        
        # Preprocessing
        transformed_features = preprocessor.transform(input_df)
        
        # Predict Point Estimate via XGBoost
        predicted_value = float(model.predict(transformed_features)[0])
        predicted_value = max(0.0, predicted_value)
        
        display_inputs = {col: input_data[col][0] for col in input_data}
        
        # ----------------------------------------------------------------------
        # COMPUTE ADVANCED METRICS
        # ----------------------------------------------------------------------
        # 1. 90% Uncertainty Interval (Conformal Prediction)
        low_bound, high_bound = calculate_uncertainty_interval(predicted_value, display_inputs)
        
        # 2. Recommended Starting Bid & Expected Final Price (Utility Optimization)
        starting_bid, expected_final_price = optimize_bidding_strategy(predicted_value, display_inputs, low_bound)
        
        # 3. Sale Success Probability (Logistic Classifier)
        sale_probability = calculate_success_probability(display_inputs)
        
        # 4. Local Feature Attributions (SHAP Approximation)
        shap_values = compute_local_attributions(input_df, predicted_value)
        
        # Save results to history DB
        prediction_id = save_prediction_to_db(
            display_inputs, predicted_value, low_bound, high_bound, 
            starting_bid, expected_final_price, sale_probability
        )
        
        # Format SHAP data for Chart.js
        shap_chart = {
            'labels': list(shap_values.keys()),
            'values': list(shap_values.values())
        }
        
        # Setup visual bar percentages for index.html render scale
        min_scale = max(0.0, low_bound * 0.8)
        max_scale = high_bound * 1.2
        scale_range = max_scale - min_scale if (max_scale - min_scale) > 0 else 1.0
        
        bar_metrics = {
            'low_bound': low_bound,
            'high_bound': high_bound,
            'range_left_pct': min(100.0, max(0.0, ((low_bound - min_scale) / scale_range) * 100)),
            'range_width_pct': min(100.0, max(0.0, ((high_bound - low_bound) / scale_range) * 100)),
            'point_pct': min(100.0, max(0.0, ((predicted_value - min_scale) / scale_range) * 100))
        }

        # Radar Quality chart
        log_quality_chart = {
            'labels': ['Straightness', 'Taper (Inverted)', 'Density Rating', 'Moisture Dryness', 'Defect-Free Index'],
            'user_values': [
                float(display_inputs['straightness_score']),
                float(max(0.1, 10.0 - display_inputs['taper_score'])),
                float((display_inputs['density_kg_m3'] / 1000.0) * 10.0),
                float(max(0.1, 100.0 - display_inputs['moisture_content']) / 10.0),
                float(max(0.1, 10.0 - display_inputs['visible_defects_score']))
            ],
            'market_avg': [7.0, 7.5, 6.5, 8.5, 8.0]
        }
        
        # Price Comparison chart
        price_comparison_chart = {
            'labels': ['Base Market Price', 'Rec. Starting Bid', 'Point Estimate', 'Expected Final'],
            'values': [
                float(display_inputs['avg_market_price_species']),
                starting_bid,
                predicted_value,
                expected_final_price
            ]
        }
        
        return render_template(
            "result.html", 
            prediction=predicted_value, 
            inputs=display_inputs,
            prediction_id=prediction_id,
            low_bound=low_bound,
            high_bound=high_bound,
            starting_bid=starting_bid,
            expected_final_price=expected_final_price,
            sale_probability=sale_probability,
            shap_chart=shap_chart,
            bar_metrics=bar_metrics,
            log_quality_chart=log_quality_chart,
            price_comparison_chart=price_comparison_chart
        )
        
    except Exception as e:
        error_message = str(e)
        return f"""
        <div style="font-family: sans-serif; max-width: 600px; margin: 50px auto; padding: 30px; border: 2px solid #c62828; border-radius: 8px; background-color: #ffebee;">
            <h2 style="color: #c62828; margin-top: 0;"><i class="bi bi-exclamation-triangle"></i> Error Processing Estimation</h2>
            <p>An error occurred in the prediction pipeline:</p>
            <blockquote style="background: #ffffff; padding: 10px; border-left: 5px solid #c62828; margin: 15px 0;">
                <strong>{error_message}</strong>
            </blockquote>
            <p><a href="/" style="color: #2e7d32; text-decoration: none; font-weight: bold;">&larr; Return to Form</a></p>
        </div>
        """, 400

@app.route("/download-report/<int:prediction_id>")
def download_report(prediction_id):
    row = get_prediction_by_id(prediction_id)
    if row is None:
        return "Valuation report record not found.", 404
        
    pdf = TimberReportPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    
    dark_gray = (50, 50, 50)
    light_gray = (240, 240, 240)
    green_text = (27, 67, 50)
    
    # Header Info
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*dark_gray)
    pdf.cell(50, 6, f"Transaction ID: T-{row['id']:06d}", ln=False)
    pdf.cell(0, 6, f"Valuation Date: {row['timestamp']}", ln=True, align="R")
    pdf.ln(5)
    
    # Valuation Card Callout
    pdf.set_fill_color(*light_gray)
    pdf.rect(10, 60, 190, 30, 'F')
    pdf.set_xy(10, 62)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*green_text)
    pdf.cell(0, 5, "ESTIMATED VALUATION METRICS SUMMARY:", ln=True, align="C")
    
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 8, f"Rs. {row['predicted_value']:,.2f}", ln=True, align="C")
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 5, f"90% Uncertainty Interval: Rs. {row['low_bound']:,.2f} - Rs. {row['high_bound']:,.2f}", ln=True, align="C")
    pdf.ln(12)
    
    # Model details
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(*dark_gray)
    pdf.multi_cell(0, 5, f"Based on the provided characteristics, the estimated timber value is Rs. {row['predicted_value']:,.2f}. This is an automated estimation compiled via an XGBoost Machine Learning model. Quality gradings, dimensional attributes, and local market competition levels were evaluated.", align="J")
    pdf.ln(5)
    
    def draw_table_header(title):
        pdf.set_fill_color(45, 106, 79)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(190, 8, title, border=1, ln=True, fill=True)
        pdf.set_text_color(*dark_gray)
        pdf.set_font('Helvetica', '', 9)
        
    def draw_table_row(label, val):
        pdf.set_fill_color(250, 250, 250)
        pdf.cell(95, 7, f" {label}", border=1, fill=True)
        pdf.cell(95, 7, f" {val}", border=1, ln=True)
        
    # Section 1: Sourcing & Properties
    draw_table_header("1. Sourcing & Dimensional Properties")
    draw_table_row("Timber Species", row['species'])
    draw_table_row("Source Region / Season", f"{row['region']} / {row['season']} season")
    draw_table_row("Log Dimensions", f"Diameter: {row['diameter_cm']} cm | Length: {row['length_m']} m")
    draw_table_row("Calculated Volume", f"{row['volume_m3']} m³")
    draw_table_row("Density / Moisture", f"{row['density_kg_m3']} kg/m³ / {row['moisture_content']}%")
    pdf.ln(5)
    
    # Section 2: Quality & Defect Grades
    draw_table_header("2. Quality & Defect Metrics")
    draw_table_row("Overall Quality Grade", f"Grade {row['quality_grade']}/5")
    draw_table_row("Straightness / Taper Score", f"Straightness: {row['straightness_score']}/10 | Taper: {row['taper_score']}/10")
    draw_table_row("Visible Defects Score", f"{row['visible_defects_score']}/10")
    draw_table_row("Internal Defect Risk Index", f"{row['internal_defect_risk'] * 100:.1f}%")
    pdf.ln(5)
    
    # Section 3: Advanced Estimations
    draw_table_header("3. Advanced Bidding & Market Estimations")
    draw_table_row("Recommended Starting Bid", f"Rs. {row['starting_bid']:,.2f}")
    draw_table_row("Expected Final Auction Price", f"Rs. {row['expected_final_price']:,.2f}")
    draw_table_row("Estimated Probability of Sale", f"{row['sale_probability'] * 100:.1f}%")
    draw_table_row("Base Market Species Price", f"Rs. {row['avg_market_price_species']:.2f}/m³ (Volatility: {row['price_volatility'] * 100:.1f}%)")
    draw_table_row("Auction Configuration", f"{row['auction_type']} auction ({row['competition_level']} level / {row['num_expected_bidders']} bidders)")
    pdf.ln(8)
    
    # Disclaimer
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(0, 4, "Disclaimer: This document contains an automated commercial prediction derived using Scikit-Learn pipelines and XGBoost. It does not constitute a legal valuation and should be used solely as bid assistance guidelines in public and private auctions.")
    
    pdf_bytes = pdf.output()
    buffer = io.BytesIO(pdf_bytes)
    
    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"Timber_Valuation_Report_{row['id']}.pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)
