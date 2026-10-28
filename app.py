from flask import Flask, render_template_string, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained Random Forest model
MODEL_PATH = "random_forest_algorithms.pkl"
try:
    model = joblib.load(MODEL_PATH)
    # Automatically retrieve feature names if available in the model object
    if hasattr(model, "feature_names_in_"):
        FEATURE_NAMES = list(model.feature_names_in_)
    else:
        # Fallback default features if not embedded
        FEATURE_NAMES = [
            "company_name",
            "industry",
            "country",
            "company_size",
            "month",
            "year",
            "layoffs_count",
            "layoff_percentage",
            "reason_for_layoffs",
            "ai_automation_impact",
            "ai_replacement_risk",
            "open_roles",
            "hiring_trend",
            "remote_jobs_percentage",
            "top_hiring_role",
            "stock_growth_percent",
            "revenue_growth_percent",
            "salary_budget_change",
            "ai_adoption_level",
            "employee_sentiment",
            "job_security_score",
        ]
except Exception as e:
    model = None
    FEATURE_NAMES = []
    print(f"Error loading model: {e}")

# HTML Template with a modern, attractive, and responsive layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Random Forest Prediction Portal</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #4f46e5;
            --primary-hover: #4338ca;
            --bg-color: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #1e293b;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 40px 20px;
        }

        .container {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.05);
            padding: 40px;
            overflow: hidden;
            margin-bottom: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 30px;
        }

        header h1 {
            font-size: 26px;
            font-weight: 700;
            color: var(--text-main);
            margin-bottom: 8px;
        }

        header p {
            color: var(--text-muted);
            font-size: 14px;
        }

        .alert-error {
            background-color: #fef2f2;
            border: 1px solid #f87171;
            color: #991b1b;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
            font-size: 14px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 20px;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        label {
            font-size: 13px;
            font-weight: 600;
            color: var(--text-main);
            margin-bottom: 6px;
            text-transform: capitalize;
        }

        input {
            padding: 10px 14px;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.2s ease;
            outline: none;
        }

        input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
        }

        .btn-submit {
            grid-column: 1 / -1;
            background-color: var(--primary);
            color: white;
            border: none;
            padding: 14px;
            border-radius: 8px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            transition: background-color 0.2s;
            margin-top: 10px;
        }

        .btn-submit:hover {
            background-color: var(--primary-hover);
        }

        .result-card {
            grid-column: 1 / -1;
            background: #eef2ff;
            border: 1px solid #c7d2fe;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin-top: 10px;
        }

        .result-card h3 {
            color: var(--primary);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }

        .result-card p {
            font-size: 22px;
            font-weight: 700;
            color: #312e81;
        }

        footer {
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
            margin-top: 20px;
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <h1>Random Forest Model Inference</h1>
            <p>Fill in the required parameters below to generate a real-time prediction.</p>
        </header>

        {% if error %}
            <div class="alert-error">{{ error }}</div>
        {% endif %}

        <form method="POST" action="/">
            <div class="form-grid">
                {% for feature in features %}
                <div class="form-group">
                    <label for="{{ feature }}">{{ feature | replace('_', ' ') }}</label>
                    <input type="text" id="{{ feature }}" name="{{ feature }}" value="{{ form_data.get(feature, '') }}" required>
                </div>
                {% endfor %}

                <button type="submit" class="btn-submit">Run Prediction</button>
            </div>
        </form>

        {% if prediction is not none %}
        <div class="result-card">
            <h3>Prediction Result</h3>
            <p>{{ prediction }}</p>
        </div>
        {% endif %}
    </div>

    <footer>
        Powered by Flask & Render &bull; Scikit-Learn Model Deployment
    </footer>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    form_data = {}

    if model is None:
        error = "Model file could not be loaded. Please ensure 'random_forest_algorithms.pkl' is uploaded in your directory."

    if request.method == "POST" and model is not None:
        try:
            form_data = request.form
            raw_values = [form_data.get(f) for f in FEATURE_NAMES]

            processed_values = []
            for val in raw_values:
                try:
                    processed_values.append(float(val))
                except ValueError:
                    # Fallback conversion for text/categorical inputs
                    processed_values.append(float(hash(val) % 100))

            features_array = np.array([processed_values])
            pred_result = model.predict(features_array)
            prediction = str(pred_result[0])
        except Exception as e:
            error = f"Inference Error: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        features=FEATURE_NAMES,
        prediction=prediction,
        error=error,
        form_data=form_data,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)