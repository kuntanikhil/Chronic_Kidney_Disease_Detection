from flask import Flask, render_template, request, flash, redirect, send_file, session, url_for
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import sys
import os
import json
import sqlite3
from io import BytesIO
from datetime import datetime
from functools import wraps
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash, check_password_hash

try:
    # Compatibility alias for models pickled with older scikit-learn versions.
    import sklearn.ensemble._forest as sklearn_forest
    sys.modules.setdefault("sklearn.ensemble.forest", sklearn_forest)
except Exception:
    pass

try:
    # Some older pickles reference this deprecated module path.
    import sklearn.tree._classes as sklearn_tree_classes
    sys.modules.setdefault("sklearn.tree.tree", sklearn_tree_classes)
except Exception:
    pass



app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ckd_project_secret_key")

DB_PATH = os.path.join(os.path.dirname(__file__), "kidney_app.db")

KIDNEY_REPORT_FIELDS = [
    {"key": "age", "label": "Age", "range": "1-120 years", "limits": (1, 120)},
    {"key": "bp", "label": "Blood Pressure (BP)", "range": "50-200 mmHg", "limits": (50, 200)},
    {"key": "al", "label": "Albumin (AL)", "range": "0-5", "limits": (0, 5)},
    {"key": "su", "label": "Sugar (SU)", "range": "0-5", "limits": (0, 5)},
    {"key": "rbc", "label": "Red Blood Cells (RBC)", "range": "0-1", "limits": (0, 1)},
    {"key": "pc", "label": "Pus Cell (PC)", "range": "0-1", "limits": (0, 1)},
    {"key": "pcc", "label": "Pus Cell Clumps (PCC)", "range": "0-1", "limits": (0, 1)},
    {"key": "ba", "label": "Bacteria (BA)", "range": "0-1", "limits": (0, 1)},
    {"key": "bgr", "label": "Blood Glucose Random (BGR)", "range": "50-500 mg/dl", "limits": (50, 500)},
    {"key": "bu", "label": "Blood Urea (BU)", "range": "5-250 mg/dl", "limits": (5, 250)},
    {"key": "sc", "label": "Serum Creatinine (SC)", "range": "0.4-15 mg/dl", "limits": (0.4, 15)},
    {"key": "pot", "label": "Potassium (POT)", "range": "2.5-8.5 mEq/L", "limits": (2.5, 8.5)},
    {"key": "wc", "label": "White Blood Cell Count (WC)", "range": "3000-20000 cells/cumm", "limits": (3000, 20000)},
    {"key": "htn", "label": "Hypertension (HTN)", "range": "0-1", "limits": (0, 1)},
    {"key": "dm", "label": "Diabetes Mellitus (DM)", "range": "0-1", "limits": (0, 1)},
    {"key": "cad", "label": "Coronary Artery Disease (CAD)", "range": "0-1", "limits": (0, 1)},
    {"key": "pe", "label": "Pedal Edema (PE)", "range": "0-1", "limits": (0, 1)},
    {"key": "ane", "label": "Anemia (ANE)", "range": "0-1", "limits": (0, 1)},
]


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kidney_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                prediction_value INTEGER NOT NULL,
                prediction_label TEXT NOT NULL,
                input_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        conn.commit()


def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None

    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT id, full_name, email FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()

    if not user:
        return None

    return dict(user)


def login_required(route_func):
    @wraps(route_func)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("loginPage"))
        return route_func(*args, **kwargs)

    return wrapper


@app.context_processor
def inject_current_user():
    return {"current_user": get_current_user()}


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _value_flag(field, value):
    numeric_value = _to_float(value)
    if numeric_value is None:
        return "Review"

    low, high = field["limits"]
    if low <= numeric_value <= high:
        return "In expected range"
    return "Outside expected range"


def _build_kidney_pdf_report(pred, form_values, report_record=None, user_record=None):
    report_stream = BytesIO()
    page = canvas.Canvas(report_stream, pagesize=A4)
    width, height = A4
    y = height - 48

    user_name = "NA"
    user_email = "NA"
    report_id = "NA"
    report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if user_record:
        user_name = user_record.get("full_name", "NA")
        user_email = user_record.get("email", "NA")
    if report_record:
        report_id = report_record.get("id", "NA")
        report_time = report_record.get("created_at", report_time)

    page.setTitle("Kidney Screening Consultation Report")
    page.setFont("Helvetica-Bold", 16)
    page.drawString(42, y, "Kidney Disease Screening Report")

    y -= 20
    page.setFont("Helvetica", 10)
    page.drawString(42, y, f"Generated: {report_time}")
    page.drawString(300, y, f"Report ID: {report_id}")
    y -= 14
    page.drawString(42, y, f"Patient Name: {user_name}")
    page.drawString(300, y, f"Patient Email: {user_email}")
    y -= 14
    page.drawString(42, y, "Report Type: ML-Based Clinical Screening Summary")

    y -= 28
    page.setFont("Helvetica-Bold", 12)
    page.drawString(42, y, "Prediction Outcome")
    y -= 16
    page.setFont("Helvetica", 11)

    if int(pred) == 1:
        page.drawString(42, y, "Model output indicates a higher-risk CKD pattern.")
        y -= 14
        page.drawString(42, y, "Recommendation: Prioritize nephrology consultation and confirmatory lab tests.")
    else:
        page.drawString(42, y, "Model output indicates a lower-risk (healthy) pattern.")
        y -= 14
        page.drawString(42, y, "Recommendation: Continue regular preventive monitoring and follow-up.")

    y -= 24
    page.setFont("Helvetica-Bold", 12)
    page.drawString(42, y, "Submitted Clinical Parameters")
    y -= 16

    page.setFont("Helvetica-Bold", 9)
    page.drawString(42, y, "Parameter")
    page.drawString(240, y, "Submitted Value")
    page.drawString(340, y, "Expected Range")
    page.drawString(475, y, "Status")
    y -= 8
    page.line(42, y, width - 42, y)

    for field in KIDNEY_REPORT_FIELDS:
        y -= 16
        if y < 84:
            page.showPage()
            y = height - 52
            page.setFont("Helvetica-Bold", 12)
            page.drawString(42, y, "Submitted Clinical Parameters (continued)")
            y -= 16
            page.setFont("Helvetica-Bold", 9)
            page.drawString(42, y, "Parameter")
            page.drawString(240, y, "Submitted Value")
            page.drawString(340, y, "Expected Range")
            page.drawString(475, y, "Status")
            y -= 8
            page.line(42, y, width - 42, y)

        value = str(form_values.get(field["key"], "NA"))
        status = _value_flag(field, value)

        page.setFont("Helvetica", 9)
        page.drawString(42, y, field["label"][:33])
        page.drawString(240, y, value[:16])
        page.drawString(340, y, field["range"][:24])
        page.drawString(475, y, status)

    y -= 28
    page.setFont("Helvetica-Bold", 10)
    page.drawString(42, y, "Important Clinical Note")
    y -= 14
    page.setFont("Helvetica", 9)
    page.drawString(42, y, "This report supports screening and consultation readiness only. It is not a final diagnosis.")
    y -= 12
    page.drawString(42, y, "Clinical decisions must be made by a licensed medical professional with confirmatory tests.")

    page.save()
    report_stream.seek(0)
    return report_stream

def predict(values, dic):
    if len(values) == 8:
        model = pickle.load(open('models/diabetes.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 26:
        model = pickle.load(open('models/breast_cancer.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 13:
        model = pickle.load(open('models/heart.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 18:
        model = pickle.load(open('models/kidney.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    elif len(values) == 10:
        model = pickle.load(open('models/liver.pkl','rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]
    else:
        raise ValueError(f"Unsupported input size: {len(values)}")

@app.route("/")
def home():
    return render_template('home.html')


@app.route("/register", methods=['GET', 'POST'])
def registerPage():
    if request.method == 'GET':
        return render_template('register.html')

    full_name = request.form.get("full_name", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    confirm_password = request.form.get("confirm_password", "")

    if not full_name or not email or not password:
        return render_template('register.html', message="Please fill all required fields.")
    if password != confirm_password:
        return render_template('register.html', message="Passwords do not match.")
    if len(password) < 6:
        return render_template('register.html', message="Password must be at least 6 characters.")

    password_hash = generate_password_hash(password)

    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO users (full_name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (full_name, email, password_hash, datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            )
            conn.commit()
    except sqlite3.IntegrityError:
        return render_template('register.html', message="Email already exists. Please login.")

    return redirect(url_for("loginPage"))


@app.route("/login", methods=['GET', 'POST'])
def loginPage():
    if request.method == 'GET':
        return render_template('login.html')

    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")

    with get_db_connection() as conn:
        user = conn.execute(
            "SELECT id, full_name, email, password_hash FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if not user or not check_password_hash(user["password_hash"], password):
        return render_template('login.html', message="Invalid email or password.")

    session["user_id"] = user["id"]
    session["user_name"] = user["full_name"]
    return redirect(url_for("kidneyPage"))


@app.route("/logout", methods=['GET'])
def logoutPage():
    session.clear()
    return redirect(url_for("home"))

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
@login_required
def kidneyPage():
    return render_template('kidney.html')

@app.route("/algorithms", methods=['GET'])
def algorithmsPage():
    return render_template('algorithms.html')

@app.route("/dataset", methods=['GET'])
def datasetPage():
    return render_template('dataset.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods = ['POST', 'GET'])
@login_required
def predictPage():
    if request.method != 'POST':
        return redirect('/')

    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)

        report_inputs = {field["key"]: to_predict_dict.get(field["key"], "") for field in KIDNEY_REPORT_FIELDS}
        prediction_label = "Kidney Disease Risk Detected" if int(pred) == 1 else "Healthy Pattern Detected"

        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO kidney_reports (user_id, prediction_value, prediction_label, input_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session["user_id"],
                    int(pred),
                    prediction_label,
                    json.dumps(report_inputs),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                ),
            )
            conn.commit()
            report_id = cursor.lastrowid
    except Exception as e:
        print(e)
        message = f"Prediction failed: {e}"
        return render_template("home.html", message = message)

    return render_template('predict.html', pred=pred, report_inputs=report_inputs, report_id=report_id)


@app.route("/download-kidney-report", methods=['POST'])
@login_required
def downloadKidneyReport():
    try:
        report_id = request.form.get('report_id', '').strip()
        if not report_id:
            return render_template("home.html", message="Invalid report request.")

        with get_db_connection() as conn:
            report_row = conn.execute(
                """
                SELECT id, user_id, prediction_value, prediction_label, input_json, created_at
                FROM kidney_reports
                WHERE id = ? AND user_id = ?
                """,
                (report_id, session["user_id"]),
            ).fetchone()

        if not report_row:
            return render_template("home.html", message="Report not found for your account.")

        report_record = dict(report_row)
        report_inputs = json.loads(report_record.get("input_json", "{}"))
        report_pdf = _build_kidney_pdf_report(
            report_record["prediction_value"],
            report_inputs,
            report_record=report_record,
            user_record=get_current_user(),
        )

        filename = f"kidney_screening_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(report_pdf, mimetype='application/pdf', as_attachment=True, download_name=filename)
    except Exception as e:
        print(e)
        return render_template("home.html", message=f"Could not generate report: {e}")


@app.route("/reports", methods=['GET'])
@login_required
def reportsPage():
    with get_db_connection() as conn:
        report_rows = conn.execute(
            """
            SELECT id, prediction_value, prediction_label, created_at
            FROM kidney_reports
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (session["user_id"],),
        ).fetchall()

    reports = [dict(row) for row in report_rows]
    return render_template("reports.html", reports=reports)

@app.route("/malariapredict", methods = ['POST', 'GET'])
def malariapredictPage():
    if request.method != 'POST':
        return redirect('/malaria')

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image'])
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,3))
                img = img.astype(np.float64)
                model = load_model("models/malaria.h5")
                pred = np.argmax(model.predict(img)[0])
            else:
                message = "Please upload an Image"
                return render_template('malaria.html', message = message)
        except:
            message = "Please upload an Image"
            return render_template('malaria.html', message = message)
    return render_template('malaria_predict.html', pred = pred)

@app.route("/pneumoniapredict", methods = ['POST', 'GET'])
def pneumoniapredictPage():
    if request.method != 'POST':
        return redirect('/pneumonia')

    if request.method == 'POST':
        try:
            if 'image' in request.files:
                img = Image.open(request.files['image']).convert('L')
                img = img.resize((36,36))
                img = np.asarray(img)
                img = img.reshape((1,36,36,1))
                img = img / 255.0
                model = load_model("models/pneumonia.h5")
                pred = np.argmax(model.predict(img)[0])
            else:
                message = "Please upload an Image"
                return render_template('pneumonia.html', message = message)
        except:
            message = "Please upload an Image"
            return render_template('pneumonia.html', message = message)
    return render_template('pneumonia_predict.html', pred = pred)

if __name__ == '__main__':
    init_db()
    app.run(debug = True)