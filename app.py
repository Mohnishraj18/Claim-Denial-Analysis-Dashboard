import os
os.environ["MPLBACKEND"] = "Agg"

import pandas as pd
from flask import Flask, render_template, request
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import joblib

app = Flask(__name__)

# ----------------- Load Model -----------------
model_path = "denial_model.joblib"
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")
else:
    print("⚠ No trained model found. Only dashboard will work.")

# ----------------- Helpers -----------------
def plot_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

def normalize_columns(df):
    """Normalize messy headers to expected format"""
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]", "", regex=True)
    )

    col_map = {
        "cptcode": "CPT_Code",
        "procedurecode": "CPT_Code",
        "insurancecompany": "Insurance_Company",
        "payername": "Insurance_Company",
        "insurancename": "Insurance_Company",
        "physicianname": "Physician_Name",
        "doctorfullname": "Physician_Name",
        "doctorname": "Physician_Name",
        "paymentamount": "Payment_Amount",
        "paidamount": "Payment_Amount",
        "balance": "Balance",
        "balanceamt": "Balance",
        "outstandingbalance": "Balance",
        "denialreason": "Denial_Reason",
        "reasonfordenial": "Denial_Reason",
    }

    df.rename(columns={col: col_map[col] for col in df.columns if col in col_map}, inplace=True)
    return df

def load_csv(file):
    required_cols = {"CPT_Code", "Insurance_Company", "Physician_Name", "Payment_Amount", "Balance", "Denial_Reason"}
    for h in [0, 1, 2]:
        try:
            file.seek(0)
            df = pd.read_csv(file, header=h, skip_blank_lines=True)
            df = normalize_columns(df)
            if required_cols.issubset(df.columns):
                return df
        except Exception:
            continue
    return None

# ----------------- Routes -----------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "❌ No file part in request", 400
        file = request.files["file"]

        if file.filename == "":
            return "❌ No file selected", 400

        if file and file.filename.endswith(".csv"):
            csv_data = io.StringIO(file.read().decode("utf-8"))
            df = load_csv(csv_data)
            if df is None:
                return "❌ Could not detect correct header format or missing required columns.", 400

            # ---- Data Cleaning ----
            df["Payment_Amount"] = pd.to_numeric(df["Payment_Amount"].replace(r"[\$,]", "", regex=True), errors="coerce").fillna(0)
            df["Balance"] = pd.to_numeric(df["Balance"].replace(r"[\$,]", "", regex=True), errors="coerce").fillna(0)
            df["Is_Denied"] = df["Denial_Reason"].fillna("").apply(lambda x: x.strip() != "")

            # ---- Summary Tables ----
            denied_cpt_counts = df[df["Is_Denied"]].groupby("CPT_Code").size().reset_index(name="Denied_Claims")
            total_cpt_counts = df.groupby("CPT_Code").size().reset_index(name="Total_Claims")
            cpt_analysis = pd.merge(total_cpt_counts, denied_cpt_counts, on="CPT_Code", how="left").fillna(0)
            cpt_analysis["Denial_Rate"] = (cpt_analysis["Denied_Claims"] / cpt_analysis["Total_Claims"] * 100).round(2)
            cpt_analysis_html = cpt_analysis.sort_values(by="Denied_Claims", ascending=False).to_html(classes="styled-table", index=False)

            payer_denials = df[df["Is_Denied"]].groupby("Insurance_Company").size().reset_index(name="Denied_Claims_Count").sort_values(by="Denied_Claims_Count", ascending=False)
            payer_denials_html = payer_denials.to_html(classes="styled-table", index=False)

            provider_denials = df[df["Is_Denied"]].groupby("Physician_Name").size().reset_index(name="Denied_Claims_Count").sort_values(by="Denied_Claims_Count", ascending=False)
            provider_denials_html = provider_denials.to_html(classes="styled-table", index=False)

            # ---- Root Cause Detection ----
            denial_reasons = df[df["Is_Denied"]]["Denial_Reason"].dropna().astype(str)

            root_causes = {
                "Modifier Issues": denial_reasons.str.contains("modifier", case=False).sum(),
                "LCD/NCD Mismatch": denial_reasons.str.contains("LCD|NCD|medical necessity", case=False).sum(),
                "Bundling Edits (NCCI)": denial_reasons.str.contains("bundl", case=False).sum(),
                "Lack of Documentation": denial_reasons.str.contains("documentation|record|16", case=False).sum(),
                "Prior Authorization Problems": denial_reasons.str.contains("prior authorization|auth", case=False).sum(),
                "Credentialing or Enrollment Issues": denial_reasons.str.contains("credentialing|enrollment|provider eligibility", case=False).sum(),
                "Charge Exceeds Fee Schedule": denial_reasons.str.contains("45|charge exceeds fee schedule", case=False).sum(),
                "Non-covered Service": denial_reasons.str.contains("96|non-covered service", case=False).sum(),
            }

            reasons = {
                "Modifier Issues": "Missing/incorrect modifiers can lead to denials.",
                "LCD/NCD Mismatch": "Claim doesn’t meet Medicare LCD/NCD criteria.",
                "Bundling Edits (NCCI)": "Service bundled incorrectly under NCCI edits.",
                "Lack of Documentation": "Insufficient documentation to prove necessity.",
                "Prior Authorization Problems": "No valid prior authorization at time of service.",
                "Credentialing or Enrollment Issues": "Provider not enrolled/credentialed properly.",
                "Charge Exceeds Fee Schedule": "Claimed charge exceeded the payer’s fee schedule.",
                "Non-covered Service": "Service not covered under patient’s insurance plan.",
            }

            root_causes_df = pd.DataFrame(root_causes.items(), columns=["Root Cause", "Count"])
            root_causes_df["Logical Reason"] = root_causes_df["Root Cause"].map(reasons)
            root_causes_html = root_causes_df.sort_values(by="Count", ascending=False).to_html(classes="styled-table", index=False)

            recommendations = {
                "Modifier Issues": "✔ Train staff on modifier rules, run pre-checks before submission.",
                "LCD/NCD Mismatch": "✔ Check coverage policies, attach additional docs when appealing.",
                "Bundling Edits (NCCI)": "✔ Educate coders on bundling, only override with documentation.",
                "Lack of Documentation": "✔ Use templates, audit charts before billing.",
                "Prior Authorization Problems": "✔ Track prior auths, verify before service.",
                "Credentialing or Enrollment Issues": "✔ Audit provider credentialing regularly.",
                "Charge Exceeds Fee Schedule": "✔ Review fee schedules; negotiate rates with payers.",
                "Non-covered Service": "✔ Inform patients before service, use ABNs when required.",
            }
            recommendations_df = pd.DataFrame(recommendations.items(), columns=["Root Cause", "Recommended Strategy"])
            recommendations_html = recommendations_df.to_html(classes="styled-table", index=False)

            # ---- Charts ----
            cpt_chart, payer_chart, provider_chart = None, None, None

            if not cpt_analysis.empty:
                top_cpt = cpt_analysis.sort_values("Denied_Claims", ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(12, max(6, len(top_cpt) * 0.6)), dpi=120)
                sns.barplot(x="Denied_Claims", y="CPT_Code", data=top_cpt, color="steelblue", ax=ax)
                ax.set_title("Top CPT Codes by Denials")
                cpt_chart = plot_to_base64(fig)

            if not payer_denials.empty:
                top_payers = payer_denials.head(10)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
                sns.barplot(x="Denied_Claims_Count", y="Insurance_Company", data=top_payers, color="salmon", ax=ax)
                ax.set_title("Top Payers by Denials")
                payer_chart = plot_to_base64(fig)

            if not provider_denials.empty:
                top_providers = provider_denials.head(10)
                fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
                sns.barplot(x="Denied_Claims_Count", y="Physician_Name", data=top_providers, color="seagreen", ax=ax)
                ax.set_title("Top Providers by Denials")
                provider_chart = plot_to_base64(fig)

            return render_template(
                "index.html",
                cpt_analysis=cpt_analysis_html,
                payer_denials=payer_denials_html,
                provider_denials=provider_denials_html,
                root_causes=root_causes_html,
                recommendations=recommendations_html,
                cpt_chart=cpt_chart,
                payer_chart=payer_chart,
                provider_chart=provider_chart,
            )

    return render_template(
        "index.html",
        cpt_analysis=None,
        payer_denials=None,
        provider_denials=None,
        root_causes=None,
        recommendations=None,
        cpt_chart=None,
        payer_chart=None,
        provider_chart=None,
    )

if __name__ == "__main__":
    print("🚀 Starting Flask app on http://127.0.0.1:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
