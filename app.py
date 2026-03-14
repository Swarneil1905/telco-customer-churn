import io
import json
from html import escape
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from src.modeling import ALGORITHMS, FEATURES, SERVICE_COLUMNS
from src.modeling import validate_inputs

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
st.set_page_config(
    page_title="Signal | Telco churn",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "models/churn_model.joblib")


@st.cache_data
def read_report(name):
    return pd.read_csv(REPORTS / name)


def load_chart(name):
    figure = pio.read_json(REPORTS / name)
    figure.update_layout(height=640)
    return figure


def choose(label, field, values, default):
    index = values.index(default) if default in values else 0
    return st.selectbox(label, values, index=index, key=field)


def numeric(label, field, bundle, default):
    low, high = bundle["bounds"][field]
    if field == "tenure":
        return st.number_input(
            label,
            min_value=int(low),
            max_value=int(high),
            value=int(default),
            step=1,
            key=field,
        )
    return st.number_input(
        label,
        min_value=low,
        max_value=high,
        value=float(default),
        step=1.0,
        key=field,
    )


def go_to_customer():
    st.session_state.workspace = "Customer studio"


def page_header(kicker, title, description):
    st.markdown(
        f'<div class="eyebrow">{escape(kicker)}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.write(description)


def note(label, title, body):
    st.markdown(
        '<div class="insight-card">'
        f'<div class="eyebrow">{escape(label)}</div>'
        f"<h3>{escape(title)}</h3><p>{escape(body)}</p></div>",
        unsafe_allow_html=True,
    )


def score_gauge(probability):
    figure = go.Figure(
        go.Indicator(
            mode="gauge",
            value=probability * 100,
            gauge=dict(
                axis=dict(range=[0, 100], ticksuffix="%", tickwidth=0),
                bar=dict(color="#df6b3f", thickness=0.65),
                bgcolor="#f2eee7",
                borderwidth=0,
                steps=[dict(range=[0, 100], color="#f2eee7")],
                threshold=dict(
                    line=dict(color="#152b3c", width=4),
                    thickness=0.9,
                    value=threshold * 100,
                ),
            ),
        )
    )
    figure.update_layout(
        height=220,
        margin=dict(l=35, r=35, t=25, b=15),
        paper_bgcolor="white",
        font=dict(color="#152b3c"),
    )
    return figure


style = (ROOT / "assets/styles.css").read_text(encoding="utf-8")
st.markdown(f"<style>{style}</style>", unsafe_allow_html=True)
if not (ROOT / "models/churn_model.joblib").exists():
    st.info("Run the project notebook once to create the model and reports.")
    st.stop()

bundle = load_bundle()
summary = json.loads(
    (REPORTS / "run_summary.json").read_text(encoding="utf-8")
)
example = bundle["example"]
categories = bundle["categories"]
threshold = bundle["threshold"]
selected = bundle["name"]
metrics = summary["test_scores"]

with st.sidebar:
    st.markdown(
        '<div class="brand">S<span>:</span> SIGNAL</div>'
        '<div class="brand-sub">CUSTOMER RETENTION LAB</div>',
        unsafe_allow_html=True,
    )
    workspace = st.radio(
        "Workspace",
        [
            "Overview",
            "Customer studio",
            "Batch scoring",
            "Model evidence",
            "3D explorer",
            "Methodology",
        ],
        key="workspace",
    )
    st.divider()
    st.caption("THE EXPERIMENT")
    st.write("**7,043** sample customers")
    st.write("**8** classifiers + a baseline")
    st.caption("Train / validation / test: 60 / 20 / 20")
    st.divider()
    st.caption(
        "A retrospective classification study. Scores are estimates, "
        "not validated forecasts of next-month churn."
    )

if workspace == "Overview":
    st.markdown(
        '<section class="hero">'
        '<div class="eyebrow">TELCO / CLASSIFICATION STUDY 02</div>'
        "<h1>Understand who<br>might leave.</h1>"
        "<p>From customer profiles to churn signals. Explore the evidence "
        "behind the score, then choose how to interpret it.</p>"
        '<div class="hero-tags"><span>7,043 customers</span>'
        "<span>19 input features</span>"
        "<span>PyTorch + scikit-learn</span></div></section>",
        unsafe_allow_html=True,
    )
    left, middle, right = st.columns(3)
    left.metric("TEST RANKING / AP", f"{metrics['AP']:.3f}")
    middle.metric("TEST RECALL", f"{metrics['Recall']:.1%}")
    right.metric("TEST PRECISION", f"{metrics['Precision']:.1%}")
    st.caption(
        f"{selected} on 1,409 test customers. "
        f"Precision and recall use the frozen {threshold:.2f} threshold."
    )
    chart_column, story_column = st.columns([1.55, 1], gap="large")
    with chart_column:
        st.subheader("Model comparison")
        st.caption("Held-out average precision. Higher is better.")
        comparison = read_report("test_metrics.csv").sort_values("AP")
        comparison["selection"] = comparison.model.map(
            lambda name: (
                "Validation selection"
                if name == selected
                else "Other candidates"
            )
        )
        figure = px.bar(
            comparison,
            x="AP",
            y="model",
            orientation="h",
            color="selection",
            text="AP",
            color_discrete_map={
                "Validation selection": "#df6b3f",
                "Other candidates": "#a9bec7",
            },
        )
        figure.update_traces(
            texttemplate="%{text:.3f}", textposition="outside"
        )
        figure.update_layout(
            height=380,
            showlegend=False,
            margin=dict(l=0, r=30, t=10, b=10),
            paper_bgcolor="#faf8f4",
            plot_bgcolor="#faf8f4",
            font=dict(color="#152b3c"),
            xaxis_title=None,
            yaxis_title=None,
            xaxis=dict(range=[0, 0.75]),
        )
        figure.update_yaxes(
            categoryorder="array", categoryarray=comparison.model.tolist()
        )
        st.plotly_chart(figure, width="stretch", theme=None)
    with story_column:
        st.subheader("Read the signal")
        note(
            "01 / THE OPERATING POINT",
            "More coverage. More false alarms.",
            "At threshold 0.27, test F1 rises from 0.551 to 0.621. "
            "Recall increases, while precision falls.",
        )
        note(
            "02 / THE MODEL CHOICE",
            "A close field, not a clear winner.",
            "Gradient boosting was selected on validation. Its test AP "
            "advantage over logistic regression is inconclusive. "
            "The test ranking does not change that selection.",
        )
        st.button(
            "Open customer studio",
            type="primary",
            on_click=go_to_customer,
            width="stretch",
        )

elif workspace == "Customer studio":
    page_header(
        "WORKSPACE / 01",
        "A customer, in focus.",
        "Build a profile on the left. Score it to see where it falls "
        "against the validation-selected decision threshold.",
    )
    editor, score_column = st.columns([1.6, 1], gap="large")
    values = {}
    with editor:
        with st.container(border=True):
            st.subheader("01 / Account & billing")
            first, second = st.columns(2)
            with first:
                values["tenure"] = numeric(
                    "Tenure (months)", "tenure", bundle, example["tenure"]
                )
                values["Contract"] = choose(
                    "Contract",
                    "Contract",
                    categories["Contract"],
                    example["Contract"],
                )
                values["PaperlessBilling"] = choose(
                    "Paperless billing",
                    "PaperlessBilling",
                    categories["PaperlessBilling"],
                    example["PaperlessBilling"],
                )
            with second:
                values["MonthlyCharges"] = numeric(
                    "Monthly charges",
                    "MonthlyCharges",
                    bundle,
                    example["MonthlyCharges"],
                )
                values["PaymentMethod"] = choose(
                    "Payment method",
                    "PaymentMethod",
                    categories["PaymentMethod"],
                    example["PaymentMethod"],
                )
                unknown = st.checkbox("Total charges are unknown")
                if unknown:
                    values["TotalCharges"] = float("nan")
                else:
                    total = example["TotalCharges"]
                    if pd.isna(total):
                        total = bundle["bounds"]["TotalCharges"][0]
                    values["TotalCharges"] = numeric(
                        "Total charges", "TotalCharges", bundle, total
                    )
        with st.expander("02 / Connected services", expanded=True):
            first, second = st.columns(2)
            with first:
                phone = choose(
                    "Phone service",
                    "PhoneService",
                    categories["PhoneService"],
                    example["PhoneService"],
                )
                values["PhoneService"] = phone
                options = (
                    ["No phone service"] if phone == "No" else ["No", "Yes"]
                )
                values["MultipleLines"] = choose(
                    "Multiple lines",
                    "MultipleLines",
                    options,
                    example["MultipleLines"],
                )
            with second:
                internet = choose(
                    "Internet service",
                    "InternetService",
                    categories["InternetService"],
                    example["InternetService"],
                )
                values["InternetService"] = internet
            options = (
                ["No internet service"] if internet == "No" else ["No", "Yes"]
            )
            labels = [
                "Online security",
                "Online backup",
                "Device protection",
                "Tech support",
                "Streaming TV",
                "Streaming movies",
            ]
            for i, (label, field) in enumerate(zip(labels, SERVICE_COLUMNS)):
                with first if i % 2 == 0 else second:
                    values[field] = choose(
                        label, field, options, example[field]
                    )
        with st.expander("03 / Customer profile"):
            first, second = st.columns(2)
            fields = [
                ("Gender (dataset category)", "gender"),
                ("Senior citizen (0 = no, 1 = yes)", "SeniorCitizen"),
                ("Partner", "Partner"),
                ("Dependents", "Dependents"),
            ]
            for i, (label, field) in enumerate(fields):
                with first if i % 2 == 0 else second:
                    values[field] = choose(
                        label, field, categories[field], example[field]
                    )
            st.caption(
                "These fields reproduce the benchmark. Operational use "
                "requires a separate fairness review."
            )
    with score_column:
        with st.container(border=True):
            st.caption("THE CUSTOMER SIGNAL")
            st.subheader("Risk snapshot")
            signature = pd.DataFrame([values])[FEATURES].to_json()
            clicked = st.button(
                "Estimate churn risk",
                key="score_customer",
                type="primary",
                width="stretch",
            )
            if clicked:
                try:
                    sample = validate_inputs(pd.DataFrame([values]), bundle)
                    probability = float(
                        bundle["model"].predict_proba(sample)[0, 1]
                    )
                    st.session_state.scored_profile = (signature, probability)
                except ValueError as error:
                    st.error(str(error))
            previous = st.session_state.get("scored_profile")
            if previous and previous[0] == signature:
                probability = previous[1]
                st.metric("Estimated churn probability", f"{probability:.1%}")
                st.plotly_chart(
                    score_gauge(probability), width="stretch", theme=None
                )
                flag = probability >= threshold
                st.markdown(
                    "**Above the review threshold**"
                    if flag
                    else "**Below the review threshold**"
                )
                st.caption(
                    "The dark gauge marker is the decision threshold. "
                    "A score is not a certainty or a contact recommendation."
                )
            else:
                st.markdown(
                    '<div class="empty-score">--</div>', unsafe_allow_html=True
                )
                st.write(
                    "Profile ready to score"
                    if not previous
                    else "Profile changed. Score again to update."
                )
                st.caption("Your result will appear here.")
            st.divider()
            st.write(f"**Model**  {selected}")
            st.write(f"**Decision threshold**  {threshold:.2f}")
            st.caption("Selected on validation F1 and fixed before testing.")
        st.caption(
            "Defaults represent one training customer. Bounds and service "
            "checks do not establish that every combination is realistic."
        )

elif workspace == "Batch scoring":
    page_header(
        "WORKSPACE / 02",
        "A list becomes a shortlist.",
        "Upload customer profiles, calculate risk scores, and download "
        "the results for review.",
    )
    st.write(
        "Supply the 19 model columns using the template. An optional "
        "customerID is carried into the output. Churn labels are ignored."
    )
    st.download_button(
        "Download input template",
        pd.DataFrame([example], columns=FEATURES).to_csv(index=False),
        file_name="churn_input_template.csv",
        mime="text/csv",
    )
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(io.BytesIO(uploaded.getvalue()))
            if not 1 <= len(raw) <= 5000:
                raise ValueError("Upload between 1 and 5,000 customers.")
            clean = validate_inputs(raw, bundle)
            output = pd.DataFrame(
                {
                    "row": range(1, len(clean) + 1),
                    "churn_probability": bundle["model"].predict_proba(clean)[
                        :, 1
                    ],
                }
            )
            if "customerID" in raw:
                output.insert(1, "customerID", raw.customerID)
            output["flagged"] = output.churn_probability >= threshold
            st.dataframe(output, hide_index=True, width="stretch")
            st.download_button(
                "Download predictions",
                output.to_csv(index=False),
                "churn_predictions.csv",
                "text/csv",
            )
            st.caption(
                "The app processes uploads in memory "
                "and does not save them to disk."
            )
        except (ValueError, pd.errors.ParserError) as error:
            st.error(str(error))
elif workspace == "Model evidence":
    page_header(
        "EVIDENCE / 01",
        "Make the score accountable.",
        "Compare held-out performance, inspect errors, and explore "
        "the precision-recall trade-off.",
    )
    st.subheader(f"Selected on validation: {selected}")
    a, b, c, d = st.columns(4)
    metrics = summary["test_scores"]
    a.metric("Test average precision", f"{metrics['AP']:.3f}")
    b.metric("Test ROC AUC", f"{metrics['ROC_AUC']:.3f}")
    c.metric("Recall at frozen threshold", f"{metrics['Recall']:.1%}")
    d.metric("Precision at frozen threshold", f"{metrics['Precision']:.1%}")
    st.caption(
        f"Held-out customers: {summary['split_sizes']['test']:,}. "
        "The table below uses threshold 0.50 for every model. "
        "Ranking metrics are independent of the decision threshold."
    )
    st.dataframe(
        read_report("test_metrics.csv").sort_values("AP", ascending=False),
        hide_index=True,
        width="stretch",
    )
    st.image(str(FIGURES / "model_comparison.png"), width="stretch")
    st.image(str(FIGURES / "confusion_matrices.png"), width="stretch")
    with st.expander("Ranking, calibration, and feature sensitivity"):
        st.image(str(FIGURES / "ranking_curves.png"), width="stretch")
        st.image(str(FIGURES / "calibration.png"), width="stretch")
        st.image(str(FIGURES / "permutation_importance.png"), width="stretch")
    st.subheader("Validation threshold explorer")
    st.caption(
        "Exploration here does not change "
        "the saved model or its decision rule."
    )
    table = read_report("validation_thresholds.csv")
    explored = st.slider(
        "Explore a threshold", 0.01, 0.99, float(threshold), 0.01
    )
    row = table.iloc[(table.threshold - explored).abs().argmin()]
    left, middle, right = st.columns(3)
    left.metric("Validation precision", f"{row.precision:.1%}")
    middle.metric("Validation recall", f"{row.recall:.1%}")
    right.metric("Customers flagged", f"{int(row.flagged):,}")
    figure = px.line(
        table,
        x="threshold",
        y=["precision", "recall", "F1"],
        labels={"value": "Validation score"},
        template="plotly_white",
    )
    figure.add_vline(x=explored, line_dash="dash")
    st.plotly_chart(figure, width="stretch", theme=None)
    with st.expander("Subgroup checks and measured findings"):
        st.caption(
            "Descriptive error differences; small groups can be noisy. "
            "Not a fairness certification."
        )
        st.dataframe(read_report("subgroup_metrics.csv"), hide_index=True)
        st.markdown((REPORTS / "findings.md").read_text())

elif workspace == "3D explorer":
    page_header(
        "EVIDENCE / 02",
        "Turn the evidence around.",
        "Rotate, zoom, and inspect two different views of churn behavior.",
    )
    scene = st.radio(
        "Choose a scene",
        ["Customer cohorts", "Decision costs"],
        horizontal=True,
        key="scene",
    )
    chart_column, explanation = st.columns([2.5, 1], gap="large")
    with chart_column:
        name = (
            "cohort_3d.json" if scene == "Customer cohorts" else "cost_3d.json"
        )
        st.plotly_chart(load_chart(name), width="stretch", theme=None)
    with explanation:
        st.caption("HOW TO READ THIS VIEW")
        if scene == "Customer cohorts":
            st.subheader("Where churn concentrates")
            st.write(
                "Training customers grouped by tenure and monthly charges. "
                "Higher points mean a higher observed churn rate."
            )
            st.write("**Marker size** reflects customer count.")
            st.write("**Vertical bars** show 95% Wilson intervals.")
            st.write("**Minimum group size:** 20 customers.")
            st.caption(
                "These are associations. Contract and service differences "
                "can contribute; changing a bill is not a tested treatment."
            )
        else:
            st.subheader("What does a mistake cost?")
            st.write(
                "The surface combines validation false alarms and missed "
                "churners while varying the decision threshold and cost ratio."
            )
            st.code("100 * (FP + ratio * FN) / customers", language=None)
            st.write("**One false alarm:** one hypothetical cost unit.")
            st.write("**One missed churner:** 1 to 10 cost units.")
            st.caption(
                "This is not retention profit or the measured effect of "
                "an offer. The costs are illustrative assumptions."
            )
    st.divider()
    st.subheader("The hypothesis ledger")
    st.dataframe(
        read_report("hypotheses.csv"), hide_index=True, width="stretch"
    )
    st.caption(
        "1,000 paired customer bootstrap samples. Intervals exclude "
        "uncertainty from retraining and model selection."
    )

elif workspace == "Methodology":
    page_header(
        "FIELD NOTES",
        "Inside the experiment.",
        "Eight ways to learn a churn signal, "
        "measured against a prior baseline.",
    )
    names, detail = st.columns([1, 2], gap="large")
    with names:
        algorithm = st.radio("Algorithm", list(ALGORITHMS))
    with detail:
        with st.container(border=True):
            st.subheader(algorithm)
            st.write(ALGORITHMS[algorithm])
            if algorithm == "PyTorch MLP":
                st.image(
                    str(FIGURES / "neural_learning_curve.png"), width="stretch"
                )
                st.caption(
                    "32 and 16 hidden units, ReLU, dropout, Adam, BCE. "
                    "CPU execution keeps deployment small."
                )
            else:
                table = read_report("test_metrics.csv")
                st.dataframe(
                    table[table.model.eq(algorithm)],
                    hide_index=True,
                    width="stretch",
                )
        st.subheader("The evaluation sequence")
        st.write(
            "**01 Train.** Fit preprocessing inside each training pipeline. "
            "Use three-fold cross-validation within the training partition."
        )
        st.write(
            "**02 Select.** Choose the model by validation average precision, "
            "then its threshold by validation F1."
        )
        st.write(
            "**03 Measure.** Evaluate the fixed choices on untouched test "
            "customers. Test ranking never changes the selected model."
        )
        st.caption(
            "The sample has no dates or interventions. Future forecasting, "
            "causal explanations, and retention ROI require new data."
        )
