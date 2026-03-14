from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.modeling import FEATURES, NUMERIC, read_data, scores
from src.modeling import split_data, validate_inputs

ROOT = Path(__file__).resolve().parent


def main():
    data = read_data(ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    train, validation, test = split_data(data)
    manifest = pd.read_csv(ROOT / "reports/split_manifest.csv")
    assert manifest.customerID.is_unique
    assert set(manifest.customerID) == set(data.customerID)
    for name, expected in zip(
        ["train", "validation", "test"], [train, validation, test]
    ):
        actual = set(manifest.loc[manifest.split.eq(name), "customerID"])
        assert actual == set(expected.customerID)

    bundle = joblib.load(ROOT / "models/churn_model.joblib")
    valid_metrics = pd.read_csv(
        ROOT / "reports/validation_metrics.csv", index_col="model"
    )
    assert bundle["name"] == valid_metrics.AP.idxmax()
    table = pd.read_csv(ROOT / "reports/validation_thresholds.csv")
    choice = table.sort_values(
        ["F1", "threshold"], ascending=[False, False]
    ).iloc[0]
    np.testing.assert_allclose(bundle["threshold"], choice.threshold)

    predicted = bundle["model"].predict_proba(test[FEATURES])[:, 1]
    stored = pd.read_csv(ROOT / "reports/test_predictions.csv")
    assert stored.customerID.tolist() == test.customerID.tolist()
    np.testing.assert_allclose(predicted, stored[bundle["name"]])
    actual_scores = scores(
        test.Churn.eq("Yes").astype(int), predicted, bundle["threshold"]
    )
    for metric, value in actual_scores.items():
        np.testing.assert_allclose(value, bundle["test_scores"][metric])
    model = bundle["model"]
    processor = (
        model.preprocessor_
        if hasattr(model, "preprocessor_")
        else model.named_steps["prepare"]
    )
    imputer = processor.named_transformers_["numeric"].named_steps["impute"]
    np.testing.assert_allclose(imputer.statistics_, train[NUMERIC].median())
    print(
        "Customer isolation, training-only imputation, and saved scores pass."
    )

    sample = pd.DataFrame([bundle["example"]], columns=FEATURES)
    validate_inputs(sample, bundle)
    missing_total = sample.copy()
    missing_total["TotalCharges"] = np.nan
    validate_inputs(missing_total, bundle)
    assert np.isfinite(model.predict_proba(missing_total)).all()
    invalid_cases = []
    unknown = sample.copy()
    unknown["Contract"] = "Unknown plan"
    invalid_cases.append(unknown)
    negative = sample.copy()
    negative["MonthlyCharges"] = -10
    invalid_cases.append(negative)
    inconsistent = sample.copy()
    inconsistent["InternetService"] = "No"
    inconsistent["TechSupport"] = "Yes"
    invalid_cases.append(inconsistent)
    invalid_cases.append(sample.drop(columns="tenure"))
    for invalid in invalid_cases:
        try:
            validate_inputs(invalid, bundle)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted.")
    print("Missing values and CSV validation edge cases pass.")

    notebook = nbformat.read(ROOT / "Telco_Customer_Churn.ipynb", as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert cell.execution_count is not None
            assert not any(o.output_type == "error" for o in cell.outputs)
    print("All notebook code cells executed without errors.")

    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=90)
    app.run()
    assert not app.exception, app.exception
    app.radio(key="workspace").set_value("Customer studio").run()
    assert not app.exception, app.exception
    app.button(key="score_customer").click().run()
    assert not app.exception, app.exception
    expected = model.predict_proba(sample)[0, 1]
    assert app.metric[0].value == f"{expected:.1%}"
    new_charge = min(
        float(sample.MonthlyCharges.iloc[0]) + 5,
        bundle["bounds"]["MonthlyCharges"][1],
    )
    app.number_input(key="MonthlyCharges").set_value(new_charge)
    app.button(key="score_customer").click().run()
    assert not app.exception, app.exception
    sample["MonthlyCharges"] = new_charge
    expected = model.predict_proba(sample)[0, 1]
    assert app.metric[0].value == f"{expected:.1%}"
    app.selectbox(key="InternetService").select("No").run()
    assert not app.exception, app.exception
    assert app.selectbox(key="TechSupport").value == "No internet service"
    app.button(key="score_customer").click().run()
    assert not app.exception, app.exception
    assert not app.error, app.error
    app.number_input(key="MonthlyCharges").set_value(new_charge - 1).run()
    assert len(app.metric) == 0
    for workspace in [
        "Overview",
        "Batch scoring",
        "Model evidence",
        "3D explorer",
        "Methodology",
    ]:
        app.radio(key="workspace").set_value(workspace).run()
        assert not app.exception, app.exception
    app.radio(key="workspace").set_value("3D explorer").run()
    app.radio(key="scene").set_value("Decision costs").run()
    assert not app.exception, app.exception
    print("All workspaces load; predictions and stale-result handling pass.")


if __name__ == "__main__":
    main()
