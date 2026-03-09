import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from torch import nn

SEED = 42
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL
SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
ALGORITHMS = {
    "Prior baseline": ("Predicts the training churn prevalence for everyone."),
    "Logistic regression": (
        "A linear combination of features becomes a probability "
        "through a sigmoid."
    ),
    "Decision tree": (
        "A sequence of learned feature splits separates customer " "groups."
    ),
    "Random forest": ("Averages many trees trained on bootstrap samples."),
    "Gradient boosting": (
        "Adds small trees that correct the current prediction errors."
    ),
    "Gaussian Naive Bayes": (
        "Combines class-conditional feature densities under an "
        "independence assumption."
    ),
    "K-nearest neighbors": (
        "Uses the outcomes of nearby customers in scaled, encoded "
        "feature space."
    ),
    "RBF SVM": (
        "Learns a nonlinear margin; internal training-fold "
        "calibration supplies probabilities."
    ),
    "PyTorch MLP": (
        "Uses two hidden ReLU layers, dropout, and binary cross- "
        "entropy with Adam."
    ),
}


def read_data(path):
    data = pd.read_csv(path)
    if data.customerID.duplicated().any():
        raise ValueError("Customer IDs must be unique for this experiment.")
    if not data.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Unexpected target labels.")
    data["TotalCharges"] = pd.to_numeric(data.TotalCharges, errors="coerce")
    return data


def split_data(data):
    train, rest = train_test_split(
        data, test_size=0.4, stratify=data.Churn, random_state=SEED
    )
    validation, test = train_test_split(
        rest, test_size=0.5, stratify=rest.Churn, random_state=SEED
    )
    return train.copy(), validation.copy(), test.copy()


def make_preprocessor():
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC),
            ("categorical", categorical, CATEGORICAL),
        ]
    )


def build_network(width):
    return nn.Sequential(
        nn.Linear(width, 32),
        nn.ReLU(),
        nn.Dropout(0.15),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


class TorchChurn(ClassifierMixin, BaseEstimator):
    def __init__(self, max_epochs=160, patience=18, random_state=SEED):
        self.max_epochs = max_epochs
        self.patience = patience
        self.random_state = random_state

    def _train(self, x, y, epochs, validation=None):
        torch.manual_seed(self.random_state)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        network = build_network(x.shape[1])
        optimizer = torch.optim.Adam(
            network.parameters(), lr=0.001, weight_decay=0.001
        )
        loss_fn = nn.BCEWithLogitsLoss()
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(np.asarray(y), dtype=torch.float32).reshape(-1, 1)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x, y),
            batch_size=128,
            shuffle=True,
        )
        if validation is not None:
            xv = torch.tensor(validation[0], dtype=torch.float32)
            yv = torch.tensor(
                np.asarray(validation[1]), dtype=torch.float32
            ).reshape(-1, 1)
        history, best_loss, stale = [], float("inf"), 0
        best_epoch = epochs
        for epoch in range(1, epochs + 1):
            network.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(network(xb), yb)
                loss.backward()
                optimizer.step()
            network.eval()
            with torch.no_grad():
                train_loss = float(loss_fn(network(x), y))
                valid_loss = (
                    float(loss_fn(network(xv), yv))
                    if validation is not None
                    else np.nan
                )
            history.append((epoch, train_loss, valid_loss))
            if validation is not None:
                if valid_loss < best_loss - 0.0001:
                    best_loss, best_epoch, stale = valid_loss, epoch, 0
                else:
                    stale += 1
                if stale >= self.patience:
                    break
        return network, best_epoch, history

    def fit(self, x, y):
        self.classes_ = np.array([0, 1])
        xa, xb, ya, yb = train_test_split(
            x,
            y,
            test_size=0.15,
            stratify=y,
            random_state=self.random_state,
        )
        inner = make_preprocessor()
        a = inner.fit_transform(xa)
        b = inner.transform(xb)
        _, self.best_epoch_, self.history_ = self._train(
            a, ya, self.max_epochs, validation=(b, yb)
        )
        self.preprocessor_ = make_preprocessor()
        transformed = self.preprocessor_.fit_transform(x)
        self.network_, _, _ = self._train(transformed, y, self.best_epoch_)
        self.network_.eval()
        return self

    def predict_proba(self, x):
        values = self.preprocessor_.transform(x)
        self.network_.eval()
        with torch.no_grad():
            logits = self.network_(torch.tensor(values, dtype=torch.float32))
            p = torch.sigmoid(logits).numpy().ravel()
        return np.column_stack([1 - p, p])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def make_models():
    estimators = {
        "Prior baseline": DummyClassifier(strategy="prior"),
        "Logistic regression": LogisticRegression(max_iter=1500, C=1.0),
        "Decision tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=25, random_state=SEED
        ),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=8,
            max_features="sqrt",
            n_jobs=2,
            random_state=SEED,
        ),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.05,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=SEED,
        ),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "K-nearest neighbors": KNeighborsClassifier(
            n_neighbors=35, weights="distance", n_jobs=2
        ),
        "RBF SVM": SVC(C=1.0, probability=True, random_state=SEED),
    }
    models = {
        name: Pipeline(
            [("prepare", make_preprocessor()), ("model", estimator)]
        )
        for name, estimator in estimators.items()
    }
    models["PyTorch MLP"] = TorchChurn()
    return models


def scores(y, probability, threshold=0.5):
    prediction = (np.asarray(probability) >= threshold).astype(int)
    return {
        "AP": average_precision_score(y, probability),
        "ROC_AUC": roc_auc_score(y, probability),
        "Brier": brier_score_loss(y, probability),
        "Accuracy": accuracy_score(y, prediction),
        "Precision": precision_score(y, prediction, zero_division=0),
        "Recall": recall_score(y, prediction, zero_division=0),
        "F1": f1_score(y, prediction, zero_division=0),
    }


def threshold_table(y, p):
    rows = []
    for threshold in np.linspace(0.01, 0.99, 99):
        tn, fp, fn, tp = confusion_matrix(
            y, np.asarray(p) >= threshold, labels=[0, 1]
        ).ravel()
        rows.append(
            {
                "threshold": threshold,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "TP": tp,
                "flagged": int(tp + fp),
                "precision": tp / max(tp + fp, 1),
                "recall": tp / max(tp + fn, 1),
                "F1": 2 * tp / max(2 * tp + fp + fn, 1),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_difference(y, first, second, metric, repeats=1000):
    rng = np.random.default_rng(SEED)
    y, first, second = map(np.asarray, (y, first, second))
    values = []
    for _ in range(repeats):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) != 2:
            continue
        values.append(metric(y[ids], first[ids]) - metric(y[ids], second[ids]))
    return np.quantile(values, [0.025, 0.975]).tolist()


def validate_inputs(frame, bundle):
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    frame = frame[FEATURES].copy()
    for col in NUMERIC:
        values = pd.to_numeric(frame[col], errors="coerce")
        blank = frame[col].isna() | frame[col].astype(str).str.strip().eq("")
        if col != "TotalCharges" and blank.any():
            raise ValueError(f"{col} cannot be blank.")
        if (values.isna() & ~blank).any():
            raise ValueError(f"{col} must contain numbers.")
        low, high = bundle["bounds"][col]
        if ((values < low) | (values > high)).any():
            raise ValueError(f"{col} must be between {low:g} and {high:g}.")
        frame[col] = values
    if not frame.tenure.dropna().mod(1).eq(0).all():
        raise ValueError("Tenure must be a whole number of months.")
    senior = pd.to_numeric(frame.SeniorCitizen, errors="coerce")
    if not senior.isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1.")
    frame["SeniorCitizen"] = senior.astype(int)
    for col in CATEGORICAL:
        if not frame[col].isin(bundle["categories"][col]).all():
            raise ValueError(f"Unexpected or missing category in {col}.")
    no_phone = frame.PhoneService.eq("No")
    if not no_phone.eq(frame.MultipleLines.eq("No phone service")).all():
        raise ValueError("Phone service and multiple-line choices disagree.")
    no_internet = frame.InternetService.eq("No")
    for col in SERVICE_COLUMNS:
        if not no_internet.eq(frame[col].eq("No internet service")).all():
            raise ValueError(f"{col} does not match InternetService.")
    return frame
