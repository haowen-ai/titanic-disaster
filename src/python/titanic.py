"""Titanic survival model with logistic regression.

Reads train.csv and test.csv from src/data, cleans the data, fits a
logistic regression model and prints every step to the terminal.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Data sits next to this folder: src/data. Same layout in Docker and locally.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 42
SUBMISSION_FILE = "submission_python.csv"
TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
KEEP_TITLES = {"Mr", "Mrs", "Miss", "Master"}
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]
DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"]

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)


def section(title):
    print(f"\n=== {title} ===")


def load_csv(name):
    """Read src/data/<name>. Stop with a clear message if the file is missing."""
    path = DATA_DIR / name
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        print("Download train.csv and test.csv from Kaggle, put them in src/data, "
              "then rebuild the image (see README).", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns -> shape {df.shape}")
    return df


def explore(df):
    """Print shape, types, first rows, missing values and survival rates."""
    print("Column types:")
    print(df.dtypes.to_string())
    print("\nFirst 5 rows:")
    print(df.head().to_string())
    print("\nMissing values per column:")
    print(df.isna().sum().to_string())
    print(f"\nSurvival rate overall: {df['Survived'].mean():.3f}")
    for col in ["Sex", "Pclass", "Embarked"]:
        rates = df.groupby(col)["Survived"].agg(["mean", "count"]).round(3)
        rates.columns = ["survival_rate", "passengers"]
        print(f"\nSurvival rate by {col}:")
        print(rates.to_string())


def extract_title(names):
    """Return the title (Mr, Mrs, Miss, Master or Rare) for each name."""
    # Names look like "Braund, Mr. Owen Harris": the title sits between ", " and "."
    raw = names.str.extract(r",\s*([^.]+)\.", expand=False).str.strip()
    raw = raw.replace(TITLE_MAP)
    return raw.where(raw.isin(KEEP_TITLES), "Rare")


def learn_fill_values(train):
    """Learn fill values from the training data only.

    Returns a dict with age median per title, overall age median,
    most common port and median fare.
    """
    titles = extract_title(train["Name"])
    return {
        "age_by_title": train["Age"].groupby(titles).median().round(1).to_dict(),
        "age_overall": round(float(train["Age"].median()), 1),
        "embarked": train["Embarked"].mode()[0],
        "fare": round(float(train["Fare"].median()), 4),
    }


def clean(df, fill, label, verbose=True):
    """Add features and fill missing values. Prints each change.

    label is "train" or "test" and only shows up in the printed lines.
    Returns a new DataFrame; the input is not changed.
    """
    say = print if verbose else (lambda *args, **kwargs: None)
    out = df.copy()

    out["Title"] = extract_title(out["Name"])
    say(f"[{label}] Added Title from Name: {out['Title'].value_counts().to_dict()}")

    missing_age = out["Age"].isna()
    by_title = out["Title"].map(fill["age_by_title"]).fillna(fill["age_overall"])
    out["Age"] = out["Age"].fillna(by_title)
    say(f"[{label}] Filled {missing_age.sum()} missing Age values with the median age "
        f"of each title (from train): {fill['age_by_title']}")

    missing_port = out["Embarked"].isna().sum()
    out["Embarked"] = out["Embarked"].fillna(fill["embarked"])
    say(f"[{label}] Filled {missing_port} missing Embarked values with '{fill['embarked']}'")

    missing_fare = out["Fare"].isna().sum()
    out["Fare"] = out["Fare"].fillna(fill["fare"])
    say(f"[{label}] Filled {missing_fare} missing Fare values with median {fill['fare']}")

    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = out["Cabin"].notna().astype(int)
    say(f"[{label}] Added FamilySize (SibSp + Parch + 1), IsAlone "
        f"({out['IsAlone'].sum()} alone), HasCabin ({out['HasCabin'].sum()} with a cabin)")

    out = out.drop(columns=DROP_COLUMNS)
    say(f"[{label}] Dropped {DROP_COLUMNS}; shape now {out.shape}")
    say(f"[{label}] Missing values left: {int(out.drop(columns='Survived', errors='ignore').isna().sum().sum())}")
    return out


def encode(df, columns=None):
    """One-hot encode the categorical columns.

    If columns is given (the training columns), the result is reindexed to
    match it so train and test line up. Returns (X, column_list).
    """
    X = pd.get_dummies(df.drop(columns="Survived", errors="ignore"),
                       columns=CATEGORICAL, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    X = X.astype(float)
    return X, list(X.columns)


def build_model():
    """Scaler + logistic regression in one pipeline."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])


def accuracy(model, X, y):
    """Share of correct predictions."""
    return float((model.predict(X) == y).mean())


def main():
    section("Step 1: load train.csv")
    train = load_csv("train.csv")

    section("Step 2: explore the training data")
    explore(train)

    section("Step 3: clean the data and add features")
    fill = learn_fill_values(train)
    train_clean = clean(train, fill, "train")
    X, columns = encode(train_clean)
    y = train_clean["Survived"]
    print(f"One-hot encoded {CATEGORICAL}; {len(columns)} features:")
    print(columns)

    section("Step 4: logistic regression, check on a holdout set")
    # Kaggle's test.csv has no answers, so keep 20% of train.csv aside to see
    # how the model does on passengers it has not seen.
    part, holdout = train_test_split(train, test_size=0.2, random_state=SEED,
                                     stratify=train["Survived"])
    part_fill = learn_fill_values(part)
    part_clean = clean(part, part_fill, "train part", verbose=False)
    holdout_clean = clean(holdout, part_fill, "holdout", verbose=False)
    X_part, part_cols = encode(part_clean)
    X_hold, _ = encode(holdout_clean, part_cols)
    model = build_model().fit(X_part, part_clean["Survived"])
    print(f"Split train.csv 80/20 (seed {SEED}): {len(part)} rows to fit, {len(holdout)} rows held out")
    print(f"Accuracy on the 80% used to fit: {accuracy(model, X_part, part_clean['Survived']):.4f}")
    print(f"Holdout accuracy (20% not used to fit): {accuracy(model, X_hold, holdout_clean['Survived']):.4f}")

    section("Step 5: fit on all of train.csv")
    model = build_model().fit(X, y)
    print(f"Training accuracy (all {len(y)} rows): {accuracy(model, X, y):.4f}")
    coefs = pd.Series(model.named_steps["logreg"].coef_[0], index=columns)
    print("Coefficients on scaled features (positive = more likely to survive):")
    print(coefs.sort_values(ascending=False).round(3).to_string())

    section("Step 6: load test.csv and predict")
    test = load_csv("test.csv")
    print("Missing values per column:")
    print(test.isna().sum().to_string())
    test_clean = clean(test, fill, "test")
    X_test, _ = encode(test_clean, columns)
    preds = model.predict(X_test)
    print(f"Made {len(preds)} predictions; predicted survivors: {int(preds.sum())} "
          f"({preds.mean():.3f} of test passengers)")
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": preds})
    print("First 10 predictions:")
    print(submission.head(10).to_string(index=False))
    # Same format Kaggle asks for: PassengerId,Survived
    submission.to_csv(DATA_DIR / SUBMISSION_FILE, index=False)
    print(f"Saved all {len(submission)} predictions to src/data/{SUBMISSION_FILE}")

    section("Step 7: test accuracy")
    if "Survived" in test.columns:
        print(f"Test accuracy: {accuracy(model, X_test, test['Survived']):.4f}")
    else:
        print("test.csv has no Survived column (Kaggle keeps the answers), so test accuracy "
              "cannot be computed here.")
        print("Use the holdout accuracy from Step 4 as the estimate. The real test score "
              f"comes from uploading src/data/{SUBMISSION_FILE} to Kaggle (score in README).")


if __name__ == "__main__":
    main()
