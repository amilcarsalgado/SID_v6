"""
===============================================================================
SCRIPT: HOURLY ML ORACLE TRAINER & THRESHOLD OPTIMIZER
LOCATION: scripts/train_hourly_model.py
===============================================================================
"""

import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Feature schema matching the inference engine
FEATURE_COLS = [
    'RSI_Extreme_Value',
    'Dist_From_EMA_%',
    'Entry_RVOL',
    'Entry_ATR',
    'Days_To_Earnings',
    'Days_Since_Earnings',
    'Entry_Month',
    'Risk_Tier'
]

TARGET_COL = 'Target_Label'


def load_hourly_dataset() -> pd.DataFrame:
    """Loads the mined hourly dataset from parquet or CSV."""
    dataset_dir = PROJECT_ROOT / "data" / "datasets"
    parquet_path = dataset_dir / "hourly_ml_training_data.parquet"
    csv_path = dataset_dir / "hourly_ml_training_data.csv"

    if parquet_path.exists():
        print(f"📦 Loading dataset from Parquet: {parquet_path.name}")
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        print(f"📄 Loading dataset from CSV: {csv_path.name}")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Dataset not found. Please run 'python scripts/build_hourly_dataset.py' first.")

    # Ensure chronological order
    if 'Entry_Datetime' in df.columns:
        df['Entry_Datetime'] = pd.to_datetime(df['Entry_Datetime'])
        df = df.sort_values('Entry_Datetime').reset_index(drop=True)
    elif 'Entry_Date' in df.columns:
        df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
        df = df.sort_values('Entry_Date').reset_index(drop=True)

    return df


def main():
    print("=" * 70)
    print("🧠 HOURLY ML BRAIN TRAINING ENGINE (IE_ORACLE v6.0 HOURLY)")
    print("=" * 70)

    # 1. Load Dataset
    df = load_hourly_dataset()
    print(f"📊 Total Records Loaded: {len(df):,}")

    # Clean missing values
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    # 2. Chronological 80/20 Train/Test Split
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET_COL]
    X_test, y_test = test_df[FEATURE_COLS], test_df[TARGET_COL]

    train_start = str(train_df['Entry_Date'].iloc[0]) if 'Entry_Date' in train_df.columns else "Start"
    train_end = str(train_df['Entry_Date'].iloc[-1]) if 'Entry_Date' in train_df.columns else "Split"
    test_start = str(test_df['Entry_Date'].iloc[0]) if 'Entry_Date' in test_df.columns else "Split"
    test_end = str(test_df['Entry_Date'].iloc[-1]) if 'Entry_Date' in test_df.columns else "Present"

    print(f"\n📅 Chronological Split Boundary:")
    print(f"  • In-Sample Training (80%)  : {len(X_train):,} samples ({train_start} to {train_end})")
    print(f"  • Out-of-Sample Test (20%) : {len(X_test):,} samples ({test_start} to {test_end})")

    # 3. Model Training
    print("\n⚙️ Training Multi-Class Gradient Boosting Classifier...")
    t0 = time.time()

    # HistGradientBoostingClassifier handles scale variations efficiently
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"✅ Training completed in {time.time() - t0:.2f}s")

    # 4. Out-of-Sample Performance Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n" + "=" * 70)
    print("📈 OUT-OF-SAMPLE TEST PERFORMANCE (20% UNSEEN RECENT DATA)")
    print("=" * 70)
    print(f"Overall Test Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")

    target_names = ["Class 0 (Trap)", "Class 1 (Base Hit / RSI 50)", "Class 2 (Whale / RSI 60)"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # 5. Threshold Calibration Check
    print("\n" + "-" * 70)
    print("🎯 PROBABILITY CALIBRATION TELEMETRY:")
    print(f"  • Average P(Trap / Class 0)  : {y_proba[:, 0].mean():.3f} (Std: {y_proba[:, 0].std():.3f})")
    print(f"  • Average P(Base / Class 1)  : {y_proba[:, 1].mean():.3f} (Std: {y_proba[:, 1].std():.3f})")
    print(f"  • Average P(Whale / Class 2) : {y_proba[:, 2].mean():.3f} (Std: {y_proba[:, 2].std():.3f})")
    print("-" * 70)

    # 6. Export Model Artifact
    models_dir = PROJECT_ROOT / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    export_path = models_dir / "IE_Oracle_v6.0_Hourly.joblib"

    joblib.dump(model, export_path)
    print(f"\n💾 Model successfully serialized and exported to:")
    print(f"   👉 {export_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()