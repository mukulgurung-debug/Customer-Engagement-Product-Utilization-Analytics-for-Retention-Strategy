import os
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def engineer_features(data):
    df = data.copy()
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['TenureAgeRatio'] = df['Tenure'] / (df['Age'] + 1)
    df['CreditScoreAgeRatio'] = df['CreditScore'] / (df['Age'] + 1)
    df['IsZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['EngagementScore'] = df['IsActiveMember'] * 2 + df['HasCrCard'] + (df['NumOfProducts'] >= 2).astype(int)
    df['RelationshipStrengthIndex'] = (df['NumOfProducts'] / 4.0) * 0.4 + df['IsActiveMember'] * 0.4 + df['HasCrCard'] * 0.2
    return df

def main():
    print("Loading data...")
    csv_file = DATA_DIR / "3.csv"
    if not csv_file.exists():
        csv_file = BASE_DIR / "3.csv"
    
    df = pd.read_csv(csv_file)
    drop_cols = [c for c in ['Year', 'CustomerId', 'Surname'] if c in df.columns]
    df_clean = df.drop(columns=drop_cols)
    df_feat = engineer_features(df_clean)
    
    X = df_feat.drop('Exited', axis=1)
    y = df_feat['Exited']
    
    categorical_features = ['Geography', 'Gender']
    numerical_features = [col for col in X.columns if col not in categorical_features]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ]
    )
    
    # 1. Benchmark Logistic Regression
    pipe_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipe_lr.fit(X_train, y_train)
    p_lr = pipe_lr.predict_proba(X_test)[:, 1]
    y_lr = pipe_lr.predict(X_test)
    
    # 2. Benchmark Random Forest
    pipe_rf = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42))
    ])
    pipe_rf.fit(X_train, y_train)
    p_rf = pipe_rf.predict_proba(X_test)[:, 1]
    y_rf = pipe_rf.predict(X_test)
    
    # 3. Production Gradient Boosting Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42))
    ])
    
    print("Training Gradient Boosting Pipeline...")
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    
    # Save model comparison table
    comparison_df = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            "Accuracy": round(accuracy_score(y_test, y_lr), 4),
            "Precision": round(precision_score(y_test, y_lr), 4),
            "Recall": round(recall_score(y_test, y_lr), 4),
            "F1-Score": round(f1_score(y_test, y_lr), 4),
            "ROC-AUC": round(roc_auc_score(y_test, p_lr), 4)
        },
        {
            "Model": "Random Forest",
            "Accuracy": round(accuracy_score(y_test, y_rf), 4),
            "Precision": round(precision_score(y_test, y_rf), 4),
            "Recall": round(recall_score(y_test, y_rf), 4),
            "F1-Score": round(f1_score(y_test, y_rf), 4),
            "ROC-AUC": round(roc_auc_score(y_test, p_rf), 4)
        },
        {
            "Model": "Gradient Boosting (Production)",
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1-Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        }
    ])
    comparison_df.to_csv(MODELS_DIR / "model_comparison.csv", index=False)
    
    # Save model metrics json
    total_customers = len(df)
    churn_n = int(df['Exited'].sum())
    metrics = {
        "total_customers": total_customers,
        "total_churners": churn_n,
        "portfolio_churn_rate": round(float(churn_n / total_customers), 4),
        "at_risk_capital": round(float(df[df['Exited'] == 1]['Balance'].sum()), 2),
        "avg_churner_balance": round(float(df[df['Exited'] == 1]['Balance'].mean()), 2),
        "avg_retained_balance": round(float(df[df['Exited'] == 0]['Balance'].mean()), 2),
        "best_model": "Gradient Boosting Classifier",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(auc), 4)
    }
    with open(MODELS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
        
    # Fit full dataset and export model
    pipeline.fit(X, y)
    joblib.dump(pipeline, MODELS_DIR / "model.pkl")
    joblib.dump(pipeline, BASE_DIR / "model.pkl")
    
    # Feature importances
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_cols = list(cat_encoder.get_feature_names_out(categorical_features))
    feature_names = numerical_features + encoded_cat_cols
    importances = pipeline.named_steps['clf'].feature_importances_
    
    feat_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    feat_df.to_csv(MODELS_DIR / "feature_importance.csv", index=False)
    
    # Score customers dataset
    df_scored = df.copy()
    full_feat = engineer_features(df_scored)
    X_full = full_feat[X.columns]
    df_scored['ChurnProbability'] = pipeline.predict_proba(X_full)[:, 1]
    df_scored['PredictedExited'] = (df_scored['ChurnProbability'] >= 0.5).astype(int)
    df_scored['RiskTier'] = pd.cut(
        df_scored['ChurnProbability'],
        bins=[-0.01, 0.30, 0.60, 1.0],
        labels=['Low Risk', 'Moderate Risk', 'High Risk']
    )
    df_scored.to_csv(DATA_DIR / "customers_risk_scored.csv", index=False)
    print("Model serialized and exported successfully!")

if __name__ == '__main__':
    main()
