import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")


def compute_shap_values(model, X_train, X_test=None, max_evals=1000):
    """
    Compute SHAP values for a model.

    Parameters:
    - model: Trained model (must be compatible with shap)
    - X_train: Training features (pandas DataFrame)
    - X_test: Test features (optional, uses X_train if None)
    - max_evals: Maximum evaluations for shap (for tree models)

    Returns:
    - shap_values: SHAP values
    - explainer: SHAP explainer object
    """
    if X_test is None:
        X_test = X_train

    # Try different explainers based on model type
    try:
        if hasattr(model, "predict_proba"):
            # For classification models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
        else:
            # For regression models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
    except:
        # Fallback to permutation explainer
        explainer = shap.PermutationExplainer(model.predict, X_train)
        shap_values = explainer.shap_values(X_test, max_evals=max_evals)

    return shap_values, explainer


def plot_shap_summary(shap_values, X_test, feature_names=None, max_display=20):
    """
    Plot SHAP summary plot.

    Parameters:
    - shap_values: SHAP values from compute_shap_values
    - X_test: Test features
    - feature_names: List of feature names (optional)
    - max_display: Maximum features to display
    """
    if feature_names is None and hasattr(X_test, "columns"):
        feature_names = X_test.columns.tolist()

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X_test,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Feature Importance Summary")
    plt.tight_layout()
    plt.show()


def plot_partial_dependence(
    model, X_train, features, target_feature=None, kind="average"
):
    """
    Plot Partial Dependence Plots (PDP).

    Parameters:
    - model: Trained model
    - X_train: Training features
    - features: List of features or single feature for PDP
    - target_feature: Target feature name for title
    - kind: 'average' or 'individual' for PDP
    """
    if isinstance(features, str):
        features = [features]

    fig, ax = plt.subplots(figsize=(10, 6))

    try:
        PartialDependenceDisplay.from_estimator(
            model, X_train, features, ax=ax, kind=kind
        )

        title = f"Partial Dependence Plot"
        if target_feature:
            title += f" for {target_feature}"
        plt.title(title)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating PDP: {e}")
        print("Trying alternative plotting...")

        # Fallback: simple PDP calculation
        feature = features[0] if isinstance(features, list) else features
        feature_values = X_train[feature].unique()
        feature_values.sort()

        partial_deps = []
        for val in feature_values:
            X_temp = X_train.copy()
            X_temp[feature] = val
            pred = model.predict(X_temp)
            partial_deps.append(np.mean(pred))

        plt.figure(figsize=(8, 6))
        plt.plot(feature_values, partial_deps, "b-", linewidth=2)
        plt.xlabel(feature)
        plt.ylabel("Partial Dependence")
        plt.title(f"Partial Dependence Plot for {feature}")
        plt.grid(True, alpha=0.3)
        plt.show()


def plot_feature_importance(model, feature_names=None, top_n=20):
    """
    Plot feature importance from the model.

    Parameters:
    - model: Trained model with feature_importances_ attribute
    - feature_names: List of feature names
    - top_n: Number of top features to display
    """
    if not hasattr(model, "feature_importances_"):
        print("Model does not have feature_importances_ attribute")
        return

    importances = model.feature_importances_

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importances))]

    # Sort features by importance
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importances[::-1])
    plt.yticks(range(len(top_features)), top_features[::-1])
    plt.xlabel("Feature Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\nTop Feature Importances:")
    for i, (feature, importance) in enumerate(zip(top_features, top_importances)):
        print(f"{i+1}. {feature}: {importance:.3f}")


def explain_model(model, X_train, X_test, feature_names=None, target_name="Target"):
    """
    Comprehensive model explanation using SHAP, PDP, and feature importance.

    Parameters:
    - model: Trained model
    - X_train: Training features
    - X_test: Test features for SHAP
    - feature_names: Feature names
    - target_name: Name of target variable
    """
    print(f"Explaining model for {target_name}")
    print("=" * 50)

    # 1. Feature Importance
    print("\n1. Feature Importance:")
    plot_feature_importance(model, feature_names)

    # 2. SHAP Analysis
    print("\n2. SHAP Analysis:")
    try:
        shap_values, explainer = compute_shap_values(model, X_train, X_test)
        plot_shap_summary(shap_values, X_test, feature_names)
    except Exception as e:
        print(f"SHAP analysis failed: {e}")

    # 3. Partial Dependence Plots for top features
    print("\n3. Partial Dependence Plots:")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        # Get top 3 features
        top_indices = np.argsort(importances)[::-1][:3]
        top_features = [feature_names[i] for i in top_indices]

        for feature in top_features:
            if feature in X_train.columns:
                print(f"\nPDP for {feature}:")
                plot_partial_dependence(model, X_train, feature, target_name)

    print("\nModel explanation complete!")
