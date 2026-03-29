import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def init_explainer(model):
    """
    Initialize SHAP explainer
    """
    return shap.TreeExplainer(model)


def explain_prediction(model, X_sample, feature_names):
    """
    Generate SHAP explanation for one sample
    """

    explainer = shap.TreeExplainer(model)

    # Ensure correct shape
    if isinstance(X_sample, pd.DataFrame):
        X = X_sample.values
    else:
        X = np.array(X_sample).reshape(1, -1)

    # Get SHAP values
    shap_values = explainer.shap_values(X)

    # Handle multiclass
    if isinstance(shap_values, list):
        shap_values = shap_values[0]   # take first class safely

    # Convert to 1D array safely
    values = np.array(shap_values[0]).flatten()

    # Convert all values to float (VERY IMPORTANT)
    values = [float(v) for v in values]

    # Create contributions dictionary
    contributions = dict(zip(feature_names, values))

    # Sort safely by absolute importance
    contributions = dict(
        sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    )

    return contributions


def print_explanation(contributions, top_n=5):
    """
    Print explanation in simple language
    """

    print("\n🧠 Model Explanation:\n")

    for feat, val in list(contributions.items())[:top_n]:
        direction = "increases" if val > 0 else "decreases"
        print(f"- {feat} {direction} resistance prediction (impact: {val:.4f})")


def plot_shap_bar(model, X_sample, feature_names, save_path=None):
    """
    Plot SHAP bar chart
    """

    explainer = shap.TreeExplainer(model)

    if isinstance(X_sample, pd.DataFrame):
        X = X_sample
    else:
        X = pd.DataFrame([X_sample], columns=feature_names)

    shap_values = explainer.shap_values(X)

    # Handle multiclass
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap.summary_plot(shap_values, X, plot_type="bar", show=False)

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()