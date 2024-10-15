from flask import Flask, render_template, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import sklearn.datasets as datasets
import warnings
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting

app = Flask(__name__)

# Ignore warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [10, 5]

# Load and process the breast cancer dataset
def load_and_process_data():
    data = datasets.load_breast_cancer()

    # Load the data into a pandas DataFrame
    data_frame = pd.DataFrame(data.data, columns=data.feature_names)

    # Add the target variable (label)
    data_frame['label'] = data.target

    # Preview the raw data
    raw_data_preview = data_frame.head(5).to_html(classes='table table-striped', index=False)

    # Generate a heatmap of correlations between features
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_frame.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.savefig('static/correlation_matrix.png')
    plt.clf()

    # New plot - Missing Data Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(data_frame.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.savefig('static/missing_data_heatmap.png')
    plt.clf()

    # New plot - Statistical Measures of Features
    data_frame.describe().plot(kind='bar', figsize=(15, 10))
    plt.title("Statistical Measures of Features")
    plt.xticks(rotation=90)
    plt.ylabel("Value")
    plt.savefig('static/statistical_measures.png')
    plt.clf()

    return data_frame, raw_data_preview

# Train models and return their accuracy scores
def train_models(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

    model_scores = {}
    models = {}

    # Logistic Regression (best model selected)
    log_reg = LogisticRegression(max_iter=200)
    log_reg.fit(x_train, y_train)
    y_pred_log_reg = log_reg.predict(x_test)
    log_reg_accuracy = round(accuracy_score(y_test, y_pred_log_reg) * 100, 2)
    model_scores["Logistic Regression"] = log_reg_accuracy
    models["Logistic Regression"] = log_reg

    # Decision Tree Classifier
    decision_tree = DecisionTreeClassifier(random_state=2)
    decision_tree.fit(x_train, y_train)
    y_pred_tree = decision_tree.predict(x_test)
    tree_accuracy = round(accuracy_score(y_test, y_pred_tree) * 100, 2)
    model_scores["Decision Tree"] = tree_accuracy
    models["Decision Tree"] = decision_tree

    # SVM Classifier
    svm = SVC()
    svm.fit(x_train, y_train)
    y_pred_svm = svm.predict(x_test)
    svm_accuracy = round(accuracy_score(y_test, y_pred_svm) * 100, 2)
    model_scores["SVM"] = svm_accuracy
    models["SVM"] = svm

    # New Plot - Model Evaluation (Logistic Regression)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_log_reg)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Logistic Regression Model Evaluation")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.savefig('static/log_reg_model_evaluation.png')
    plt.clf()

    # New Plot - Model Evaluation (Decision Tree)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_tree)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Decision Tree Model Evaluation")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.savefig('static/decision_tree_model_evaluation.png')
    plt.clf()

    # New Plot - Model Evaluation (SVM)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_svm)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("SVM Model Evaluation")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.savefig('static/svm_model_evaluation.png')
    plt.clf()

    return model_scores, models, x_test, y_test

# Function to evaluate model performance
def evaluate_model(model, X_test, Y_test):
    Y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(Y_test, Y_pred)

    return accuracy, precision, recall, f1, roc_auc, conf_matrix

# Generate boxplot and histogram visualizations for the features
def generate_feature_plots(data_frame):
    # Boxplots
    plt.figure(figsize=(15, 10))
    data_frame.boxplot()
    plt.title("Boxplots of Features")
    plt.xticks(rotation=90)
    plt.savefig('static/boxplots.png')
    plt.clf()

    # Histograms
    data_frame.hist(bins=30, figsize=(20, 20))
    plt.title("Histograms of Features")
    plt.savefig('static/histograms.png')
    plt.clf()

# Route for the homepage
@app.route('/')
def index():
    # Load and process the data
    data_frame, raw_data_preview = load_and_process_data()

    # Generate feature plots
    generate_feature_plots(data_frame)

    # Split the features and the target variable
    X = data_frame.drop(columns='label', axis=1)
    Y = data_frame['label']

    # Train models and get accuracy scores
    model_scores, models, X_test, Y_test = train_models(X, Y)

    # Get first 10 rows of the processed dataset
    first_10_rows = data_frame.head(10).to_html(classes='table table-striped', index=False)

    # Evaluate models and store metrics
    evaluation_results = {}
    for model_name, model in models.items():
        eval_metrics = evaluate_model(model, X_test, Y_test)
        evaluation_results[model_name] = {
            "accuracy": eval_metrics[0],
            "precision": eval_metrics[1],
            "recall": eval_metrics[2],
            "f1": eval_metrics[3],
            "roc_auc": eval_metrics[4],
            "conf_matrix": eval_metrics[5].tolist()  # Convert to list for JSON serialization
        }

    # Compare model performance (e.g., based on accuracy)
    model_names = list(evaluation_results.keys())
    accuracies = [results["accuracy"] for results in evaluation_results.values()]

    plt.figure(figsize=(8, 6))
    plt.bar(model_names, accuracies)
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.savefig('static/model_performance_comparison.png')  # Save the comparison plot
    plt.clf()

    # Sort model scores
    sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    # Determine the best model
    best_model_name = sorted_scores[0][0]
    best_model_score = sorted_scores[0][1]

    # Retrieve the metrics for the best model
    best_model_metrics = evaluation_results[best_model_name]

    return render_template('index.html', raw_data_preview=raw_data_preview, 
                           scores=sorted_scores, first_10_rows=first_10_rows, 
                           best_model_name=best_model_name, best_model_score=best_model_score,
                           best_model_metrics=best_model_metrics, evaluation_results=evaluation_results)  # Pass metrics to template

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/portfolio')
def portfolio():
    return render_template('portfolio.html')

if __name__ == '__main__':
    app.run(debug=True)
