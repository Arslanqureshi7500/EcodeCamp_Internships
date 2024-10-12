from flask import Flask, render_template, url_for
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting

app = Flask(__name__)

# Ignore warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = [10, 5]
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load and process data
def load_and_process_data():
    full_data = pd.read_csv('titanic_dataset.csv')

    # Data shape
    print('train data:', full_data.shape)

    # View first few rows (before processing)
    raw_data_preview = full_data.head(5).to_html(classes='table table-striped', index=False)  # Convert to HTML table

    # Heatmap of missing data
    sns.heatmap(full_data.isnull(), yticklabels=False, cbar=False, cmap='tab20c_r')
    plt.title('Missing Data: Training Set')
    plt.savefig('static/missing_data_heatmap.png')
    plt.clf()

    # Imputation function
    def impute_age(cols):
        Age = cols[0]
        Pclass = cols[1]

        if pd.isnull(Age):
            if Pclass == 1:
                return 37
            elif Pclass == 2:
                return 29
            else:
                return 24
        else:
            return Age

    # Apply the function to the Age column
    full_data['Age'] = full_data[['Age', 'Pclass']].apply(impute_age, axis=1)

    # Remove Cabin feature
    full_data.drop('Cabin', axis=1, inplace=True)

    # Remove rows with missing data
    full_data.dropna(inplace=True)

    # Remove unnecessary columns
    full_data.drop(['Name', 'Ticket'], axis=1, inplace=True)

    # Convert objects to category data type
    objcat = ['Sex', 'Embarked']
    for colname in objcat:
        full_data[colname] = full_data[colname].astype('category')

    # Remove PassengerId
    full_data.drop('PassengerId', inplace=True, axis=1)

    # Convert categorical variables into 'dummy' or indicator variables
    sex = pd.get_dummies(full_data['Sex'], drop_first=True)
    embarked = pd.get_dummies(full_data['Embarked'], drop_first=True)

    # Add new dummy columns to data frame
    full_data = pd.concat([full_data, sex, embarked], axis=1)
    full_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)

    return full_data, raw_data_preview  # Return both the processed data and the raw data preview

# Train the models
def train_models(x, y):
    # Split data to be used in the models
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=101)

    model_scores = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=200)  # Increased max_iter for convergence
    log_reg.fit(x_train, y_train)
    y_pred_log_reg = log_reg.predict(x_test)
    log_reg_accuracy = round(accuracy_score(y_test, y_pred_log_reg) * 100, 2)
    model_scores["Logistic Regression"] = log_reg_accuracy

    # Decision Tree
    Dtree = DecisionTreeClassifier(random_state=101)  # Added random_state for reproducibility
    Dtree.fit(x_train, y_train)
    y_pred_Dtree = Dtree.predict(x_test)
    Dtree_accuracy = round(accuracy_score(y_test, y_pred_Dtree) * 100, 2)
    model_scores["Decision Tree Classifier"] = Dtree_accuracy

    # Random Forest
    rfc = RandomForestClassifier(random_state=101)  # Added random_state for reproducibility
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    rfc_accuracy = round(accuracy_score(y_test, y_pred_rfc) * 100, 2)
    model_scores["Random Forest Classifier"] = rfc_accuracy

    # Gradient Boosting
    gbc = GradientBoostingClassifier(random_state=101)  # Added random_state for reproducibility
    gbc.fit(x_train, y_train)
    y_pred_gbc = gbc.predict(x_test)
    gbc_accuracy = round(accuracy_score(y_test, y_pred_gbc) * 100, 2)
    model_scores["Gradient Boosting Classifier"] = gbc_accuracy

    return model_scores

# Generate the Age Distribution plot
def generate_age_distribution_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.histplot(full_data['Age'].dropna(), kde=False, bins=30)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('static/age_distribution.png')  # Save in /static folder
    plt.close()

# Generate the Parch Histogram plot
def generate_parch_histogram_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.histplot(full_data['Parch'], kde=False, bins=10)
    plt.title('Parch Histogram')
    plt.xlabel('Parch')
    plt.ylabel('Count')
    plt.savefig('static/parch_histogram.png')  # Save in /static folder
    plt.close()

# Generate distribution plot
def generate_dist_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.histplot(full_data['Age'].dropna(), kde=False, bins=30)  # Changed to histplot
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.savefig('static/dist_palette_plot.png')  # Save in /static folder
    plt.close()

def generate_swarm_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='Pclass', y='Age', hue='Survived', data=full_data)
    plt.title('Age Distribution by Class and Survival')
    plt.xlabel('Class')
    plt.ylabel('Age')
    plt.savefig('static/swarm_plot.png')  # Save in /static folder
    plt.close()

def generate_count_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Pclass', hue='Survived', data=full_data)
    plt.title('Survival Count by Class')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.savefig('static/count_plot.png')  # Save in /static folder
    plt.close()

def generate_point_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='Pclass', y='Age', hue='Survived', data=full_data)
    plt.title('Age Distribution by Class and Survival')
    plt.xlabel('Class')
    plt.ylabel('Age')
    plt.savefig('static/point_plot.png')  # Save in /static folder
    plt.close()

def generate_correlation_heatmap_plot(full_data):
    plt.figure(figsize=(10, 6))
    sns.heatmap(full_data.corr(), cmap = "YlGnBu", annot=True, fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig('static/correlation_heatmap.png')  # Save in /static folder
    plt.close()

@app.route('/')
def index():
    # Load and process the data, and get raw dataset preview
    full_data, raw_data_preview = load_and_process_data()

    # Generate plots
    generate_age_distribution_plot(full_data)
    generate_parch_histogram_plot(full_data)
    generate_dist_plot(full_data)
    generate_swarm_plot(full_data)
    generate_count_plot(full_data)
    generate_point_plot(full_data)
    generate_correlation_heatmap_plot(full_data)

    # Split features and target variable
    x = full_data.drop('Survived', axis=1)
    y = full_data['Survived']

    # Train models and get scores
    model_scores = train_models(x, y)

    # Get first 10 rows of processed dataset
    first_10_rows = full_data.head(10).to_html(classes='table table-striped', index=False)

    # Sort the model scores
    sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    return render_template('index.html', raw_data_preview=raw_data_preview, scores=sorted_scores, first_10_rows=first_10_rows)

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
