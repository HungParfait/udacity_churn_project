# library doc string
"""
    This is a library for calculate churn rate.
"""
# import libraries
import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, RocCurveDisplay

sns.set()
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

EDA_IMAGES_FOLDER = "./images/eda"
LOG_PATH = "./logs/churn_library.log"
RESULT_FOLDER = "./images/results"

logging.basicConfig(filename=LOG_PATH, filemode='a')

CAT_COLUMNS = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category'
]

QUANT_COLUMNS = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

KEEP_COLS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']


def generate_destination(folder, filename):
    """
    returns path where the filename is saved
    """
    return os.path.join(folder, filename)


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        data_frame = pd.read_csv(pth)
        return data_frame
    except FileNotFoundError as err:
        logging.error("File not found: %s %s", err.filename, err.strerror)
        raise err


def perform_eda(data_frame):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    logging.info(data_frame.shape)
    logging.info(data_frame.isnull().sum())
    logging.info(data_frame.describe())

    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig(generate_destination(EDA_IMAGES_FOLDER, 'Churn.png'))

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig(generate_destination(EDA_IMAGES_FOLDER, 'Customer_Age.png'))

    plt.figure(figsize=(20, 10))
    data_frame['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(generate_destination(EDA_IMAGES_FOLDER, 'Marital_Status.png'))

    plt.figure(figsize=(20, 10))
    sns.histplot(data_frame['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(generate_destination(EDA_IMAGES_FOLDER, 'Total_Trans_Ct.png'))

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(numeric_only=True), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(generate_destination(EDA_IMAGES_FOLDER, 'heatmap.png'))


def encoder_helper(data_frame, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    for item in category_lst:
        item_lst = []
        item_groups = data_frame.groupby(item).mean(numeric_only=True)[response]

        for val in data_frame[item]:
            item_lst.append(item_groups.loc[val])

        data_frame[f'{item}_{response}'] = item_lst

    return data_frame


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name
                [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """
    churn_column = df['Churn']
    data_frame = pd.DataFrame()
    category_lst = ['Gender', 'Education_Level',
                    'Marital_Status', 'Income_Category', 'Card_Category']
    df = encoder_helper(df, category_lst, response)
    data_frame[KEEP_COLS] = df[KEEP_COLS]
    return train_test_split(data_frame, churn_column, test_size=0.3, random_state=42)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    plt.rc('figure', figsize=(15, 15))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.savefig(generate_destination(
        RESULT_FOLDER, 'rf_classification_report.jpg'))
    plt.axis('off')

    plt.rc('figure', figsize=(15, 15))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.savefig(generate_destination(
        RESULT_FOLDER, 'lr_classification_report.jpg'))
    plt.axis('off')


def feature_importance_plot(model, x_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(x_train, x_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['log2', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    rfc_model = cv_rfc.best_estimator_
    lr_model = lrc

    joblib.dump(rfc_model, generate_destination('models', 'rfc_model.pkl'))
    joblib.dump(lr_model, generate_destination('models', 'logistic_model.pkl'))

    lrc_plot = RocCurveDisplay.from_estimator(lr_model, x_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    RocCurveDisplay.from_estimator(rfc_model, x_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig(generate_destination('images/results', 'curves'))

    explainer = shap.TreeExplainer(rfc_model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, plot_type="bar")

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, x_train,
                            generate_destination(RESULT_FOLDER, 'feature_importance.png'))


if __name__ == "__main__":
    df = import_data("./data/bank_data.csv")
    logging.info("Import data successfully")
    logging.info("Perform EDA")
    perform_eda(df)
    logging.info("Perform EDA successfully")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, 'Churn')
    logging.info("Perform Feature Engineering successfully")
    train_models(x_train, x_test, y_train, y_test)
    logging.info("Train model successfully")
