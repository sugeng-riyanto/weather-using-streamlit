import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.title("Weather Data Logistic Regression Analysis")

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv"])
if uploaded_file is not None:
    # Load the data
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("### Weather Data")
    st.write(df)

    # Basic data exploration
    st.write("### Data Summary")
    st.write(df.describe())

    # Handle missing values
    df = df.dropna()

    # Identify and encode categorical features
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = LabelEncoder().fit_transform(df[column])

    # Ensure all columns are numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # Drop any remaining non-numeric columns
    df = df.dropna(axis=1, how='any')

    # Define features (X) and target (y)
    # Assuming the last column is the target variable and the rest are features
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)  # Slider to adjust test size
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.write(f"### Model Evaluation")
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1-Score: {fscore}")

    # Display confusion matrix
    st.write("### Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # ROC curve
    st.write("### ROC Curve")
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Precision-Recall curve
    st.write("### Precision-Recall Curve")
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker='.', label='Logistic')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    st.pyplot(fig)
else:
    st.write("Please upload a CSV file.")
