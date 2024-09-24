import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance

# Load the pickled model
@st.cache_resource
def load_model():
    with open('fraud_detection_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Define the required columns
REQUIRED_COLUMNS = ['annual_income', 'loan_amount', 'term', 'interest_rate', 'installment', 
                    'debt_to_income', 'inquiries_last_12m', 'account_never_delinq_percent', 
                    'total_credit_utilized', 'num_historical_failed_to_pay', 
                    'public_record_bankrupt', 'open_credit_lines']

st.title('Loan Fraud Detection')

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Check if all required columns are present
    missing_columns = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing_columns:
        st.warning(f"Missing columns: {', '.join(missing_columns)}. Please ensure your CSV contains all required columns.")
    else:
        # Make predictions
        predictions = model.predict(data[REQUIRED_COLUMNS])
        probabilities = model.predict_proba(data[REQUIRED_COLUMNS])[:, 1]
        
        # Add predictions to the dataframe
        data['Predicted_Fraud'] = predictions
        data['Fraud_Probability'] = probabilities
        
        # Display results
        st.subheader('Prediction Results')
        st.write(data)
        
        # Download link for results
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="fraud_detection_results.csv",
            mime="text/csv",
        )

        # Sort the dataframe by Fraud_Probability in descending order
        data_sorted = data.sort_values('Fraud_Probability', ascending=False)

        # Display results
        st.subheader('Prediction Results (Sorted by Fraud Probability)')

        # Add a slider to control how many rows to display
        num_rows = st.slider('Number of rows to display', min_value=1, max_value=len(data_sorted), value=20)

        # Display the sorted data
        st.write(data_sorted.head(num_rows))

        # Add checkboxes to filter results
        col1, col2 = st.columns(2)
        with col1:
            show_fraudulent = st.checkbox('Show only predicted fraudulent loans', value=False)
        with col2:
            probability_threshold = st.slider('Minimum fraud probability to display', min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        # Filter based on user selection
        if show_fraudulent:
            filtered_data = data_sorted[data_sorted['Predicted_Fraud'] == 1]
        else:
            filtered_data = data_sorted[data_sorted['Fraud_Probability'] >= probability_threshold]

        # Display filtered results
        st.subheader('Filtered Results')
        st.write(filtered_data)

        # Download link for filtered results
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download filtered results as CSV",
            data=csv,
            file_name="filtered_fraud_detection_results.csv",
            mime="text/csv",
        )

        # Display summary statistics
        st.subheader('Summary Statistics')
        st.write(f"Total number of loans: {len(data)}")
        st.write(f"Number of predicted fraudulent loans: {data['Predicted_Fraud'].sum()}")
        st.write(f"Average fraud probability: {data['Fraud_Probability'].mean():.2%}")
        
        # Visualizations
        st.subheader('Visualizations')
        
        # Confusion Matrix
        
        if 'is_fraud' in data.columns:
            st.subheader('Confusion Matrix')
            st.write("""
            The confusion matrix shows the performance of our model in classifying loans:
            - True Negatives (top-left): Correctly identified non-fraudulent loans
            - False Positives (top-right): Non-fraudulent loans incorrectly flagged as fraudulent
            - False Negatives (bottom-left): Fraudulent loans missed by the model
            - True Positives (bottom-right): Correctly identified fraudulent loans
            A good model maximizes true positives and true negatives while minimizing false positives and false negatives.
            """)
            cm = confusion_matrix(data['is_fraud'], predictions)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax)
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            st.pyplot(fig)
        
            # ROC Curve
            st.subheader('ROC Curve')
            st.write("""
            The Receiver Operating Characteristic (ROC) curve illustrates the model's ability to distinguish between classes:
            - The x-axis shows the False Positive Rate (1 - Specificity)
            - The y-axis shows the True Positive Rate (Sensitivity)
            - The Area Under the Curve (AUC) indicates overall performance: 1.0 is perfect, 0.5 is no better than random
            A curve closer to the top-left corner indicates better performance.
            """)
            fpr, tpr, _ = roc_curve(data['is_fraud'], probabilities)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            st.pyplot(fig)

            # Add predictions to the dataframe
            data['Predicted_Fraud'] = predictions
            data['Fraud_Probability'] = probabilities

            # Create loan_category based on predictions
            data['loan_category'] = data['Predicted_Fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})

            st.subheader('Loan Amount vs Interest Rate by Predicted Fraud Status')
            fig = px.scatter(data, x='loan_amount', y='interest_rate', color='Predicted_Fraud',
                            color_discrete_map={0: 'blue', 1: 'red'},
                            hover_data=['annual_income', 'debt_to_income'],
                            labels={'Predicted_Fraud': 'Predicted Status'},
                            title='Loan Amount vs Interest Rate by Predicted Fraud Status')

            fig.update_layout(legend_title_text='Predicted Fraud Status',
                            xaxis_title='Loan Amount',
                            yaxis_title='Interest Rate')

            st.plotly_chart(fig, use_container_width=True)

            st.write("This interactive scatter plot visualizes the relationship between loan amount and interest rate, color-coded by predicted fraud status. You can zoom, pan, and hover over points for more information. Look for patterns or clusters that might indicate fraudulent activity.")
        
        # Feature Importance
        st.subheader('Feature Importance')
        #st.write(f"Model type: {type(model)}")

        # Perform permutation importance
        perm_importance = permutation_importance(model, data[REQUIRED_COLUMNS], data['Predicted_Fraud'], n_repeats=10, random_state=42)

        # Create dataframe of feature importances
        feature_importance = pd.DataFrame({
            'feature': REQUIRED_COLUMNS,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)

        # Create plot
        fig = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                    title='Feature Importance',
                    labels={'importance': 'Importance', 'feature': 'Feature'},
                    color='importance',
                    color_continuous_scale='Viridis')

        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

        st.write("""
        This chart shows the relative importance of each feature based on permutation importance.
        Features with higher importance have a greater influence on the model's predictions.
        Understanding feature importance can help in focusing on the most critical factors for fraud detection.
        """)
        
        # Anomaly Detection
        st.subheader('Anomaly Detection')
        for feature in REQUIRED_COLUMNS:
            mean = data_sorted[feature].mean()
            std = data_sorted[feature].std()
            anomalies = data_sorted[data_sorted[feature] > mean + 3*std]
            if not anomalies.empty:
                st.write(f"Anomalies in {feature}:")
                st.write(anomalies[[feature, 'Predicted_Fraud', 'Fraud_Probability']])

# Instructions
st.sidebar.header('Instructions')
st.sidebar.write('1. Upload a CSV file containing loan data.')
st.sidebar.write('2. Ensure your CSV contains all required columns.')
st.sidebar.write('3. View the predictions and visualizations.')
st.sidebar.write('4. Download the results if needed.')