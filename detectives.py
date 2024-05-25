[12:24] Genc, Vural
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import numpy as np
 
# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names
 
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
# Train the Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
 
# Streamlit app
st.title('Iris Flower Classifier')
 
# Sidebar for input sliders
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))
 
# Predict the class of the input features
input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
prediction = clf.predict(input_features)
prediction_proba = clf.predict_proba(input_features)
 
st.write(f"Predicted class: {class_names[prediction[0]]}")
st.write(f"Prediction probabilities: {prediction_proba}")
 
# Evaluate the model on the test set
y_pred = clf.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = clf.score(X_test, y_test)
class_report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
 
# Calculate sensitivity and specificity
sensitivity = {class_names[i]: class_report[class_names[i]]['recall'] for i in range(len(class_names))}
specificity = {}
for i, class_name in enumerate(class_names):
   tn = conf_matrix.sum() - (conf_matrix[:, i].sum() + conf_matrix[i, :].sum() - conf_matrix[i, i])
   fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
   specificity[class_name] = tn / (tn + fp)
 
# Display performance metrics
st.subheader('Model Performance')
st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Sensitivity (Recall):")
st.write(sensitivity)
st.write("Specificity:")
st.write(specificity)
