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
