import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Multi-class Prediction of Obesity Risk/train.csv")
test = pd.read_csv("/Users/kevinfaulk/Documents/DDS-8555/Multi-class Prediction of Obesity Risk/test.csv")

# Encode categorical variables
label_encoders = {}
for col in train.select_dtypes(include=['object']).columns:
    if col != 'NObeyesdad':  # Leave the target for later encoding
        # Combine train and test values for this column
        combined = pd.concat([train[col], test[col]], axis=0)
        le = LabelEncoder().fit(combined)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])
        label_encoders[col] = le


# Encode the target variable
target_le = LabelEncoder()
y=target_le.fit_transform(train['NObeyesdad'])
X = train.drop(columns=['NObeyesdad'])

# Save list of features to apply to test data
X_test = test[X.columns]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Split set from training for validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Multinomial Logistic Regression
log_model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
log_model.fit(X_train, y_train)
val_preds_log=log_model.predict(X_val)

print("\nLogistic REgression Validation Performance:")
print(classification_report(y_val, val_preds_log))

# LDA
lda_model = LDA()
lda_model.fit(X_train, y_train)
val_preds_lda = lda_model.predict(X_val)

print("\nLDA Validation Performance:")
print(classification_report(y_val, val_preds_lda))

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
val_preds_nb = nb_model.predict(X_val)

print("\nNaive Bayes Validation Performance:")
print(classification_report(y_val, val_preds_nb))

# SVM
svm_model = SVC(kernel='rbf', probability=True)
svm_model.fit(X_train, y_train)
val_preds_svm = svm_model.predict(X_val)

print("\nSVM Validation Performance:")
print(classification_report(y_val, val_preds_svm))

# Confusion Matrix
cm=confusion_matrix(y_val, val_preds_log)
cm_df=pd.DataFrame(cm, index=target_le.classes_, columns=target_le.classes_)
print("\nConfusion Matrix for Logistic Validation:")

# Correlation Matrix to Check Assumptions
train_corr=pd.DataFrame(X_scaled, columns=X.columns)
plt.figure(figsize=(12, 10))
sns.heatmap(train_corr.corr(), cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Features')
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight') # Save as PNG
plt.show()

# Full Training Data
log_model.fit(X_scaled, y)
lda_model.fit(X_scaled, y)
nb_model.fit(X_scaled, y)
svm_model.fit(X_scaled, y)

# Predict Kaggle Test Data
log_preds_test = log_model.predict(X_test_scaled)
lda_preds_test = lda_model.predict(X_test_scaled)
nb_preds_test = nb_model.predict(X_test_scaled)
svm_preds_test = svm_model.predict(X_test_scaled)

# Save predictions to CSV
def save_submission(preds, filename):
    preds_labels= target_le.inverse_transform(preds)
    submission = pd.DataFrame({'id': test['id'], 'NObeyesdad': preds_labels})
    submission.to_csv(filename, index=False)

save_submission(log_preds_test, 'logistic_regression_submission.csv')
save_submission(lda_preds_test, 'lda_submission.csv')
save_submission(nb_preds_test, 'naive_bayes_submission.csv')
save_submission(svm_preds_test, 'svm_submission.csv')

print("Submissions saved.")

