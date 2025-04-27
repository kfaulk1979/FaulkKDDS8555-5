import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ISLP import load_data
from sklearn.linear_model import LogisticRegression as lr
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import classification_report as cr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import GaussianNB

# (a)
# Load the Weekly dataset
Weekly = load_data('Weekly')
print(Weekly.head())

# Numerical summaries
print(Weekly.describe())

# Check orrelation between variables
print(Weekly.select_dtypes(include=['float64','int64']).corr())

# Graphical summaries
sns.pairplot(Weekly, diag_kind="kde")
plt.show()

#Plot Volumne over time
plt.figure(figsize=(12, 6))
plt.plot(Weekly['Volume'])
plt.title('Volume over time')
plt.xlabel('Week')
plt.ylabel('Volume')
plt.show()

# Look at Direction Counts
print(Weekly['Direction'].value_counts())
sns.countplot(x='Direction', data=Weekly)
plt.title('Direction Counts')
plt.show()

# Encode Direction (Up=1, Down=0)
le=LabelEncoder()
Weekly['DirectionBinary']=le.fit_transform(Weekly['Direction'])

# Define Features and Target
X=Weekly[['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']]
y=Weekly['DirectionBinary']

#(b)
# Set up Logistic Regression Model
model = lr(max_iter=1000)

#Set up Repeated Stratified K-Fold Cross-Validation
cv=RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

# Cross-validation scores
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# Report the cross-validation mean and standard deviation
print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Fit the model
mylr=model.fit(X, y)

# Print coefficients and model score
print("Coefficients:", mylr.coef_)
print("Intercept:", mylr.intercept_)
print("Training Accuracy:", mylr.score(X, y))

# Make predictions
y_pred_full = mylr.predict(X)

# Classification report
print(cr(y, y_pred_full))

# (c)
# Confusion matrix
cm = confusion_matrix(y, y_pred_full)
print('Confusion Matrix (Full Data Logistic Regression):\n', cm)

# Overall accuracy
acc_full = accuracy_score(y, y_pred_full)
print('Overall Accuracy:', acc_full)

# (d)
# Logistic Regression: Train (1990-2008) and Test (2009-2010) - Lag2 only

# Split the data into training and testing sets
train = Weekly[Weekly['Year'] <= 2008]
test = Weekly[Weekly['Year'] > 2008]

X_train=train[['Lag2']]
y_train=le.transform(train['Direction'])
X_test=test[['Lag2']]
y_test=le.transform(test['Direction'])

model_lag2=lr(max_iter=1000)
model_lag2.fit(X_train, y_train)
y_pred_lag2 = model_lag2.predict(X_test)

print('\nLag2 Only Logistic Regression Training: 1990-2008, Testing 2009-2010')
# Print coefficients and model score
print('Test Accuracy:', accuracy_score(y_test, y_pred_lag2))
print(cr(y_test, y_pred_lag2))

# (e)
# LDA

lda_model = LDA()
lda_model.fit(X_train, y_train)
y_pred_lda = lda_model.predict(X_test)

print('\nLDA Training: 1990-2008, Testing 2009-2010')
# Print coefficients and model score
print('Test Accuracy:', accuracy_score(y_test, y_pred_lda))
print(cr(y_test, y_pred_lda))

# (f)
# QDA
qda_model = QDA()
qda_model.fit(X_train, y_train)
y_pred_qda = qda_model.predict(X_test)

print('\nQDA Training: 1990-2008, Testing 2009-2010')
# Print coefficients and model score
print('Test Accuracy:', accuracy_score(y_test, y_pred_qda))
print(cr(y_test, y_pred_qda))

# (g)
# KNN (K=1)
knn_model = knn(n_neighbors=1)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('\nKNN (K=1) Training: 1990-2008, Testing 2009-2010')
# Print coefficients and model score
print('Test Accuracy:', accuracy_score(y_test, y_pred_knn))
print(cr(y_test, y_pred_knn))

# (h)
# Naive Bayes
nb_model=GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print('\nNaive Bayes Training: 1990-2008, Testing 2009-2010')
# Print coefficients and model score
print('Test Accuracy:', accuracy_score(y_test, y_pred_nb))
print(cr(y_test, y_pred_nb))

# (i)
# Compare all models
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'LDA', 'QDA', 'KNN (K=1)', 'Naive Bayes'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lag2),
        accuracy_score(y_test, y_pred_lda),
        accuracy_score(y_test, y_pred_qda),
        accuracy_score(y_test, y_pred_knn),
        accuracy_score(y_test, y_pred_nb)
    ]
})

print('\nComparison of Methods:')
print(results)

# (j)
print('\nTrying Lag1 and Lag2 together and KNN with K=3')

X_train2=train[['Lag1','Lag2']]
X_test2=test[['Lag1','Lag2']]


# Logistic Regression with Lag1 and Lag2
model_lags= lr(max_iter=1000)
model_lags.fit(X_train2, y_train)
print('Logistic Regression with Lag1 and Lag2:')
print('Test Accuracy:', accuracy_score(y_test, model_lags.predict(X_test2)))

# KNN with K=3
knn3_model = knn(n_neighbors=3)
knn3_model.fit(X_train2, y_train)

print('KNN (K=3) with Lag1 and Lag2:')
print('Test Accuracy:', accuracy_score(y_test, knn3_model.predict(X_test2)))

