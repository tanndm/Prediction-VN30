import pandas as pd
import numpy as np
#####################
import pickle
import os
import joblib
#####################
import matplotlib.pyplot as plt
import seaborn as sns
#####################
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


data = pd.read_csv(r"https://raw.githubusercontent.com/BrianNguyen2001/Prediction-VN30/main/cleaned_data_5.csv",
                    index_col=0,
                    parse_dates=True)
                    
data.index = data.index.astype("datetime64[ns]")
data.index = data.index.set_names("Time")

data = data[['bid_quality', 'bid_volume', 'ask_quality', 'ask_volume',
       'matching_volume', 'negotiable_volume', 'SMA10', 'SMA20', 'EMA_10',
       'EMA_20', 'RSI_7d', 'RSI_9d','RSI_14d', 'Positive', 'Negative', 'return_vn30']]

# News
df = data.copy() 

# Set the independent variable property
feature = df.columns[df.columns!='return_vn30'].to_list()
target = [df.columns[-1]]

y = df[target]
X = df[feature]

# Split the train and test
n_state = 1745
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state=n_state)

# Call scale function
sc_X = StandardScaler()
# Scale train set
X_train = sc_X.fit_transform(X_train)
# Scale test set
X_test = sc_X.fit_transform(X_test)

# train SVM model
model = SVC(kernel = 'rbf', probability=True, random_state = n_state)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_pred = pd.DataFrame(y_pred,index=y_test.index,columns=['SVM-RBF Prediction'])

joblib.dump(model, open(os.path.join(os.path.dirname(__file__),"../../model_svm.pkl"), "wb"), 9)
