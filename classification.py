import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
from keras.metrics import F1Score, Recall, Precision
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

df = pd.read_csv('train.csv')

df['ID'] = df['ID'].apply(lambda x: int(x, 16))

df['Name'] = df['Name'].str.strip().str.lower()
df['Occupation'] = df['Occupation'].str.strip().str.lower()

df = pd.get_dummies(df, columns=['Customer_ID', 'Month', 'Occupation', 'Type_of_Loan', 'Payment_of_Min_Amount'], drop_first=True)

for column in df.select_dtypes(include=[np.number]).columns:
    df[column].fillna(df[column].mean(), inplace=True)

for column in df.select_dtypes(include=['object']).columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

X = df.drop(columns=['Credit_Score'])  
y = df['Credit_Score'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim = df.shape[1] - 1
num_classes = df['Credit_Score'].nunique()

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(input_dim,)))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=CategoricalCrossentropy(), optimizer=Adam(), metrics=[F1Score(), Recall(), Precision()])
model.fit(X_train, y_train, epochs=10, batch_size=32)
model.save('model.h5')
