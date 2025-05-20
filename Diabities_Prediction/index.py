import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df=pd.read_csv('Diabities_Prediction/diabities.csv')
X=df.drop('Outcome',axis=1)
y=df['Outcome']

X.fillna(X.mean(),inplace=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scaler=StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)

model =LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)


y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

