from sklearn.datasets import load_iris
iris=load_iris()
X=iris.data
y=iris.target

import pandas as pd
df =pd.DataFrame(X,columns=iris.feature_names)
df['species']=y
print(df.head())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(X_train,y_train)

accuracy =model.score(X_test,y_test)
print(f"accuracy:{accuracy:.2f}")

new_flower = [[5.1, 3.5, 1.4, 0.2]]  # Sample input
predicted_species = model.predict(new_flower)
print(iris.target_names[predicted_species])

