import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('F:/Datasets/titanic.csv', index_col = 'PassengerId')

data['SexXx'] = data['Sex'].map( {'male' : 1, 'female' : 0} ).astype(int)
predictors = ['Pclass', 'Fare', 'Age', 'SexXx']
target = 'Survived'

train = data[['Pclass', 'Fare', 'Age', 'SexXx', 'Survived']].dropna()

clf = DecisionTreeClassifier(random_state = 241)
clf.fit(train[predictors], train[target])

importances = clf.feature_importances_

result = list(zip(importances, predictors))
print(sorted(result, reverse = True))