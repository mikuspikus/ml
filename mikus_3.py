import pandas as pd
import numpy as np
data = pd.read_csv('F:\Datasets\modern_teen_girls.csv')

def macross_val_score(estimator, X, y, cv):
    result_accracy_list = []
    for train_index, test_index in cv:
        estimator.fit(X.iloc[train_index], y.iloc[train_index])
        result_accracy_list.append(np.average(np.array(estimator.predict(X.iloc[test_index]) == np.array(y.iloc[test_index]))))
    return pd.Series(result_accracy_list)
    
from sklearn.cross_validation import KFold
kfold = KFold(n = len(data), n_folds = 5, shuffle = True, random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score

accuracies_dict = {}
for k in range(1, 50 + 1):
    classifier = KNeighborsClassifier(n_neighbors = k)
    #scores = cross_val_score(classifier, data.drop(['Class'], axis = 1), data['Class'], scoring = 'accuracy', cv = kfold)
    scores = macross_val_score(estimator = classifier, X = data.drop(['Class'], axis = 1), y = data['Class'], cv = kfold)
    accuracies_dict[k] = scores.mean()

optimal_k = sorted(accuracies_dict, key = accuracies_dict.get, reverse = True)[0]
print('Most optimal k: (', optimal_k, ', %.4f' %accuracies_dict[k], ')')

from sklearn.preprocessing import scale
scaled_data = scale(data.drop(['Class'], axis = 1))

scaled_accuracies_dict = {}
for k in range(1, 50 + 1):
    classifier = KNeighborsClassifier(n_neighbors=k)
    #scores = cross_val_score(classifier, scaled_data, data['Class'], scoring = 'accuracy', cv = kfold)
    scores = macross_val_score(estimator = classifier, X = pd.DataFrame(scaled_data), y = data['Class'], cv = kfold)
    scaled_accuracies_dict[k] = scores.mean()
    
optimal_k = sorted(scaled_accuracies_dict, key = scaled_accuracies_dict.get, reverse = True)[0]
print('Most optimal k ater scaling: (', optimal_k, ', %.4f' %scaled_accuracies_dict[k], ')')