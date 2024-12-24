import pandas as pd
import numpy as np 
from matplotlib import pyplot

import xlrd
uploaded =("C:\\Users\\ELCOT\\Downloads\\Breast Cancer Detection\\Breast Cancer Detection")

"""### *Load Dataset*"""

dataset = pd.read_csv('data.csv')

"""### *Summarize Dataset*"""

print(dataset.shape)
print(dataset.head(5))

"""### *Mapping Class String Values to Numbers*"""

dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M': 1}).astype(int)
print(dataset.head)

"""### *Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)*"""

X = dataset.iloc[:, 2:32].values
Y = dataset.iloc[:,1].values

"""### *Splitting Dataset into Train & Test*"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

"""### *Feature Scaling*
### we scale our data to make all the features contribute equally to the result
###Fit_Transform - fit method is calculating the mean and variance of each of the features present in our data
###Transform - Transform method is transforming all the features using the respective mean and variance, 
###We want our test data to be a completely new and a surprise set for our model
"""

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)

"""### *Validating some ML algorithm by its accuracy - Model Score*"""

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
res = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=None)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print('%s: %f' % (name, cv_results.mean()))

pyplot.ylim(.900, .999)
pyplot.bar(names, res, color ='maroon', width = 0.6)

pyplot.title('Algorithm Comparison')
pyplot.show()

"""### *Training & Prediction using the algorithm with high accuracy*"""

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train, y_train)
value=[[12.46,24.04,83.97,475.9,0.1186,0.2396,0.2273,0.08543,0.203,0.08243,0.2976,1.599,2.039,23.94,0.007149,0.07217,0.07743,0.01432,0.01789,0.01008,15.09,40.68,97.65,711.4,0.1853,1.058,1.105,0.221,0.4366,0.2075]]
y_pred = model.predict(value)
print(y_pred)
if y_pred == 1:
    print("patient have breast cancer")
else:
    print("patient is normal")
