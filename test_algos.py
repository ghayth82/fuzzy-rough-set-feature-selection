from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (BaggingClassifier,BaggingRegressor,
RandomForestClassifier,RandomForestRegressor,
AdaBoostClassifier,AdaBoostRegressor,
GradientBoostingClassifier,GradientBoostingRegressor)

import warnings
warnings.filterwarnings('ignore') 

models = [
    DecisionTreeClassifier(max_depth=3),
    KNeighborsClassifier(n_neighbors=5),
    SVC(C=1.0),
    GaussianNB(),
    LogisticRegression(),
    BaggingClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier()
]

def test_model(data,target,model):
    train_x,test_x,train_y,test_y = train_test_split(data,target,test_size=0.2,random_state=0)
    model.fit(train_x,train_y)
    predictions = model.predict(test_x)
    accuracy = accuracy_score(predictions,test_y)
    # f1score = f1_score(predictions,test_y)
    # precisionscore = precision_score(predictions,test_y)
    # recallscore = recall_score(predictions,test_y)

    print("Model : {:30s} , Accuracy : {:.2f}".format(type(model).__name__,accuracy))
