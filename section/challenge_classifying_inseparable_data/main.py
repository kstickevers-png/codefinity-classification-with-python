import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

df = pd.read_csv('https://codefinity-content-media.s3.eu-west-1.amazonaws.com/b71ff7ac-3932-41d2-a4d8-060e24b00129/circles.csv')
X = df[['X1', 'X2']]
y = df['y']

# Write your code below
pipe = Pipeline([('poly', PolynomialFeatures(degree=2,include_bias=False)),
                 ('scaler', StandardScaler())])
X_poly = pipe.fit_transform(X)

lr = LogisticRegression()
param_grid = {'C':[0.01,0.1,1,10,100]}
grid_cv = GridSearchCV(lr,param_grid,cv=5).fit(X_poly,y)

best_score = grid_cv.best_score_
best_model = grid_cv.best_estimator_

# Testing the result
print(best_score)
print(best_model)