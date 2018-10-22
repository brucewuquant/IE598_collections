
# coding: utf-8

# MLF_GP2_EconCycle contains 223 monthly observations of the US Treasury bond yield curve, the commercial paper yield curve and the USPHCI Economic Activity Index.  In 1993 Friedman and Kuttner published a paper entitled "Why does the Paper-Bill spread predict real economic activity?"  They found evidence that the spread on commercial paper, a short term form of corporate borrowing, and the US Treasury bill widens before recessions and contracts after and could be a useful predictor of real economic activity.  We will not rigorously replicate this study, but instead use it as the basis for a regression exercise in machine learning.  (Although this is a time series dataset, we will not be using it as such. You should discard the Date column and treat each row as an independent observation.  You should also standardize all your data, otherwise your model will overfit to the level, since USHPCI is rising throughout this period.  You will not include the Index itself in your model.) Using machine learning regression techniques, produce a model that uses these features and any additional features you engineer (use your imagination) to predict the percent change in the USHPCI 3, 6 and 9 months ahead.  
# 
# Each report should have two Chapters; first one of the CreditSCore problem and then one on the EconomicCycle problem.
# 
# Each Chapter should have 6 subsections addressing the following topics: 
# 
# 1) Introduction/Exploratory Data Analysis, 
# 2) Preprocessing, feature extraction, feature selection, 
# 3) Model fitting and evaluation, (you should fit at least 3 different machine learning models)
# 4) Hyperparameter tuning, 
# 5) Ensembling and 
# 6) Conclusions
# 
# The report should have an Appendix with links to the code repository.  The report itself may include charts, graphs and tables, but should NOT include unformatted 'cut-and-pasted' output from Python. 
# 
# The completed project will also have a cover page which lists the full names of all of the team members, page numbers, headings and subheadings.
# 
# It will consist of a single pdf file labelled "IE598MLF_Group_project" and be uploaded through the Compass assignment submission page.
# 
# This project will count 25% of your course grade.

# In[1]:


#Import necessary modules
import warnings
import sklearn
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
from sklearn.linear_model import LinearRegression
import pandas as pd
import re
import copy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet


# 1) Introduction/Exploratory Data Analysis

# In[2]:


econData = pd.read_csv(r'C:\Users\wdz\Desktop\UIUC课程\IE598\CourseModules\Module7GroupProject\MLF_GP2_EconCycle.csv', index_col  = None)
#Discard the Date column
econData = econData.iloc[:,1:]
#Discard all index
#print(str(econData.columns))
#searchObj = re.findall(r'\w\d+\w\s'+'Index', str(econData.columns))
searchObj = re.findall(r'USPHCI', str(econData.columns))
econData.drop(columns = searchObj, inplace = True)
print(econData.head())
print(f'shape of EconCyle Data is  {econData.shape}')
summary = econData.describe()
#econData.isnull().any().any()
print(summary)
summary.to_excel('econ.summary.xlsx')


# In[3]:


#boxplot
def make_boxplot(df, cols):
    
    #standardize data without any change to the original DataFrame
    tmp1 = copy.deepcopy(df)
    #vectorization way
    normalized_df=(tmp1- tmp1.mean())/tmp1.std()
    #if we instead want to use min-max: normalized_df=(df-df.min())/(df.max()-df.min())
    sns.boxplot(data =normalized_df, orient = 'h') #orient for better display
    plt.show()

make_boxplot(econData, econData.columns)


# In[4]:


def make_scatter(df, cols):
    sns.pairplot(df[cols], size = 2.5)
    plt.show()

make_scatter(econData, econData.columns)


# In[5]:


#Heatmap
def make_heatmap(df, cols):
    corMat = df[cols].corr()
    sns.heatmap(corMat, cbar = True, annot = False, square = True, fmt ='.2f',
                annot_kws = {'size' : 10}, yticklabels = cols, xticklabels = cols)
    plt.show()

print(econData.columns)
make_heatmap(econData, econData.columns)


# In[6]:


#preprocesing 
X = econData.iloc[:,:12].values
#predicting PCT 3MO FMD
y = econData.iloc[:, 14].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)

sc  = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test) # no need to fit the test data


# In[7]:


class ModelEval(object): 
    def __init__(self, regressor, X_train, X_test, y_train, y_test):
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        self.regressor = regressor.fit(X_train, y_train)
        self.y_train_pred = (regressor.predict(X_train)).reshape((-1,1))
        self.y_test_pred = (regressor.predict(X_test)).reshape((-1,1))
    
    def model_result(self, title = None):
        if type(self.regressor) == sklearn.pipeline.Pipeline or self.regressor != r'Linear+':
            pass
        else:  
            print('coef:', self.regressor.coef_)
            print('intercept:', self.regressor.intercept_)
        print('MSE train: %.5f, test: %.5f' %(mean_squared_error(y_train,self.y_train_pred)
    , mean_squared_error(y_test, self.y_test_pred)))

        print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, self.y_train_pred)
    ,r2_score(y_test, self.y_test_pred)))
        plt.scatter(self.y_train_pred,  self.y_train_pred - y_train,
                c='steelblue', marker='o', edgecolor='white',
                label='Training data')
        plt.scatter(self.y_test_pred,  self.y_test_pred - y_test,
                c='limegreen', marker='s', edgecolor='white',
                label='Test data')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title = title
        plt.legend(loc='upper left')
        plt.hlines(y=0, xmin= min(self.y_test_pred) -0.005 , xmax= max(self.y_test_pred)+0.005, color='black', lw=2)
        plt.xlim([min(self.y_test_pred) -0.005, max(self.y_test_pred)+0.005])
        plt.tight_layout()
        plt.show()
        


# In[8]:


class HyperTuning(object):
    def __init__(self, estimator, param_grid, scoring, 
                 X_train, y_train, cv = 10):
        
        self.param_grid = param_grid
        self.scoring = scoring
        self.X_train = X_train
        self.y_train = y_train
        self.estimator = estimator.fit(X_train, y_train)
        
    def gs_tune(self):
        gs = GridSearchCV(estimator = self.estimator, param_grid = self.param_grid,
                         scoring = self.scoring, cv = 10, n_jobs = 4, verbose = 0)
        gs.fit(X_train, y_train)
        print(f'best param is {gs.best_params_}, best score is {gs.best_score_}')
        return gs.best_estimator_
           


# In[9]:


#Regular linear
linreg_eval = ModelEval(LinearRegression(), X_train_std, X_test_std, y_train, y_test)
linreg_eval.model_result(title = 'Linear Regression' )


# In[10]:


#LinReg with pca
pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)

plt.bar(range(0,12), pca.explained_variance_ratio_, alpha = 0.5, align = 'center')
plt.step(range(0,12), np.cumsum(pca.explained_variance_ratio_), where = 'mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.show()

linreg_pca = Pipeline(steps = [('pca', pca), ('linear', LinearRegression())])
param_grid = {
    'pca__n_components': [2,4,6,8,10]
}
linreg_pca_tuning = HyperTuning(linreg_pca, param_grid, 'r2', X_train_std, y_train,  cv =10)
linreg_pca_best = linreg_pca_tuning.gs_tune()
linreg_pca_eval = ModelEval(linreg_pca_best, X_train_std, X_test_std, y_train, y_test)
linreg_pca_eval.model_result(title = 'Linear Regression with PCA' )


# In[11]:


#LinReg with kpca
kpca = KernelPCA()
linreg_kpca = Pipeline(steps = [('kpca', kpca), ('linear', LinearRegression())])
param_grid = {
    'kpca__n_components':[2,4,6,8,10],
    'kpca__kernel': [ 'linear', 'poly', 'rbf', 'sigmoid'],
    'kpca__gamma': [ 0.001, 0.01, 0.0829]
    
}
linreg_kpca_tuning = HyperTuning(linreg_kpca, param_grid, 'r2', X_train_std, y_train,  cv =10)
linreg_kpca_best = linreg_kpca_tuning.gs_tune()
linreg_kpca_eval = ModelEval(linreg_kpca_best, X_train_std, X_test_std, y_train, y_test)
linreg_kpca_eval.model_result(title = 'Linear Regression with KPCA' )


# In[ ]:


#  LR with Elastic Net
# elastic = ElasticNet()
# elastic.get_params().keys()
elastic = ElasticNet()
param_space_ele = np.arange(0.01,1,0.01).tolist()
param_grid = {
    'alpha' : param_space_ele,
    'l1_ratio': param_space_ele
}
def tuning_and_res(model, param_grid):
    
    model_tuning = HyperTuning(model, param_grid, 'r2', X_train_std, y_train,  cv =10)
    model_best = model_tuning.gs_tune()
    model_eval = ModelEval(model_best, X_train_std, X_test_std, y_train, y_test)
    model_eval.model_result(title = f'Linear Regression with {model}' )

tuning_and_res(elastic, param_grid)


# In[ ]:


import tensorflow as tf
import tensorflow.contrib.keras as keras
np.random.seed(123)
tf.set_random_seed(123)


# In[ ]:


model_nn = keras.models.Sequential()
model_nn.add(
    keras.layers.Dense(
    units=50,
    input_dim=X_train_std.shape[1],
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='tanh'))

model_nn.add(
    keras.layers.Dense(
    units=50,
    input_dim=50,
    kernel_initializer='glorot_uniform',
    bias_initializer='zeros',
    activation='tanh'))

model_nn.add(
    keras.layers.Dense(1, activation = 'relu'))
model_nn.compile(loss='mean_squared_error', optimizer='sgd')
model_nn.fit(X_train_std, y_train, epochs=1000, verbose=0)
y_train_pred_nn = model_nn.predict(X_train_std)
y_test_pred_nn = model_nn.predict(X_test_std)

print('MSE train: %.5f, test: %.5f' %(mean_squared_error(y_train, y_train_pred_nn)
    , mean_squared_error(y_test, y_test_pred_nn)))

print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_train_pred_nn)
    ,r2_score(y_test, y_test_pred_nn)))


# In[ ]:


#Ensemble using EXtreme Gradient Boosting
xgb_baseline = xgb.XGBRegressor( learning_rate = 0.1, n_estimators=100, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=1,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed = 42, return_train_score = True,
                         scoring='r2',n_jobs=4,iid=False)
xgb_baseline.fit(X_train_std,y_train)
y_pred = xgb_baseline.predict(X_test_std)
print(f'MSE for untuned xgboost is {mean_squared_error(y_test, y_pred)}')
print(f'R-squared for untuned xgboost is {r2_score(y_test,y_pred)}')


# In[ ]:


param_test1 = {
 'max_depth':range(3,17,1),
 'min_child_weight':range(1,13,1)
}

xgb_init = xgb.XGBRegressor( learning_rate = 0.1, n_estimators=100, gamma=0, subsample=0.8, colsample_bytree=1,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed = 42, return_train_score = True,
                         scoring='r2',n_jobs=4,iid=False)

# n_iter_search = 30
xgb_rs = RandomizedSearchCV(xgb_init, param_distributions=param_test1, cv=10)
xgb_rs.fit(X_train_std, y_train)

y_pred_rand = xgb_rs.best_estimator_.predict(X_test_std)
print(f'MSE for tuned xgboost using RandomSearch is {mean_squared_error(y_test, y_pred_rand)}')
print(f'R-squared for tuned xgboost using RandomSearch is {r2_score(y_test,y_pred_rand)}')

tuning_and_res(xgb_init, param_test1)


# In[ ]:


xgb_trial = xgb.XGBRegressor( learning_rate = 0.1, n_estimators=100, max_depth=5,
 min_child_weight=10, gamma=0, subsample=0.8, colsample_bytree=1,
 objective= 'reg:linear', nthread=4, scale_pos_weight=1, seed = 42, return_train_score = True,
                         scoring='r2',n_jobs=4,iid=False)

xgb_trial.fit(X_train_std,y_train)
y_pred_trial = xgb_baseline.predict(X_test_std)
print(f'MSE for untuned xgboost is {mean_squared_error(y_test, y_pred_trial)}')
print(f'R-squared for untuned xgboost is {r2_score(y_test,y_pred_trial)}')


# Code to Report

# In[ ]:


print("Features,Mean,Median,Standard Deviation,Skewness,Kurtosis")
class CodetoReport(object):
    def __init__(self):
        pass
    def statistics_eda(df, cols):
        for i in range(0, len(cols)):
            column = df[df.columns[i]]
            row = [df.columns[i], str(column.mean()), str(column.median()), str(column.std()), str(column.skew()), str(column.kurt())]
            print(",".join(row))
CodetoReport.statistics_eda(econData, econData.columns)

