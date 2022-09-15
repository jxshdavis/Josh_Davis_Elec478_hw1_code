from pyexpat import features
import pandas as pd
from pandas import read_csv
import numpy as np # linear algebra
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn import datasets, linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import itertools
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from scipy import stats

# ########################################################################

# #                      VISUALIZE THE DATA

# fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
# index = 0
# axs = axs.flatten()
# log_data = data.copy()
# log_data["MEDV"] = y
# for k,v in log_data.items():
#     sns.boxplot(y=k, data=log_data, ax=axs[index])
#     index += 1
# plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
# plt.show()

# ########################################################################




#1.A.i.1 LR OLS
 
def feature_Analysis_LinearRegression(random_state):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)


    # fit the linear model
    lm = sm.OLS(y_train, X_train)
    fii = lm.fit()
    p_values = fii.summary2().tables[1]['P>|t|']
    print(p_values)


#1.A.i.2 BEST SUBSETS

def processSubset(feature_set, y_train, X_train):
    # Fit model on feature_set and calculate RSS

    X = X_train.copy()
    y = y_train.copy()
    
    feature_set = ("Intercept",)+feature_set
    # print()
    # print("feature set:",str(feature_set))
    # print()

    # print(X)
    X.insert(0, 'Intercept', 1)

    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}


def getBest(k, y_train, X_train):
    
    results = []
    
    for combo in itertools.combinations(X_train.columns, k):
        results.append(processSubset(combo, y_train, X_train))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    
    print("Processed", models.shape[0], "models on", k)
    
    # Return the best model, along with some other useful information about the model
    print(best_model["model"].summary())


def feature_best_subsets(random_state = 0):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
     #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)

    #standerdize the data
    list_numerical = X_train.columns
    scaler_train = StandardScaler().fit(X_train[list_numerical])
    scalar_test = StandardScaler().fit(X_test[list_numerical])

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scalar_test.transform(X_test[list_numerical])


    for k in range (1,8):
        print("results for k = ",str(k))
        getBest(k, y_train, X_train)

#1.A.i.3 RFE

def feature_Analysis_RFE(random_state = 0):
     # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
     #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)
    
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=4, step=1)
    # selector = RFE(estimator, n_features_to_select=5, step=1)
    selector = selector.fit(X_train, y_train)
    
    print(selector.get_support())
    print(selector.get_feature_names_out())

#1.A.i.4 LASSO

def feature_Analysis_LASSO(random_state=42):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)

    #standerdize the data
    list_numerical = X_train.columns
    scaler_train = StandardScaler().fit(X_train[list_numerical])
    scalar_test = StandardScaler().fit(X_test[list_numerical])

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scalar_test.transform(X_test[list_numerical])

    lambas = [.01,.1,.2]
    for alpha in lambas:
        model = linear_model.Lasso(alpha = alpha, fit_intercept= True)
        model.fit(X_train, y_train)
        print('LASSO WHEN LAMDA is:', str(alpha))
        features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

        sorted_pairs = sorted(list(zip(model.coef_,features)),key = lambda x: x[0])
        for pair  in sorted_pairs:
            print(pair[1]) 
        for pair  in sorted_pairs:
            print(round(pair[0], 5)) 
        
         

#1.A.i.5 ELASTIC NET

def feature_Analysis_ENET(random_state=42):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)


    #standerdize the data
    list_numerical = X_train.columns
    scaler_train = StandardScaler().fit(X_train[list_numerical])
   

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])

    lambas = [.01,.1,0.2]
    for alpha in lambas:

        RATIO = 0.5

        model = linear_model.ElasticNet(alpha = alpha, l1_ratio=RATIO, fit_intercept= True, )
        model.fit(X_train, y_train)
        print('Elastic Net WHEN LAMDA is:', str(alpha))
        print("Ratio is:", str(RATIO))
        features = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        sorted_pairs = sorted(list(zip(model.coef_,features)),key = lambda x: x[0])
        for pair  in sorted_pairs:
            print(pair[1]) 
        for pair  in sorted_pairs:
            print(round(pair[0], 5)) 
#1.A.i.6 Adaptive Lasso
def feature_Analysis_ALASSO(random_state=42):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)

    #standerdize the data
    list_numerical = X_train.columns
    scaler_train = StandardScaler().fit(X_train[list_numerical])
   

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])

   

    lambas = [.1]
    for alpha in lambas:

        g = lambda w: np.sqrt(np.abs(w))
        gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

        n_samples, n_features = X_train.shape
       

        weights = np.ones(n_features)
        n_lasso_iterations = 4

        for k in range(n_lasso_iterations):
            X_w = X_train / weights[np.newaxis, :]
            model = Lasso(alpha=alpha, fit_intercept=True)
            model.fit(X_w, y_train)
            coef_ = model.coef_ / weights
            weights = gprime(coef_)
            print("ITERATION:",str(k))


            sorted_pairs = sorted(list(zip(coef_,X_train.columns.tolist())),key = lambda x: x[0])
            for pair  in (sorted_pairs):
                print(pair[1])
            for pair  in (sorted_pairs):
                print(round(pair[0], 6))
            
        
    

# Question 1, Part a, Section i.

#TODO: linear regression analysis

# print("P-values with linear agression analysis")
# feature_Analysis_LinearRegression(random_state=42)


#TODO: Best Subsets
# feature_best_subsets()


#TODO: Step-wise approaches (and/or Recursive Feature Elimination)
# feature_Analysis_RFE()


#TODO: Lasso
# feature_Analysis_LASSO(random_state=42)


#TODO: Elastic Net.
# feature_Analysis_ENET(random_state = 42)


#TODO: Adaptive Lasso.
# feature_Analysis_ALASSO()


#1.A.ii.1
# regularization path for LASSO

def regularization_path_lasso(random_state=42):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)
    list_numerical = X_train.columns
    # standerdize the data
    scaler_train = StandardScaler().fit(X_train[list_numerical])
   

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])

    features = X_train.columns.to_list()
    alphas = np.linspace(0.001,1,100)
    lasso = Lasso(max_iter=10000)
    coefs = []

    for a in alphas:
        lasso.set_params(alpha=a)
        lasso.fit(X_train, y_train)
        coefs.append(lasso.coef_)

    ax = plt.gca()

    ax.plot(alphas, coefs, label = features)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('lambda')
    plt.ylabel(' Standerdized Coefficients')
    plt.title('Lasso coefficients as a function of lambda')
    plt.legend()
    plt.show()


#1.A.ii.2
# regularization path for ENET alpha = .2m .7


def regularization_path_ENET(random_state=42):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)
    list_numerical = X_train.columns
    # standerdize the data
    scaler_train = StandardScaler().fit(X_train[list_numerical])
   

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
    features = X_train.columns.to_list()

    alphas = np.linspace(0.001,10,1000)

    RATIOS =[.2, .8]


    RATIO = .8
    for alpha in alphas:
        model = linear_model.ElasticNet(alpha = alpha, l1_ratio=RATIO, fit_intercept= True, max_iter=10000 )
        coefs = []

        for a in alphas:
            model.set_params(alpha=a)
            model.fit(X_train, y_train)
            coefs.append(model.coef_)

        ax = plt.gca()

        ax.plot(alphas, coefs, label = features)
        ax.set_xscale('log')
        plt.axis('tight')
        plt.xlabel('lambda')
        plt.ylabel(' Standerdized Coefficients')
        plt.title('Elastic Net coefficients as a function of lambda with alpha as: '+ str(RATIO))
        plt.legend()
        plt.show()

#1.A.ii.3
# regularization path for RIDGE


def regularization_path_RIDGE(random_state=42):
     # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    # split the data into testing and training
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=random_state)
    list_numerical = X_train.columns
    # standerdize the data
    # standerdize the data
    scaler_train = StandardScaler().fit(X_train[list_numerical])
    

    X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
    X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
    features = X_train.columns.to_list()
    alphas = np.linspace(1,100000,10000)
    model = linear_model.Ridge(max_iter=10000, fit_intercept = True )
    coefs = []

    for a in alphas:
        model.set_params(alpha=a)
        model.fit(X_train, y_train)
        coefs.append(model.coef_)

    ax = plt.gca()

    ax.plot(alphas, coefs, label = features)
    ax.set_xscale('log')
    plt.axis('tight')
    plt.xlabel('lambda')
    plt.ylabel(' Standerdized Coefficients')
    plt.title('RIDGE coefficients as a function of lambda')
    plt.legend()
    plt.show()



# Question 1, Part a, Section ii

#TODO: Lasso Regularization
# regularization_path_lasso()


#TODO: ENET Regularization
# regularization_path_ENET()



#TODO: RIDGE Regularization
# regularization_path_RIDGE()





#1.B.i.1 LR OLS
 
def prediction_LinearRegression():
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    NUM_TESTS = 10
    mse_sum = 0
    for iteration in range(NUM_TESTS):
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        # X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33, random_state=random_state)


        # fit the linear model
        lm=linear_model.LinearRegression()
        lm.fit(X_train, y_train)
        y_predict = lm.predict(X_test)
        mse_sum += mean_squared_error(y_test, y_predict)
    
    avg_mse = mse_sum / NUM_TESTS
    

    
    print('Linear Regression Avg Mean Squared Error over:', str(NUM_TESTS), "tests:", str(avg_mse))


#1.B.i.2 RIDGE
def prediction_RidgeRegression():
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')





    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)


    NUM_TESTS = 10
    mse_sum = 0
    for iteration in range(NUM_TESTS):
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33)
        list_numerical = X_train.columns
        # standerdize the data
        scaler_train = StandardScaler().fit(X_train[list_numerical])
        

        X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
        X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
        X_validate[list_numerical] = scaler_train.transform(X_validate[list_numerical])


        # fit the linear model
        alphas = np.linspace(0, 100, 5000)

        errors =[]
        alpha_list =[]

        lowest_mse = 1
        for alpha in alphas:
            lm=linear_model.Ridge(alpha = alpha)
            lm.fit(X_train, y_train)
            y_predict = lm.predict(X_validate)
            mse = mean_squared_error(y_validate, y_predict)

            errors.append(mse)
            alpha_list.append(alpha)
            # print(mse)
            if mse < lowest_mse:
                lowest_mse = mse
                best_model = lm
                best_alpha = alpha
        # print()
       
        ax = plt.gca()
        ax.set_xscale("log")
        ax.plot(alpha_list, errors)
        
        y_test_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
            
        
        print( "Lambda chosen at:",str(best_alpha))
        mse_sum += mse
    
    avg_mse = mse_sum / NUM_TESTS
    print("RIDGE test avg mse:",str(avg_mse))
    plt.show()

    

    

    
    
    print('Ridge Regression Avg Mean Squared Error over:', str(NUM_TESTS), "tests:", str(avg_mse))



#1.B.i.3 BEST SUBSETS

def processSubset(feature_set, y_train, X_train):
    # Fit model on feature_set and calculate RSS

    X = X_train.copy()
    y = y_train.copy()
    
    feature_set = ("Intercept",)+feature_set
    # print()
    # print("feature set:",str(feature_set))
    # print()

    # print(X)
    X.insert(0, 'Intercept', 1)

    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS, "features": feature_set}


def getBestModel(k, y_train, X_train):
    
    results = []
    
    for combo in itertools.combinations(X_train.columns, k):
        results.append(processSubset(combo, y_train, X_train))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    
    # print("Processed", models.shape[0], "models on", k)

    
    # Return the best model, along with some other useful information about the model
    return (best_model)


def prediction_best_subsets():
   # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    # data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    plt.figure(figsize=(20,20))
    sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')


    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)
   

    # print(X)

    NUM_TESTS = 10
    mse_sum = 0
    
    for iteration in range(NUM_TESTS):
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33)
        X_validate.insert(0, 'Intercept', 1)

        # fit the linear model
        
        lowest_mse = 1

        for k in range(1, len((X_train.columns).to_list())+1):

            optimal_k = getBestModel(k, y_train, X_train)

            feature_set = optimal_k["features"]
            model_k = optimal_k["model"]
            
    
            
            
            y_predict =  model_k.predict(X_validate[list(feature_set)])

            
            mse = mean_squared_error(y_validate, y_predict)
            if mse < lowest_mse:
                lowest_mse = mse
                best_model = model_k
                best_features = feature_set
        
        print("Best Features:", str(best_features))
        X_test.insert(0, 'Intercept', 1)
        y_predict_test =  best_model.predict(X_test[list(best_features)])
        test_mse = mean_squared_error(y_predict_test,y_test)
        mse_sum += test_mse
    
    avg_mse = mse_sum / NUM_TESTS


    print('Ridge Regression Avg Mean Squared Error over:', str(NUM_TESTS), "tests:", str(avg_mse))
    
    


#1.B.i.4 RFE

def prediction_Analysis_RFE():
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')



    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)

    NUM_TESTS = 10
    mse_sum = 0
    best_ks =[]
    for iteration in range(NUM_TESTS):
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33)
        list_numerical = X_train.columns
        # standerdize the data
        scaler_train = StandardScaler().fit(X_train[list_numerical])
       

        X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
        X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
        X_validate[list_numerical] = scaler_train.transform(X_validate[list_numerical])


        # fit the RFE model
       
        min_mse =1
        for k in range(1, len((X_train.columns).to_list())+1):
            estimator = SVR(kernel="linear")
            selector = RFE(estimator, n_features_to_select=k, step=1)
            # selector = RFE(estimator, n_features_to_select=5, step=1)
            selector = selector.fit(X_train, y_train)

            y_predict = selector.predict(X_validate)
            mse = mean_squared_error(y_validate, y_predict)
            print(mse)
            
            
            print(iteration*k+k)

            if mse < min_mse:
                min_mse = mse
                best_model = selector
                best_k = k
        


        y_predict_test = best_model.predict(X_test)
        test_mse = mean_squared_error(y_predict_test,y_test)
        mse_sum += test_mse
        best_ks.append(best_k)
       
        

    
    avg_mse = mse_sum / NUM_TESTS
    

    
    print('RFE Avg Mean Squared Error over:', str(NUM_TESTS), "tests:", str(avg_mse))
    print(best_ks)

#1.B.i.5 LASSO

def prediction_Analysis_LASSO(random_state=42):
    # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')





    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)


    NUM_TESTS = 10
    mse_sum = 0
    chosen_lambdas = []
    for iteration in range(NUM_TESTS):
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33)
        list_numerical = X_train.columns
        # standerdize the data
        
        scaler_train = StandardScaler().fit(X_train[list_numerical])
       

        X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
        X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
        X_validate[list_numerical] = scaler_train.transform(X_validate[list_numerical])


        # fit the linear model
        alphas = np.linspace(0, 100, 5000)

        errors =[]
        alpha_list =[]
        

        lowest_mse = 1
        for alpha in alphas:
            lm=linear_model.Lasso(alpha = alpha)
            lm.fit(X_train, y_train)
            y_predict = lm.predict(X_validate)
            mse = mean_squared_error(y_validate, y_predict)

            errors.append(mse)
            alpha_list.append(alpha)
            
            if mse < lowest_mse:
                lowest_mse = mse
                best_model = lm
                best_alpha = alpha
        
        chosen_lambdas.append(best_alpha)
        ax = plt.gca()
        ax.set_xscale("log")
        ax.plot(alpha_list, errors)
        
        y_test_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
            
        
        
        mse_sum += mse
    
    avg_mse = mse_sum / NUM_TESTS
    print("LASSO test avg mse:",str(avg_mse))
    print( "Lambdas chosen:",str(chosen_lambdas))
    plt.show()

#1.Bi.6 ELASTIC NET

def predict_Analysis_ENET(random_state=42):
   # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')





    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)


    NUM_TESTS = 10
    mse_sum = 0
    hyp_pairs=[]
    for iteration in range(NUM_TESTS):
        
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33)
        list_numerical = X_train.columns
        # standerdize the data
        scaler_train = StandardScaler().fit(X_train[list_numerical])
       

        X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
        X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
        X_validate[list_numerical] = scaler_train.transform(X_validate[list_numerical])


        # fit the linear model
        alphas = np.linspace(.1, 5, 100)
        ratios = np.linspace(0, 1, 10)

        errors =[]
        alpha_list =[]

        lowest_mse = 1
       
        for ratio in ratios:
            for alpha in alphas:
                lm = linear_model.ElasticNet(alpha = alpha, l1_ratio=ratio, fit_intercept= True)
                lm.fit(X_train, y_train)
                y_predict = lm.predict(X_validate)
                mse = mean_squared_error(y_validate, y_predict)

                errors.append(mse)
                alpha_list.append(alpha)
                print(mse)
                if mse < lowest_mse:
                    lowest_mse = mse
                    best_model = lm
                    best_alpha = alpha
                    best_ratio = ratio
        # print()
       
        ax = plt.gca()
        ax.set_xscale("log")
        ax.plot(alpha_list, errors)
        
        y_test_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        hyp_pairs.append((best_ratio, best_alpha))
        
        print( "Lambda chosen at:",str(best_alpha))
        mse_sum += mse
    
    avg_mse = mse_sum / NUM_TESTS
    print("ENET test avg mse:",str(avg_mse))
    print(hyp_pairs)
    plt.show()

#1.B.i.7 Adaptive Lasso
def predict_Analysis_ALASSO(random_state=42):
     # get data
    data = pd.read_csv (r'/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls')
    #Lets load the dataset and sample some
    column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    data = read_csv('/Users/joshdavis/Desktop/Machine Learning/hw1/code/housing.csv.xls', header=None, delimiter=r"\s+", names=column_names)
    data.insert(0, 'Intercept', 1)
    # print (data)
    # Finding out the correlation between the features
    corr = data.corr()
    # plt.figure(figsize=(20,20))
    # sns.heatmap(corr, cbar=True, square= True, fmt='.1f', annot=True, annot_kws={'size':15}, cmap='Greens')





    # seperate the labels from the features. In this cade MEDV is the feature we will be trying to predict.
    y = np.log(data["MEDV"])
    # print(y)
    X = data.drop('MEDV', axis=1)

    # print(X)


    NUM_TESTS = 10
    mse_sum = 0
    for iteration in range(NUM_TESTS):
        # split the data into testing and training and validation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


        X_train, X_validate, y_train, y_validate =  train_test_split(X_train, y_train, test_size=0.33)
        list_numerical = X_train.columns
        # standerdize the data
        scaler_train = StandardScaler().fit(X_train[list_numerical])
    

        X_train[list_numerical] = scaler_train.transform(X_train[list_numerical])
        X_test[list_numerical] = scaler_train.transform(X_test[list_numerical])
        X_validate[list_numerical] = scaler_train.transform(X_validate[list_numerical])


       
       
       

        lambas = np.linspace(.0001, 1, 1000)

        
        lowest_mse = 1
        alpha_list =[]
        errors=[]
        for alpha in lambas:
            alpha_list.append(alpha)
            
            g = lambda w: np.sqrt(np.abs(w))
            gprime = lambda w: 1. / (2. * np.sqrt(np.abs(w)) + np.finfo(float).eps)

            n_samples, n_features = X_train.shape
        

            weights = np.ones(n_features)
            n_lasso_iterations = 10

            for k in range(n_lasso_iterations):
                X_w = X_train / weights[np.newaxis, :]
                model = Lasso(alpha=alpha, fit_intercept=True)
                model.fit(X_w, y_train)
                coef_ = model.coef_ / weights
                weights = gprime(coef_)
                # print("ITERATION:",str(k))

            y_pred = model.predict(X_validate)
            mse = mean_squared_error(y_validate, y_pred)
            errors.append(mse)
            # print(mse)
            if mse < lowest_mse:
                lowest_mse = mse
                best_model = model
                best_alpha = alpha
            # print()
        
        ax = plt.gca()
        ax.set_xscale("log")
        ax.plot(alpha_list, errors)
        
        y_test_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
            
        
        print( "Lambda chosen at:",str(best_alpha))
        mse_sum += mse
        
    avg_mse = mse_sum / NUM_TESTS
    print("ALASSO test avg mse:",str(avg_mse))
    plt.show()

        
           
       
# Question 1, Part B, Section i.

#TODO: linear regression analysis
# prediction_LinearRegression()

#TODO: ridge regression analysis
# prediction_RidgeRegression()


#TODO: Best Subsets
# prediction_best_subsets()


#TODO: Step-wise approaches (and/or Recursive Feature Elimination)
prediction_Analysis_RFE()


#TODO: Lasso
# prediction_Analysis_LASSO(random_state=42)


#TODO: Elastic Net.
# predict_Analysis_ENET(random_state = 42)


#TODO: Adaptive Lasso.
# predict_Analysis_ALASSO()