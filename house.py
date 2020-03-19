import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin

import os



df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

df_train.SalePrice.describe()


df_train['SalePrice'] = np.log1p(df_train['SalePrice'])

#normalfit(for linear model)
fig, ax1 = plt.subplots()
sns.distplot(df_train['SalePrice'], ax=ax1, fit=stats.norm)

#sigma and MU
(mu, sigma) = stats.norm.fit(df_train['SalePrice'])
ax1.set(title='Normal distribution ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma))


fig, ax2 = plt.subplots()
stats.probplot(df_train['SalePrice'], plot=plt)





#find corelation in numeric fetures
corr = df_train.corr()   # or df_train[num_columns].corr()
top_corr_feat = corr['SalePrice'].sort_values(ascending=False)[:28]
print(top_corr_feat)

threshold = 0.55
top_corr = corr.index[np.abs(corr["SalePrice"]) > threshold]



plt.figure(figsize=(10,10))
sns.heatmap(df_train[top_corr].corr(),annot=True,cmap="RdBu_r")


for col in top_corr_feat.index[:25]:
    print('{} - unique values: {} - mean: {:.2f}'.format(col, df_train[col].unique()[:5], np.mean(df_train[col])))
    
    
cols = 'SalePrice GrLivArea GarageArea TotalBsmtSF YearBuilt 1stFlrSF MasVnrArea '.split()

with plt.rc_context(rc={'font.size':14}): 
    fig, ax = plt.subplots(figsize=(20,15), tight_layout=True)    
    pd.plotting.scatter_matrix(df_train[cols], ax=ax, diagonal='kde', alpha=0.8)
    
    
cut_area = 4600
# remove points in the SalePrice - GrLivArea scatter plot
df_train = df_train.loc[df_train['GrLivArea'] < cut_area]



df_train_id = df_train['Id']
df_test_id = df_test['Id']

df_train.drop("Id", axis=1, inplace=True)
df_test.drop("Id", axis=1, inplace=True)

# same transformation to the train / test datasets to avoid irregularities
size_train = len(df_train.index)
size_test = len(df_test.index)

df_tot = pd.concat([df_train, df_test], sort=False).reset_index(drop=True)
df_tot.drop(['SalePrice'], axis=1, inplace=True)

y_train = df_train['SalePrice'].values


#nan
df_na = (df_tot.isnull().sum()) / len(df_tot) * 100
df_na = df_na.drop(df_na[df_na==0].index).sort_values(ascending=False)

#plot nan
 

with plt.rc_context(rc={'font.size':14}):
    fig, ax = plt.subplots(figsize=(16,6))
    sns.barplot(df_na.index, df_na, ax=ax)
    ax.set(xlabel='Features', ylabel='Missing values percentages')
    ax.tick_params(axis='x', rotation=90)
    
for col in 'PoolQC MiscFeature Alley Fence FireplaceQu'.split():
    df_tot[col].fillna('None', inplace=True)
    
    
df_tot["LotFrontage"] = df_tot.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))    
    

# Replace Garage categorical values by None
for col in 'GarageType GarageFinish GarageQual GarageCond'.split():
    df_tot[col].fillna('None', inplace=True)
    
# Replace Garage numeric values by 0
for col in 'GarageYrBlt GarageCars GarageArea'.split():
    df_tot[col].fillna(0, inplace=True)

# Same replacements (that Garage columns)
for col in 'BsmtQual BsmtCond BsmtExposure BsmtFinType1 BsmtFinType2'.split():
    df_tot[col].fillna('None', inplace=True)
    
# Replace numeric values by 0
for col in 'BsmtFinSF1 BsmtFinSF2 BsmtUnfSF TotalBsmtSF BsmtFullBath BsmtHalfBath'.split():
    df_tot[col].fillna(0, inplace=True)
    
df_tot['MasVnrArea'].fillna(0, inplace=True)
df_tot['MasVnrType'].fillna('None', inplace=True)


df_tot['MSZoning'].value_counts() / len(df_tot) * 100
df_tot['MSZoning'].fillna('RL', inplace=True)

df_tot = df_tot.drop(['Utilities'], axis=1)
df_tot['Functional'].fillna("Typ")
df_tot['Electrical'].fillna(df_tot['Electrical'].mode()[0])

df_tot['Exterior1st'].fillna(df_tot['Exterior1st'].mode()[0], inplace=True)
df_tot['Exterior2nd'].fillna(df_tot['Exterior2nd'].mode()[0], inplace=True)
df_tot['KitchenQual'].fillna(df_tot['KitchenQual'].mode()[0], inplace=True)

df_tot['SaleType'].fillna(df_tot['SaleType'].mode()[0], inplace=True)
df_tot['MSSubClass'].fillna('None', inplace=True)







#Check remaining missing values if any 
all_data_na = (df_tot.isnull().sum() / len(df_tot)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()




#num_cols = df_tot.select_dtypes(exclude='object').columns
#print('Numeric columns ({}) \n{}'.format(len(num_cols), num_cols))

#categ_cols = df_tot.select_dtypes(include='object').columns
#print('\nCategorical columns ({}) \n{}'.format(len(categ_cols), categ_cols))


#MSSubClass=The building class
df_tot['MSSubClass'] = df_tot['MSSubClass'].apply(str)


#Changing OverallCond into a categorical variable
df_tot['OverallCond'] = df_tot['OverallCond'].astype(str)




#Year and month sold are transformed into categorical features.
df_tot['YrSold'] = df_tot['YrSold'].astype(str)
df_tot['MoSold'] = df_tot['MoSold'].astype(str)

from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
# process columns, apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(df_tot[c].values)) 
    df_tot[c] = lbl.transform(list(df_tot[c].values))







 
df_tot['TotalSF'] = df_tot['TotalBsmtSF'] + df_tot['1stFlrSF'] + df_tot['2ndFlrSF'] + df_tot['GrLivArea'] + df_tot['GarageArea']

# Combine the bathrooms
df_tot['Bathrooms'] = df_tot['FullBath'] + df_tot['HalfBath']* 0.5 

numeric_feats = df_tot.dtypes[df_tot.dtypes != "object"].index


from scipy import stats
from scipy.stats import norm, skew
# Check the skew of all numerical features
skewed_feats = df_tot[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})


skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    df_tot[feat] = boxcox1p(df_tot[feat], lam)

#drop_elements1 = ['BsmtFinType2','Utilities','Alley','GarageYrBlt','GarageQual','YearRemodAdd','RoofMatl','Heating','CentralAir','BsmtHalfBath','Functional','GarageCond','PavedDrive','OpenPorchSF','EnclosedPorch','ScreenPorch','PoolQC','Fence','MiscFeature', 'SaleType', 'MiscVal', 'PoolArea', '3SsnPorch','LowQualFinSF','Electrical']


#df_tot = df_tot.drop(drop_elements1, axis = 1)


df_tot = pd.get_dummies(df_tot)

#a=[ 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
#       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt',
#       'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
#       'BsmtFinSF1']

#dft = df_tot[a]

df_train = df_tot[:size_train]
df_test = df_tot[size_train:]






#from sklearn.model_selection import train_test_split
#Xtrain, Xtest, ytrain, ytest = train_test_split(df_train, y_train, shuffle=True, 
#                                                test_size=0.2, random_state=28)

#from sklearn.ensemble import RandomForestRegressor
#regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
#regressor.fit(df_train, y_train)





from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb











n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(df_train.values)
    rmse= np.sqrt(-cross_val_score(model, df_train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0) 
    
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))


ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)

model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1) 
    
averaged_models = AveragingModels(models = (regressor, ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))







averaged_models.fit(df_train, y_train)



y_pred_train=averaged_models.predict(df_train)
#y_pred_test=regressor.predict(df_train)
import math 
import sklearn.metrics as sklm
print('Root Mean Square Error train = ' + str(math.sqrt(sklm.mean_squared_error(y_train, y_pred_train))))
#print('Root Mean Square Error test = ' + str(math.sqrt(sklm.mean_squared_error(ytest, y_pred_test))))   

y_p=averaged_models.predict(df_test)
#yp=regressor.predict(df_test)
df_su = pd.DataFrame({'Id': df_test_id, 'SalePrice': y_p})
print(df_su.head())
df_su.to_csv('submission7.csv',index=False)


