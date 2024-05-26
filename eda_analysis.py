import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_column',None)
pd.set_option('display.width',700)

df = sns.load_dataset('titanic')
df.head(5)

#About
df.shape
df.info()
df.describe().T

#About Columns
def grab_col_names(dataframe, cat_th=10,car_th=10):
    ## CATEGORY COLUMNS
    category_cols = [col for col in df.columns if str(df[col].dtypes) in ['category','object','bool']]
    num_but_cat = [col for col in df.columns if df[col].nunique()<10 and df[col].dtypes in ['int64','float64']]
    cat_but_car = [col for col in df.columns if df[col].nunique()>20 and str(df[col].dtypes) in ['category','object']]
    category_cols = category_cols + num_but_cat
    category_cols = [col for col in category_cols if col not in cat_but_car]

    ## NUMERIC COLUMNS
    num_cols = [col for col in df.columns if df[col].dtypes in ['int64','float64']] # burda qarisiq verirdi deye digeri
    num_cols = [col for col in df.columns if col not in category_cols]

    print(f"Observations:{dataframe.shape[0]}") # setir yeni
    print(f"Variables:{dataframe.shape[1]}") # sutun yeni
    print(f"Category_Cols: {len(category_cols)}")
    print(f"Numeric_Cols: {len(num_cols)}")
    print(f"Cat_but_car: {len(cat_but_car)}")
    print(f"Num_but_cat: {len(num_but_cat)}")


#Overview
def check_dif(dataframe,head =10):
    print('################### SHAPE ####################')
    print(dataframe.shape)
    print('################### TYPES ####################')
    print(dataframe.dtypes)
    print('################### HEAD ####################')
    print(dataframe.head(head)) # altdaki ele sadece basligi
    print('################### TAİL ####################')
    print(dataframe.tail(head)) # sonlar head ucun 10 nezerde tutulub check_dif(df,30)
    print('################### MİSSİNG VALUES  ####################')
    print(dataframe.isnull().values.any())
    print(dataframe.isnull().sum()) # NA
    print('################### Quantiles ####################')
    print(dataframe.describe([0,0.05,0.5,0.95,0.99,1]).T)

# Category Visualization
import matplotlib.pyplot as plt
def cat_summary(dataframe,col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(), 'Ratio': 100*dataframe[col_name].value_counts()/len(dataframe)}))
    print('#################################################')
    if plt.plot:
        sns.countplot(x=dataframe[col_name], data= dataframe)
        plt.show(block = True )

category_cols = [col for col in df.columns if str(df[col].dtypes) in ['category','object','bool']]
for col in category_cols:
    cat_summary(df,col)

#Numerical Visualization
def num_summary(dataframe, numerical_col):
    quantiles = [0,0.05,0.5,0.7,0.8,0.9,0.95,0.99,1]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plt.plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block = True)

num_cols = [col for col in df.columns if df[col].dtypes in ['int64','float64']] # burda qarisiq verirdi deye digeri
for col in num_cols:
    num_summary(df,col)

#Encoding Scaling tətbiq etmək mümkündür
#Label encoding prioritet

#TARGET

#CATEGORY
def target_summary_with_category(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end=' ')
#sample target_summary_with_category(df,'survived','class')
category_cols = [col for col in df.columns if str(df[col].dtypes) in ['category','object','bool']]
for col in category_cols:
    target_summary_with_category(df,'survived',col)

#NUMERICAL
def target_summary_with_numeric(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: 'mean'}), end=' ')
#sample target_summary_with_numeric(df,'survived','age')
num_cols = [col for col in df.columns if df[col].dtypes in ['int64','float64']] # burda qarisiq verirdi deye digeri
for col in num_cols:
    target_summary_with_category(df,'survived',col)


#Correlation
correlation = df[num_cols].corr()
sns.set(rc={'figure.figsize':(12,12)})
sns.heatmap(correlation,cmap = 'GnBu')
plt.show()

#///////////////////////////////////////
corr_matrix = df[num_cols].corr().abs() # indi burda axi mirror kimi idi bir sayi iki defe var idi bunun yarsinin atdiq
import numpy as np
upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
# drop list yaradiriq
# yuksekleri silirik cunki bezen bize ziyan verirler
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
# drop list yuksek correlasiyon olanlardi
df.drop(drop_list,axis=1)

#//////////////////////////////////
# HIGH CORRELATION FAR AWAY
def high_correlation_cols(dataframe, corr_th= 0.90):
    num_cols = [col for col in df.columns if df[col].dtype in ['int64','float64']]
    corr = dataframe[num_cols].corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col]>0.90)]
    df.drop(drop_list, axis=1)


high_correlation_cols(df)
df.head()

#OUTLIERS
def find_outliers(dataframe, col_name):
    q1 = df['age'].quantile(0.25)
    q3 = df['age'].quantile(0.75)
    iqr = q3-q1
    up = q3 + 1.5*iqr
    low= q1-1.5*iqr
    outliers = dataframe[(dataframe[col_name] < low) | (dataframe[col_name] > up)]
    return outliers

#SAMPLE
age_outliers = find_outliers(df, 'age')
print("Outliers in age column:")
print(age_outliers)
df = df[~((df['age'] < age_outliers['age'].min()) | (df['age'] > age_outliers['age'].max()))]
print("Data shape after removing outliers:", df.shape)