#import the necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import jaconv
import re
import pickle as pkl
import warnings
import statsmodels.api as sm
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# To ignore the wanrings. Got many warnings from the data type
warnings.filterwarnings("ignore")
plt.rcParams['font.family'] = 'MS Gothic'

NUMBER_OF_DATA = 50000

## Data Preparation
house_price_test = pd.read_csv("school\\データサイエンス超入門\\通常課題１\\data\\utf\\13_Tokyo_20231_20231.csv")
house_price_history_data = pd.read_csv("school\\データサイエンス超入門\\通常課題１\\data\\utf\\13_Tokyo_20053_20224.csv")
# house_price_history_data = pd.read_csv("school\\データサイエンス超入門\\通常課題１\\data\\utf\\SuperD_Class2_tochi_sample_xdm.csv")
house_price_alltime = house_price_history_data
house_price_recent50000 = house_price_history_data[house_price_history_data["取引時点"] == '2022年第４四半期']
code_reference = pd.read_csv("school\\データサイエンス超入門\\通常課題１\\data\\utf\\SuperD_Class2_shicyoukuson_code_utf8.csv")

# get the most recent 50000 data
if len(house_price_recent50000) > NUMBER_OF_DATA:
    house_price_recent50000 = house_price_recent50000.sample(n = NUMBER_OF_DATA, random_state = 42)
else:
    house_price_history_data.sort_values(by = '取引時点', ascending= False, inplace = True)
    house_price_recent50000 = house_price_history_data.iloc[0:NUMBER_OF_DATA,:]
# get random 50000 data
house_price_random50000 = house_price_history_data.sample(n = NUMBER_OF_DATA, random_state= 42)
# print(house_price_recent50000["取引時点"].value_counts())
# print(house_price_random50000["取引時点"].value_counts())

## Price distribution
# house_price_recent50000["取引価格（総額）"].hist(bins=50)
# plt.title("Histogram of price")
# plt.show()

#Take log price
def log_price(df):
    df["log_price"] = np.log(df["取引価格（総額）"])
    return df
# house_price_recent50000["log_price"].hist(bins=50)
# plt.title("Histogram of price")
# plt.show()

## Transaction date

# Create a dataframe
time = pd.DataFrame(house_price_alltime["取引時点"].value_counts())
# Find how many unique quarters in the data
# print(set(time.index.str[6]))
# make a dictionary out of it for further use
quarterly_dict = {'１': 1, '２': 4, '３': 7, '４': 10}

# Finally
transaction_date2date = {}
for i in range(len(time)):
    # the original str
    transaction_date = time.index[i]
    # year
    year_num = time.index[i][0:4]
    # quarterly to month
    month_num = quarterly_dict[jaconv.h2z(time.index[i][6])]
    # transform to datetime format
    date = datetime.datetime.strptime(str(year_num) + str(month_num), "%Y%m")
    transaction_date2date.update({transaction_date: date})

## Longitude and Latitude

# create code dict
code2lng, code2lat = ({} for _ in range(2))
for i in range(len(code_reference)):
    code = str(code_reference["コード"].iloc[i])
    if (len(code) == 5):
        code = code[0:4]
    else:
        code = code[0:5]
    code2lng.update({int(code): code_reference["経度"].iloc[i]})
    code2lat.update({int(code): code_reference["緯度"].iloc[i]})

## Location convert
def location_convert(df):
    df["longitude"] = df["市区町村コード"].map(code2lng)
    df["latitude"] = df["市区町村コード"].map(code2lat)
    return df

## Date Convert
def date_convert(df):
    df["date"] = df["取引時点"].map(transaction_date2date)
    return df

## Mean log price
def mean_log_price(df):
    df["date_mean_log_price"] = df.groupby("date")["log_price"].transform(np.mean)
    df_date = pd.DataFrame(house_price_recent50000.groupby("date")["log_price"].apply(np.mean))
    plt.plot(df_date["log_price"])
    plt.title("mean log price 2005-2023",size=16)
    plt.xlabel("year",size=16)
    plt.ylabel("mean log price",size=16)
    plt.show()
    return df

## Area Convert
def area_convert(df):
    df_area = pd.DataFrame(df["面積（㎡）"].value_counts())
    square_meters2area = {}
    for i in range(len(df_area)):
        square_meters = df_area.index[i]
        area = re.sub("㎡以上", "", square_meters)
        area = re.sub("m&sup2;以上", "", area)
        area = re.sub(",", "", area)
        area = int(area)
        square_meters2area.update({square_meters: area})
    df["area"] = df["面積（㎡）"].map(square_meters2area)
    return df

## Type Convert
def type_convert(df):
    df = pd.get_dummies(df, columns=["種類"])
    return df

## Circumstances
def cirum_convert(df):
    df = pd.get_dummies(df, columns=['取引の事情等'])
    return df

## Total Convert
def total_convert(df):
    df = cirum_convert(type_convert(area_convert(log_price(location_convert(date_convert(df))))))
    return df

## Create feature and label dataframe
def create_feature_and_label_dataframe(df):
    df_feature = df[[
    "area", "longitude", "latitude", '種類_中古マンション等',
    '種類_宅地(土地)', '種類_宅地(土地と建物)', '種類_林地', '種類_農地',
    '取引の事情等_その他事情有り',
    '取引の事情等_瑕疵有りの可能性',
    '取引の事情等_私道を含む取引', '取引の事情等_調停・競売等',
    '取引の事情等_調停・競売等', '取引の事情等_調停・競売等、私道を含む取引',
    '取引の事情等_関係者間取引', '取引の事情等_関係者間取引、私道を含む取引', '取引の事情等_隣地の購入',
    '取引の事情等_隣地の購入、私道を含む取引'
    ]]
    # Target
    df_label = df[["log_price"]]
    return df_feature, df_label

## Extract and load
def etl(df_feature, df_label):
    # write to csv
    df_feature.to_csv("land_price_x.csv", index=False)
    df_label.to_csv("land_price_y.csv", index=False)
    # save as pickle
    with open('df_x.pickle', 'wb') as f:
        pkl.dump(df_feature, f)
    with open('df_y.pickle', 'wb') as f:
        pkl.dump(df_label, f)
    # load pickle
    with open('df_x.pickle', 'rb') as f:
        df_feature = pkl.load(f)
    with open('df_y.pickle', 'rb') as f:
        df_label = pkl.load(f)
    return df_feature.values, df_label.values

## Datatype adjustment
def to_float(data):
    data = data.astype(float)
    return data

## Dealing with training sets
house_price_recent50000 = total_convert(house_price_recent50000)
house_price_random50000 = total_convert(house_price_random50000)

# print(house_price_recent50000["log_price"].describe())
# print(house_price_random50000["log_price"].describe())
# print(house_price_recent50000["取引時点"].value_counts())
# print(house_price_random50000["取引時点"].value_counts())

recent50000_feature_df, recent50000_label = create_feature_and_label_dataframe(house_price_recent50000)
random50000_feature_df, random50000_label = create_feature_and_label_dataframe(house_price_random50000)

# print(recent50000_feature_df.columns)
# print(recent50000_feature_df["area"].value_counts())
# print(random50000_feature_df.columns)
# print(random50000_feature_df["area"].value_counts())

recent50000_feature, recent50000_label = etl(recent50000_feature_df, recent50000_label)
random50000_feature, random50000_label = etl(random50000_feature_df, random50000_label)
# print(recent50000_feature.shape, recent50000_label.shape, random50000_feature.shape, random50000_label.shape)

recent50000_feature = recent50000_feature_df.astype(float)
recent50000_label = recent50000_label.astype(float)
random50000_feature = random50000_feature_df.astype(float)
random50000_label = random50000_label.astype(float)

## Dealing with testing sets
house_price_test = total_convert(house_price_test)
test_feature, test_label = create_feature_and_label_dataframe(house_price_test)
test_feature, test_label = etl(test_feature, test_label)
test_feature = test_feature.astype(float)
test_label = test_label.astype(float)

print(type(test_feature))
## Machine Learning

def data_normalize(data_array):
    return normalize(data_array, norm='l1')

def show_Rsquare_and_mse(y_test, y_test_predict):
    mse = mean_squared_error(y_test, y_test_predict)
    r2 = r2_score(y_test, y_test_predict)
    print(f'R-squared: {r2}')
    print(f'Mean Squared Error: {mse}')


## Linear Regression powered by sklearn
def sklearn_linear_learning(x_train, x_test, y_train, y_test, df):
    y_train = np.reshape(y_train, [-1])
    y_test  = np.reshape(y_test, [-1])
    model = LinearRegression()
    model.fit(data_normalize(x_train), y_train)
    # y_train_predict = model.predict(x_train)
    y_test_predict = model.predict(data_normalize(x_test))
    print("\nTraining result powered by Sklearn")
    show_Rsquare_and_mse(y_test, y_test_predict)
    coefficients = model.coef_
    for feature, coef in zip(df.columns, coefficients):
        print(f"{feature}: {coef}")
    return y_test, y_test_predict
    # r2 = r2_score(y_train, y_train_predict)
    # mse = mean_squared_error(y_train, y_train_predict)
    # print(f'R-squared: {r2:.4f}')
    # print(f'Mean Squared Error: {mse:.4f}')

    

## Linear Regression powered by statisics model
def stats_linear_regression(x_train, x_test, y_train, y_test):
    # stats model
    reg_linear = sm.OLS(y_train, x_train)
    result = reg_linear.fit()
    # prediction
    y_test_predict = result.predict(x_test)
    print("\nTraining result powered by Stats Model")
    # print(mean_squared_error(y_test_predict, y_test))
    print(result.summary())
    return y_test, y_test_predict

## Lasso Regression Model

def lasso_linear_regression(x_train, x_test, y_train, y_test, df):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)
    alpha = 0.01
    lasso_model = Lasso(alpha=alpha)
    lasso_model.fit(X_train_scaled, y_train)
    y_test_predict = lasso_model.predict(X_test_scaled)
    print("\nTraining result powered by Lasso Regression Model")
    show_Rsquare_and_mse(y_test, y_test_predict)
    coefficients = lasso_model.coef_
    for feature, coef in zip(df.columns, coefficients):
        print(f"{feature}: {coef}")
    return y_test, y_test_predict


## Random Forest

def show_importance_map(importance, df, title):
    plt.figure(figsize=(6,6))
    importances = pd.Series(importance, index=df.columns)
    importances = importances.sort_values()
    importances[-15:].plot(kind="barh")
    plt.title(title)

def ranfom_forest(x_train, x_test, y_train, y_test, df, importance_map_title):
    # Hyper Parameter Learning
    hyper_para_learn = 0
    reg_rf = RandomForestRegressor()

    if hyper_para_learn == 1:
        reg_rf_cv = GridSearchCV(reg_rf, {
            'max_depth': [2, 6, 10],
            'n_estimators': [100, 1000, 5000],
            'max_features': ['log2'],
            'n_jobs': [24]
        }, verbose=1)
        reg_rf_cv.fit(x_train, y_train)
        parm_dict = reg_rf_cv.best_params_
        print(parm_dict)
        print(reg_rf_cv.best_score_)
    else:
        parm_dict = {}       
        parm_dict["max_depth"] = 10
        parm_dict["max_features"] = 'log2'
        parm_dict["n_estimators"] = 5000
        parm_dict["n_jobs"] = 24

    reg_rf = RandomForestRegressor(
        max_depth = parm_dict["max_depth"], 
        max_features = parm_dict["max_features"], 
        n_estimators = parm_dict["n_estimators"], 
        n_jobs = parm_dict["n_jobs"])
    reg_rf.fit(x_train, y_train)
    y_test_predict = reg_rf.predict(x_test)
    print("\nTraining result powered by Random Forest")
    show_Rsquare_and_mse(y_test, y_test_predict)
    print("Model Feature Importance:")
    importances = reg_rf.feature_importances_
    for feature, importance in zip(df.columns, importances):
        print(f"{feature}: {importance}")
    show_importance_map(reg_rf.feature_importances_, df, importance_map_title)
    return y_test, y_test_predict

## Grident Boost

def gradient_boosting(x_train, x_test, y_train, y_test, df, importance_map_title):
    # Hyper Parameter Learning
    learning_rate = 0.1
    hyper_para_learn = 0
    reg_rf = RandomForestRegressor()
    if hyper_para_learn == 1:
        reg_rf_cv = GridSearchCV(reg_rf, {
            'max_depth': [2, 6, 10],
            'n_estimators': [100, 1000, 5000],
            'n_jobs': [24]
        }, verbose=1)
        reg_rf_cv.fit(x_train, y_train)
        parm_dict = reg_rf_cv.best_params_
        print(parm_dict)
        print(reg_rf_cv.best_score_)
    else:
        parm_dict = {}       
        parm_dict["max_depth"] = 10
        parm_dict["n_estimators"] = 5000
        parm_dict["n_jobs"] = 24

    reg_xgb = xgb.XGBRegressor(learning_rate = learning_rate,
                               n_estimators = parm_dict["n_estimators"],
                               max_depth = parm_dict["max_depth"],
                               n_jobs = parm_dict["n_jobs"])
    reg_xgb.fit(x_train, y_train)
    y_test_predict = reg_xgb.predict(x_test)
    print("\nTraining result powered by Grident Boosting")
    show_Rsquare_and_mse(y_test, y_test_predict)
    print("Model Feature Importance:")
    importances = reg_xgb.feature_importances_
    for feature, importance in zip(df.columns, importances):
        print(f"{feature}: {importance}")
    show_importance_map(reg_xgb.feature_importances_, df, importance_map_title)
    return y_test, y_test_predict




def show_predict_accuracy_figure(y_test, y_test_predict, title):
    x = np.linspace(15, 23, 100)
    y = x  
    plt.figure(figsize=(6,6))
    plt.plot(y_test_predict,y_test,marker="o",linestyle="",alpha=0.6,color="black")
    plt.plot(x, y, color = 'red', linestyle = '-')
    plt.title(title,size=16)
    plt.xlabel("prediction",size=16)
    plt.ylabel("true",size=16)


# print("\n\n########## Linear regression result on recent 50000 data powered by Sklearn ##########")
# lr_y_test_recent, lr_y_test_predict_recent = sklearn_linear_learning(recent50000_feature, 
#                                                                 test_feature,
#                                                                 recent50000_label,
#                                                                 test_label, 
#                                                                 random50000_feature_df)

# print("\n\n########## Linear regression result on random 50000 data powered by Sklearn ##########")
# lr_y_test_random, lr_y_test_predict_random = sklearn_linear_learning(random50000_feature, 
#                                                                 test_feature,
#                                                                 random50000_label,
#                                                                 test_label, 
#                                                                 random50000_feature_df)
    
# show_predict_accuracy_figure(lr_y_test_recent, 
#                              lr_y_test_predict_recent, 
#                              "Accuracy of recent 50000 by Linear Regression Powered by Sklearn")
# show_predict_accuracy_figure(lr_y_test_random, 
#                              lr_y_test_predict_random, 
#                              "Accuracy of random 50000 by Linear Regression Powered by Sklearn")

# print("\n\n########## Lasso Linear regression result on recent 50000 data powered by Sklearn ##########")
# las_y_test_recent, las_y_test_predict_recent = lasso_linear_regression(recent50000_feature, 
#                                                                         test_feature,
#                                                                         recent50000_label, 
#                                                                         test_label,
#                                                                         recent50000_feature_df)

# print("\n\n########## Lasso Linear regression result on random 50000 data powered by Sklearn ##########")
# las_y_test_random, las_y_test_predict_random = lasso_linear_regression(random50000_feature, 
#                                                                         test_feature,
#                                                                         random50000_label, 
#                                                                         test_label,
#                                                                         random50000_feature_df)

# show_predict_accuracy_figure(las_y_test_recent, 
#                              las_y_test_predict_recent, 
#                              "Accuracy of recent 50000 by Lasso Linear Regression")
# show_predict_accuracy_figure(las_y_test_random, 
#                              las_y_test_predict_random, 
#                              "Accuracy of random 50000 by Lasso Linear Regression")

print("\n\n########## Lasso Linear regression result on recent 50000 data powered by Statismodel ##########")
OLS_y_test_random, OLS_y_test_predict_random = stats_linear_regression(random50000_feature, 
                                                                        test_feature,
                                                                        random50000_label, 
                                                                        test_label)

print("\n\n########## Lasso Linear regression result on random 50000 data powered by Statismodel ##########")
OLS_y_test_random, OLS_y_test_predict_random = stats_linear_regression(random50000_feature, 
                                                                        test_feature,
                                                                        random50000_label, 
                                                                        test_label)


# print("\n\n########## Linear regression result on recent 50000 data ##########")
# sklearn_linear_learning(recent50000_feature, 
#                         test_feature, 
#                         recent50000_label, 
#                         test_label, 
#                         recent50000_feature_df)
# stats_linear_regression(recent50000_feature, 
#                         test_feature, 
#                         recent50000_label, 
#                         test_label)
# lasso_linear_regression(recent50000_feature, 
#                         test_feature, 
#                         recent50000_label, 
#                         test_label, 
#                         recent50000_feature_df)

# print("\n\n########## Linear regression result on random 50000 data ##########")
# sklearn_linear_learning(random50000_feature, test_feature, random50000_label, test_label, random50000_feature_df)
# stats_linear_regression(random50000_feature, test_feature, random50000_label, test_label)
# lasso_linear_regression(random50000_feature, test_feature, random50000_label, test_label, random50000_feature_df)

print("\n\n########## Random Forest regression result on recent 50000 data ##########")
rf_y_test_recent, rf_y_test_predict_recent = ranfom_forest(recent50000_feature, 
                                                           test_feature, 
                                                           recent50000_label, 
                                                           test_label, 
                                                           recent50000_feature_df, 
                                                           "importance map of recent 50000 powered by random forest")

print("\n\n########## Random Forest regression result on random 50000 data ##########")
rf_y_test_random, rf_y_test_predict_random = ranfom_forest(random50000_feature, 
                                                           test_feature, 
                                                           random50000_label, 
                                                           test_label, 
                                                           random50000_feature_df, 
                                                           "importance map of random 50000 powered by random forest")
# show_predict_accuracy_figure(rf_y_test_recent, 
#                              rf_y_test_predict_recent, 
#                              "Accuracy of recent 50000 by Random Forest")
# show_predict_accuracy_figure(rf_y_test_random, 
#                              rf_y_test_predict_random, 
#                              "Accuracy of random 50000 by Random Forest")

print("\n\n########## Grident Boosting regression result on recent 50000 data ##########")
gb_y_test_recent, gb_y_test__predict_recent = gradient_boosting(recent50000_feature, 
                                                                test_feature, 
                                                                recent50000_label, 
                                                                test_label, 
                                                                recent50000_feature_df, 
                                                                "importance map of recent 50000 powered by Grident Boosting")
print("\n\n########## Grident Boosting regression result on recent 50000 data ##########")
gb_y_test_random, gb_y_test__predict_random = gradient_boosting(random50000_feature, 
                                                                test_feature, 
                                                                random50000_label, 
                                                                test_label, 
                                                                random50000_feature_df, 
                                                                "importance map of random 50000 powered by Grident Boosting")
# show_predict_accuracy_figure(gb_y_test_recent, 
#                              gb_y_test__predict_recent, 
#                              "Accuracy of recent 50000 by Grident Boosting")
# show_predict_accuracy_figure(gb_y_test_random, 
#                              gb_y_test__predict_random, 
#                              "Accuracy of random 50000 by Grident Boosting")



## Figure Generation

# y_test, y_test_predict = sklearn_linear_learning(recent50000_feature, test_feature, recent50000_label, test_label, recent50000_feature_df)
# plt.figure(figsize=(6,5))
# plt.plot(y_test_predict,y_test,marker="o",linestyle="",alpha=0.6,color="black")
# plt.plot(x, y, color = 'red', linestyle = '-')
# plt.title("Prediction Result using the Recent 500000 data",size=16)
# plt.xlabel("prediction",size=16)
# plt.ylabel("true",size=16)

plt.show()




