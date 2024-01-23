from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix,f1_score,precision_score,recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('heart.csv')

columns_to_encode = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Create an instance of OneHotEncoder
encoder = OneHotEncoder(sparse=False, drop='first')

# Fit and transform the selected columns
one_hot_encoded = encoder.fit_transform(df[columns_to_encode])

# Convert the one-hot encoded array to a DataFrame
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns_to_encode))

# Concatenate the one-hot encoded DataFrame with the original DataFrame
result_df = pd.concat([one_hot_df,df], axis=1)
result_df.drop(columns_to_encode, axis=1,inplace=True)


# dividing inputs and outputs from data
X = df.drop("HeartDisease",axis=1)
y = df["HeartDisease"]

x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.25,shuffle=True)
encoder=LabelEncoder()

x_train["Sex"] = encoder.fit_transform(x_train["Sex"])
x_test["Sex"]=encoder.transform(x_test["Sex"] )
x_train["ChestPainType"] = encoder.fit_transform(x_train["ChestPainType"])
x_test["ChestPainType"]=encoder.transform(x_test["ChestPainType"] )
x_train["RestingECG"] = encoder.fit_transform(x_train["RestingECG"])
x_test["RestingECG"]=encoder.transform(x_test["RestingECG"] )
x_train["ExerciseAngina"] = encoder.fit_transform(x_train["ExerciseAngina"])
x_test["ExerciseAngina"]=encoder.transform(x_test["ExerciseAngina"] )
x_train["ST_Slope"] = encoder.fit_transform(x_train["ST_Slope"])
x_test["ST_Slope"]=encoder.transform(x_test["ST_Slope"] )

# clf = DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# y_test_1 = clf.predict()
# print("Score Test score:", accuracy_score(y_test_1, y_test), " - ", precision_score(y_test_1, y_test), " - ", recall_score(y_test_1, y_test), " - ", f1_score(y_test_1,y_test))

clf = RandomForestClassifier(n_estimators=42,max_depth=3,random_state=10)
clf.fit(x_train,y_train)
print(x_test[1:2])
# y_test_1 = clf.predict(x_test)
# print(y_test_1)
# print(clf.predict([[48, 1, 0, 124, 274, 0, 0, 166, 0, 0.5, 1]]))
y_test_1 = clf.predict(x_test)

import pickle

pickle.dump(clf, open('Health_Model1.sav', 'wb'))