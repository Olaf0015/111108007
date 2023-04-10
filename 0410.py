import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
train.head(10)
train.info()

total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['缺失數', '缺失率'])
missing_data

train.Name = train.Name.fillna('Unknown')
print(train.isnull().sum())

train['CryoSleep']=train['CryoSleep'].fillna(False)
train['VIP']=train['VIP'].fillna(False)
print(train.isnull().sum())

train['Age']=train['Age'].fillna(train['Age'].mean())
print(train.isnull().sum())

train['RoomService']=train['RoomService'].fillna(train['RoomService'].mean())
train['FoodCourt']=train['FoodCourt'].fillna(train['FoodCourt'].mean())
train['ShoppingMall']=train['ShoppingMall'].fillna(train['ShoppingMall'].mean())
train['Spa']=train['Spa'].fillna(train['Spa'].mean())
train['VRDeck']=train['VRDeck'].fillna(train['VRDeck'].mean())
print(train.isnull().sum())

analys = train.loc[:,['HomePlanet','Destination']]
analys['numeric'] =1
analys.groupby(['Destination','HomePlanet']).count()

train['Destination']=train['Destination'].fillna('TRAPPIST-1e')
train['HomePlanet']=train['HomePlanet'].fillna('Earth')
print(train.isnull().sum())

plt.figure(figsize=(10, 5))
sns.histplot(data=train, x='Age', binwidth=1, kde=True)
plt.title('Age distribution')
plt.xlabel('Age (years)');

fig, ax = plt.subplots(nrows=2, ncols=2)
fig.set_size_inches(18, 12)
fig.subplots_adjust(wspace=0.3, hspace=0.3)
temp = train.fillna(-1)
sns.barplot(x = "HomePlanet", y= "Transported", data=temp, ax = ax[0][0])
sns.barplot(x = "CryoSleep", y= "Transported", data=temp, ax = ax[0][1])
sns.barplot(x = "VIP", y= "Transported", data=temp, ax = ax[1][0])
sns.barplot(x = "Destination", y= "Transported", data=temp, ax = ax[1][1])

corrmat = train.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot=True)
plt.show()

corrmat = train.corr()
k = 6 
high_corr_values = corrmat.nlargest(k, 'Transported')['Transported'].index
high_corr_values = high_corr_values.drop('Transported')
high_corr_values

from sklearn.model_selection import train_test_split
X = train[high_corr_values]
y = train['Transported']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
hgbc = HistGradientBoostingClassifier()
hgbc.fit(X_train,y_train)
hgbc_pred = hgbc.predict(X_test)
print("Hist gradient boosting accuracy: {}".format(accuracy_score(y_test,hgbc_pred)))

test_ids = test["PassengerId"]
from sklearn.preprocessing import LabelEncoder
categorical_values_test = test.select_dtypes(include=['object']).columns

for i in categorical_values_test:
    lbl = LabelEncoder()
    lbl.fit(list(test[i].values))
    test[i] = lbl.transform(list(test[i].values))

real_predictions = hgbc.predict(test[high_corr_values])
test["PassengerId"] = test_ids
real_predictions = list(map(bool,real_predictions))
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Transported': real_predictions})
output.to_csv('submission.csv', index=False)