#!/usr/bin/env python
# coding: utf-8

# In[237]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[238]:


df = pd.read_csv("BacheloretteDSFinal-Dogu.csv")


# In[239]:


df.head()


# In[240]:


df.describe()


# In[241]:


df.info()


# In[242]:


df.isnull().sum()


# In[243]:


df.drop_duplicates(inplace=True)


# In[244]:


df.isnull().sum()


# In[245]:


df[df["Season"].isnull()]


# In[246]:


df.dropna(how='all',inplace=True)


# In[247]:


df.shape


# In[248]:


df.isnull().sum()


# In[249]:


df[df["College"].isnull()]


# In[250]:


#since the person whose college is missing belong to TX state we're looking for colleges of other participants from TX
df[df.values=='TX']


# In[251]:


df["College"].fillna(value="University of Texas at Austin",inplace=True)#filling missing value with other value


# In[252]:


df.isnull().sum()


# In[253]:


df[df["Height (cm)"].isnull()]#checking the missing values in height column


# In[254]:


df["Height (cm)"].value_counts()


# In[255]:


df["Height (cm)"].fillna(value=180.80,inplace=True)


# In[256]:


df["Height (cm)"].value_counts()


# In[257]:


df.isnull().sum()


# In[258]:


df["Season"].unique()   # to see how many seasons the data covers


# In[259]:


df[df['Win_Loss']==1]


# In[260]:


# replace outcome with "Yes" or "No"

df['Win_Loss'] = df['Win_Loss'].astype(object)
df.replace({'Win_Loss':{'1.0':'Yes','0.0':'No'}},inplace=True)


# In[261]:


#cleaning up a little 

df['Age'] = df["Age"].astype('int')
df['Height (cm)'].round(decimals=2)


# In[262]:


print(df['Hometown'].nunique())
print(df['Occupation'].nunique())
print(df['College'].nunique())
print(df['State'].nunique())


# In[263]:


print('In seasons 11-15, there were {:,} unique contestants. {:,} contestants have appeared in more than one season.'.format(df['Name'].nunique(), len([x for x in df['Name'].value_counts() if x > 1])))


# In[264]:


print('In seasons 11-15, there were {:,} unique hometowns. {:,} hometowns have appeared multiple times.'.format(df['Hometown'].nunique(), len([x for x in df['Hometown'].value_counts() if x > 1])))


# In[265]:


#Takes a while to load, Heat Map of the US
perstate = df[df['State'] != '']['State'].value_counts().to_dict()

data = [dict(
        type = 'choropleth',
        autocolorscale = False,
        colorscale = 'Reds',
        reversescale = True,
        locations = list(perstate.keys()),
        locationmode = 'USA-states',
        text = list(perstate.values()),
        z = list(perstate.values()),
        marker = dict(
            line = dict(
                color = 'rgb(255, 255, 255)',
                width = 2)
            ),
        )]

layout = dict(
         title = 'Bachelorette Contestants by State',
         geo = dict(
             scope = 'usa',
             projection = dict(type = 'albers usa'),
             countrycolor = 'rgb(255, 255, 255)',
             showlakes = True,
             lakecolor = 'rgb(255, 255, 255)')
         )

figure = dict(data = data, layout = layout)
iplot(figure)


# In[266]:


#simple histogram for all ages; season 11-15

df['Age'].hist(bins=15, color='DarkRed')


# In[267]:


#Gives min age
df['Age'].min()


# In[268]:


#Gives the counts for each eye color
df['Eye Color'].value_counts()


# In[269]:


#Gives the percentage of each eye color
total_eye_count= df['Eye Color'].value_counts().sum()
partial_eye_count =  df['Eye Color'].value_counts()
for i in partial_eye_count:
    eye_percentage= partial_eye_count/total_eye_count
    eye_percentage= eye_percentage* 100
print(eye_percentage)


# In[270]:


#Donut plot for eye color distributions
#Creates a pie chart for the different eye colors
names='Brown','Blue','Green'
size=[85.106383,9.929078,4.964539]
plt.pie(size, labels=names, colors=['sienna','skyblue','lightgreen'])

#Creates a white circle for the center of the plot
#Makes pie chart become a donut plot
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()


# In[271]:


df['Hair Color'].value_counts()


# In[272]:


#Gives the percentage of each hair color
total_hair_count= df['Hair Color'].value_counts().sum()
partial_hair_count= df['Hair Color'].value_counts()
for i in partial_hair_count:
    hair_percentage= partial_hair_count/total_hair_count
    hair_percentage= hair_percentage* 100
print(hair_percentage)


# In[273]:


#Donut Plot for Hair Color Distributions
#Creates pie chart with specific hair colors
names='Brown','Blonde'
size=[94.326241,5.673759]
plt.pie(size, labels=names, colors=['sienna','gold'])

# Creates a white circle for the center of the plot
#Makes pie chart become a donut plot
my_circle=plt.Circle( (0,0), 0.7, color='white')
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.show()
plt.savefig('DonutPlotHair')


# In[274]:


#Finds the mean height in cm, this could have been done with .mean() but the specific steps were shown to demonstrate understanding of the process
totalheight= df['Height (cm)'].sum()
count_of_height = df['Height (cm)'].value_counts().sum()
average = (totalheight)/(count_of_height)
print(average)


# In[275]:


#Takes the height in cm and returns it in ft and in
def to_inch(x):
    ft_raw = 0.0328084*x
    ft = int(ft_raw)
    rem = ft_raw-ft
    inches = round(rem*12,2)
    print('The average contestant is',ft,'feet and', inches,'inches tall.')
    
to_inch(180.9)


# In[276]:


#Counts the repitition of  heights
df['Height (cm)'].value_counts()


# In[277]:


from sklearn.cluster import KMeans


# In[278]:


df1=df.iloc[:,[2,8]]


# In[279]:


sse=[]


# In[280]:


for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df1)
    sse.append(kmeans.inertia_)


# In[281]:


sse


# In[286]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1,11),sse,marker='o',color='red')
plt.title("Elbow method")
plt.xlabel("K")
plt.ylabel("SSE")
plt.show()


# In[287]:


kmeans=KMeans(4)
clusters=kmeans.fit_predict(df1)
df["Clusters"]=clusters
df


# In[288]:


#print the cluster groups
gb = df[['Name','Clusters']].groupby('Clusters')
for key, item in gb:
    print(gb.get_group(key), "\n\n")


# In[289]:


plt.figure(figsize=(15,7))
sns.scatterplot(df1[clusters==0].iloc[:,0],df1[clusters==0].iloc[:,1],color='yellow',label='cluster 1',s=50)
sns.scatterplot(df1[clusters==1].iloc[:,0],df1[clusters==1].iloc[:,1],color='blue',label='cluster 2',s=50)
sns.scatterplot(df1[clusters==2].iloc[:,0],df1[clusters==2].iloc[:,1],color='green',label='cluster 3',s=50)
sns.scatterplot(df1[clusters==3].iloc[:,0],df1[clusters==3].iloc[:,1],color='grey',label='cluster 4',s=50)
sns.scatterplot(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='red',label='centroid', s=300,marker=",")
plt.grid(False)
plt.title("Visualizing Clusters")
plt.xlabel("Age")
plt.ylabel("Height in cms")
plt.legend()
plt


# In[290]:


import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[291]:


df.head()


# In[292]:


df[['Age', 'Win_Loss']].groupby(['Age'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[293]:


df[["State", "Win_Loss"]].groupby(['State'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[294]:


df[['College', 'Win_Loss']].groupby(['College'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[295]:


df[['Hair Color', 'Win_Loss']].groupby(['Hair Color'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[296]:


df[['Eye Color', 'Win_Loss']].groupby(['Eye Color'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[297]:


df[['Occupation', 'Win_Loss']].groupby(['Occupation'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[298]:


df[['Height (cm)', 'Win_Loss']].groupby(['Height (cm)'], as_index=False).mean().sort_values(by='Win_Loss', ascending=False)


# In[299]:


g = sns.FacetGrid(df, col='Win_Loss')
g.map(plt.hist, 'Age', bins=20)


# In[300]:



grid = sns.FacetGrid(df, col='Win_Loss', row='Height (cm)', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[301]:



grid = sns.FacetGrid(df, col='Win_Loss', row='Hair Color', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[302]:



grid = sns.FacetGrid(df, col='Win_Loss', row='Eye Color', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()


# In[303]:



grid = sns.FacetGrid(df, row='Eye Color', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Age', 'Win_Loss', 'Hair Color', palette='deep')
grid.add_legend()


# In[304]:



grid = sns.FacetGrid(df, row='Eye Color', col='Win_Loss', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Hair Color', 'Height (cm)', alpha=.5, ci=None)
grid.add_legend()


# In[305]:


df = df.drop(['Occupation', 'Clusters'], axis=1)


# In[306]:


df['Hair Color'] = df['Hair Color'].map( {'Brown': 1, 'Blonde': 0} ).astype(int)

df.head()


# In[307]:


grid = sns.FacetGrid(df, row='Age', col='Hair Color', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Height (cm)', alpha=.5, bins=20)
grid.add_legend()


# In[308]:


guess_height = np.zeros((2,3))
guess_height


# In[309]:


df.loc[ df['Age'] <= 20, 'Age'] = 0
df.loc[(df['Age'] > 20) & (df['Age'] <= 25), 'Age'] = 1
df.loc[(df['Age'] > 25) & (df['Age'] <= 30), 'Age'] = 2
df.loc[(df['Age'] > 30) & (df['Age'] <= 35), 'Age'] = 3
df.loc[ df['Age'] > 35, 'Age']
df.head()


# In[310]:


df.loc[ df['Eye Color'] == 'Brown', 'Eye Color'] = 0
df.loc[ df['Eye Color'] == 'Blue', 'Eye Color'] = 1
df.loc[ df['Eye Color'] == 'Green', 'Eye Color'] = 2
df.head()


# In[311]:


df.loc[ df['Girlfriend While on the Show?'] == 'No', 'Girlfriend While on the Show?'] = 0
df.loc[ df['Girlfriend While on the Show?'] == 'Yes', 'Girlfriend While on the Show?'] = 1


# In[312]:


df.head()


# In[314]:


df.loc[ df['Height (cm)'] < 175.00, 'Height (cm)'] = 0
df.loc[ (df['Height (cm)'] >= 175.00) & (df['Height (cm)'] <180.00), 'Height (cm)'] = 1
df.loc[ (df['Height (cm)'] >= 180.00) & (df['Height (cm)'] <185.00), 'Height (cm)'] = 2
df.loc[ (df['Height (cm)'] >= 185.00) & (df['Height (cm)'] <190.00), 'Height (cm)'] = 3
df.loc[ df['Height (cm)'] >= 190.00 , 'Height (cm)'] = 3
df.head()


# In[317]:


df.loc[ df['Height (cm)'] == 0.0, 'Height (cm)'] = '0'
df.loc[ df['Height (cm)'] == 1.0, 'Height (cm)'] = '1'
df.loc[ df['Height (cm)'] == 2.0, 'Height (cm)'] = '2'
df.loc[ df['Height (cm)'] == 3.0, 'Height (cm)'] = '3'
df.head()


# In[318]:


df.loc[ df['Height (cm)'] == '0', 'Height (cm)'] = 0
df.loc[ df['Height (cm)'] == '1', 'Height (cm)'] = 1
df.loc[ df['Height (cm)'] == '2', 'Height (cm)'] = 2
df.loc[ df['Height (cm)'] == '3', 'Height (cm)'] = 3
df.head()


# In[319]:


df.loc[ df['Win_Loss'] == 0.0, 'Win_Loss'] = '0'
df.loc[ df['Win_Loss'] == 0.0, 'Win_Loss'] = '0'
df.head()


# In[320]:


df.loc[ df['Win_Loss'] == 1.0, 'Win_Loss'] = '1'


# In[321]:


df.loc[ df['Win_Loss'] == '0', 'Win_Loss'] = 0
df.loc[ df['Win_Loss'] == '1', 'Win_Loss'] = 1
df.head()


# In[322]:


df = df.drop(['Name', 'Hometown','State','College'], axis=1)
df.head()


# In[323]:


df.loc[ df['Season'] == 11.0, 'Season'] = '0'
df.loc[ df['Season'] == 12.0, 'Season'] = '1'
df.loc[ df['Season'] == 13.0, 'Season'] = '2'
df.loc[ df['Season'] == 14.0, 'Season'] = '3'
df.loc[ df['Season'] == 15.0, 'Season'] = '4'
df.head()


# In[324]:


df.loc[ df['Season'] == '0', 'Season'] = 0
df.loc[ df['Season'] == '1', 'Season'] = 1
df.loc[ df['Season'] == '2', 'Season'] = 2
df.loc[ df['Season'] == '3', 'Season'] = 3
df.loc[ df['Season'] == '4', 'Season'] = 4
df.head()


# In[325]:





# In[332]:


x=[]
for i in range(0,141):
    x.append(i)
df['no.']=x
df.head()


# In[337]:


X_train = df.drop("Win_Loss", axis=1)
Y_train = df["Win_Loss"]
X_test  = df.drop("no.", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[340]:


from sklearn.linear_model import LogisticRegression


# In[348]:


convert_dict = {'Season': int, 'Age': int,'Height (cm)': int,'Girlfriend While on the Show?': int, 'Hair Color':int,'Eye Color':int,'no.':int}
df = df.astype(convert_dict)


# In[349]:


convert_dic={'Win_Loss':int}
df=df.astype(convert_dic)


# In[350]:


df.info()


# In[351]:


X_train = df.drop("Win_Loss", axis=1)
Y_train = df["Win_Loss"]
X_test  = df.drop("no.", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


# In[354]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
acc_log


# In[355]:


coeff_df = pd.DataFrame(df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)


# In[368]:


# Support Vector Machines

svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 3)
acc_svc


# In[367]:


knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 3)
acc_knn


# In[366]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 3)
acc_gaussian


# In[364]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 3)
acc_perceptron


# In[363]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 3)
acc_linear_svc


# In[369]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
acc_sgd


# In[370]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
acc_decision_tree


# In[371]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest


# In[372]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[375]:


submission = pd.DataFrame({
        "no.": df["no."],
        "Win_Loss": Y_pred
    })
submission.to_csv("submission2.csv", index=False)


# In[ ]:




