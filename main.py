#!/usr/bin/env python
# coding: utf-8

# ## Importing data and segregating data as per brands

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/deepaknarravula/Downloads/AdidasVsNike.csv')

df = pd.DataFrame(data)
adidas_data = df[df['Brand'].str.contains('Adidas', case=False)].copy()
nike_data = df[df['Brand'].str.contains('Nike', case=False)].copy()

adidas_data
df.info()

# In[2]:


adidas_data.describe()

# ## Products that have less number of reviews ( 0 to 10 )

# In[3]:


adidas_data.query('0 <= `Reviews` <= 10 ')

# ### Converting 'Price' from INR to USD

# In[4]:


df[['Listing Price', 'Sale Price']] = df[['Listing Price', 'Sale Price']] * 0.012

# In[5]:


nike_data[['Listing Price', 'Sale Price']] = nike_data[['Listing Price', 'Sale Price']] * 0.012

# In[6]:


adidas_data[['Listing Price', 'Sale Price']] = adidas_data[['Listing Price', 'Sale Price']] * 0.012
adidas_data

# In[7]:


adidas_data['Brand'].value_counts()

#
# As seen above, we can observe a typo - 'Adidas Adidas ORIGINALS' which needs to be corrected to - 'Adidas ORIGINALS'
#

# In[8]:


adidas_data['Brand'].iloc[0] = 'Adidas ORIGINALS'

# In[9]:


adidas_neo = df[df['Brand'].str.contains('NEO', case=False)].copy()
adidas_originals = df[df['Brand'].str.contains('ORIGINALS', case=False)].copy()
adidas_performance = df[df['Brand'].str.contains('PERFORMANCE', case=False)].copy()

# In[10]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

columns = ['Rating', 'Reviews', 'Listing Price', 'Sale Price', 'Discount']
scaled_data = adidas_data[columns].copy()

scaled_data = scaler.fit_transform(scaled_data)
scaled_data = pd.DataFrame(scaled_data, columns=columns)
scaled_data

# In[11]:


import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.histplot(adidas_data['Reviews'], bins=30, kde=False, color='skyblue')
plt.title('Distribution of Reviews in Adidas Data')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.show()

# This plot gives us an idea as to what is the most common number of ratings on products


# In[12]:


adidas_data['Reviews'].mean()

# In[13]:


plt.figure(figsize=(10, 6))
sns.histplot(adidas_data['Rating'], bins=30, kde=False, color='skyblue')
plt.title('Distribution of ratings in Adidas Data')
plt.xlabel('Ratings (0-5)')
plt.ylabel('Frequency')
plt.show()

# This plot gives us an idea that majority of the products have got ratings in 2-5 range, with lesser products with lower ratings


# In[14]:


adidas_data['Rating'].mean()

# In[15]:


(
    so.Plot(data=adidas_data,
            x='Rating',
            y='Reviews', )
    .add(so.Bar(), so.Agg())
)


# As seen here, the number of reviewes are consistent over the different ranges of ratings(0-5). We can also observe that most of the products have received ratings between 2-5


# In[16]:


def plot(data, color, price_column='Sale Price', bins=[0, 50, 100, 150, 200, 250, 300, float('inf')],
         labels=['0-50', '51-100', '101-150', '151-200', '201-250', '251-300', '301-350']):
    price_ranges = pd.cut(data[price_column], bins=bins, labels=labels)
    price_counts = price_ranges.value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    plt.bar(price_counts.index, price_counts.values, color=color)
    plt.xlabel('Price Range')
    plt.ylabel('Number of Products')
    plt.title('Number of Products in Each Price Range')
    plt.show()


# In[17]:


plot(adidas_performance, 'red')

# In[18]:


plot(adidas_originals, 'purple')

# In[19]:


plot(adidas_neo, 'green')

# In[20]:


plot(nike_data, 'blue')

# ## Feature Engineering
#
# We are using 'Rating' and 'Review' columns to train our model and predict whether a particular product is successful/will likely be removed. On exploring both the features, we intend to build a rule that would classify products into - 3 categories(Successful, Needs Attention, Unsuccessful)
#
# For 'Reviews':
# Mean: 48.72
# Based on the distribution above, we can observe that most reviews are in the 45+ range ( i.e most of the products have got 45+ reviews )
#
# For 'Ratings':
# Mean: 3.36
# Based the distribution for ratings, we can observe most products lie in the 2-5 rating range
#
# Rule formed based on above insights:
# if product 'Reviews' > 45 and 'Rating' > 3.2,
#     product= Successful
# else if
# if product 'Reviews' < 45 and 'Rating' > 3.2,
#     product = Successful
# else if
# if product 'Reviews' > 45 and 'Rating' < 3.2,
#     product= Needs Attention
# else
#     product = Unsuccessful
#
#
# The logic for the labels:
#
# Successful - Products with high reviews and high ratings is most likely successful. Products with high ratings and lesser reviews can still mean it is successful but not as popular
#
# Needs Attention - The products with low ratings but high reviews, indicating there could be problem areas/room for improvement since there are multiple users but the product has issues
#
# Unsuccessful - Products with low reviews and low ratings generally indicate lesser users and issues in the product hence unsuccessful

# In[21]:


adidas_data.query('`Reviews` > = 45 and `Rating`<= 3.2')

# In[22]:


adidas_data.query('`Reviews` < = 45 and `Rating`>= 3.2')

# ## Creating labels based on logic

# In[23]:


for index, row in adidas_data.iterrows():
    if row['Reviews'] > 45 and row['Rating'] > 3.2:
        adidas_data.at[index, 'product status'] = 'Successful'
    elif row['Reviews'] < 45 and row['Rating'] > 3.2:
        adidas_data.at[index, 'product status'] = 'Successful'
    elif row['Reviews'] > 45 and row['Rating'] < 3.2:
        adidas_data.at[index, 'product status'] = 'Needs Attention'
    else:
        adidas_data.at[index, 'product status'] = 'Unsuccessful'

# In[24]:


# Data is ready to be modeled
adidas_data

# In[25]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

clf = LogisticRegression()
features = ['Rating', 'Reviews']
X = adidas_data[features]
y = adidas_data['product status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7238)

# In[26]:


clf.fit(X_train, y_train)

# In[27]:


y_pred = clf.predict(X_test)

# In[28]:


model_results = pd.DataFrame({'y_pred': y_pred, 'y_true': y_test})
model_results.head(20)

# In[29]:


metrics.accuracy_score(y_test, y_pred)

# In[30]:


# Assuming 'your_data' is your DataFrame
last_visited_data = pd.to_datetime(nike_data['Last Visited'])
last_visited_df = pd.DataFrame(last_visited_data)

last_visited_df

# In[31]:


last_visited_df['Second'] = last_visited_df['Last Visited'].dt.strftime('%S')
last_visited_df['Minute'] = last_visited_df['Last Visited'].dt.strftime('%M')
last_visited_df

visit_plot_data = last_visited_df.groupby('Minute').size().reset_index(name='Count')
visit_plot_data

# In[32]:


plt.plot(visit_plot_data['Minute'], visit_plot_data['Count'])
plt.xlabel('Minute')
plt.ylabel('Count')
plt.title('Line Plot of Visit Counts Over Time')
# plt.xticks(range(0, 46, 2))

plt.show()

# In[33]:


testing = last_visited_df.set_index('Last Visited')
testing

# In[34]:


plt.figure(figsize=(10, 6))
testing.resample('T').count().plot()

# In[35]:


df.corr()

# In[36]:


mask = np.tril(df.corr())
sns.heatmap(df.corr(), cmap='coolwarm', annot=True, mask=mask)

# High corelation is observed in 'Listing Price' and 'Sale Price'

# In[37]:


# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()

# columns_to_scale = ['Rating', 'Reviews', 'Listing Price', 'Sale Price', 'Discount']
# scaled_data = adidas_data[columns_to_scale].copy()

# scaled_data = scaler.fit_transform(scaled_data)


# In[38]:


columns = ['Rating', 'Reviews']
numeric_data = scaled_data[columns]
numeric_data

# In[39]:


numeric_data['Reviews'].std()

# In[40]:


numeric_data['Rating'].std()

# In[41]:


from sklearn.cluster import KMeans

inertia = []

for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=123)
    kmeans.fit(numeric_data)
    inertia.append([k, kmeans.inertia_])

inertia

# In[42]:


(
    pd.DataFrame(inertia, columns=['k', 'inertia'])
    .pipe(so.Plot, x='k', y='inertia')
    .add(so.Line())
    .add(so.Dot())
)

# Choosing k =3

# In[43]:


# from sklearn.metrics import silhouette_score

# silhouette = []

# for k in range(2,15):

#     kmeans = KMeans(n_clusters = k, random_state = 123)
#     kmeans.fit(numeric_data)
#     silhouette.append([k, silhouette_score(numeric_data, kmeans.labels_)])

# silhouette


# In[44]:


max_iter = 1

kmeans = KMeans(n_clusters=3, random_state=9854, max_iter=max_iter, n_init=1, init='random')

kmeans.fit(numeric_data)

kmeans.labels_

# In[45]:


cluster_names = {0: 'Successful', 1: 'Unsuccessful', 2: 'Needs Attention'}

(
    numeric_data.assign(labels=kmeans.labels_.astype(str))
    .pipe(so.Plot, x='Rating', y='Reviews', color='labels')
    .add(so.Dot())
    .label(title='Clustering Results'
           ))

# In[46]:


max_iter = 2
kmeans = KMeans(n_clusters=3, random_state=9854, max_iter=max_iter, n_init=2, init='random')
kmeans.fit(numeric_data)

# Add cluster labels to your DataFrame
numeric_data['Cluster'] = kmeans.labels_

# Create a dictionary mapping cluster labels to names
cluster_names = {0: 'Successful', 1: 'Cluster B', 2: 'Not Successful'}

# Use seaborn to plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=numeric_data, x='Rating', y='Reviews', hue='Cluster', palette='viridis', s=100)
plt.title('KMeans Clustering')

legend_labels = [cluster_names[label] for label in sorted(numeric_data['Cluster'].unique())]
plt.legend(title='Cluster', labels=legend_labels, loc='upper left', bbox_to_anchor=(1, 0.75))

plt.show()

