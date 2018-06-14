
# coding: utf-8

# In[181]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored

data = pd.read_csv('/home/rajarshi/Downloads/Metadata_movie/movie_metadata.csv')
#print data.head(8)
#print data.describe()
#print data.shape
#print data.columns

#finding out the correlation  matrix
corr = data.corr()
print sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

#plotly some graphs based on the information provided by correlation matrix


plt.scatter(data["num_user_for_reviews"],data["num_voted_users"])
plt.xlabel("num_user_for_reviews")
plt.ylabel("num_voted_users")
plt.show()

plt.scatter(data["num_voted_users"], data["imdb_score"])
plt.xlabel("num_voted_users")
plt.ylabel("imdb_score")
plt.show()

data2 = data.ix[data["actor_1_facebook_likes"] <= 100000]
plt.scatter(data2["actor_1_facebook_likes"],data2["cast_total_facebook_likes"])
plt.xlabel("actor_1_facebook_likes")
plt.ylabel("cast_total_facebook_likes")
plt.show()

#some random plots to analyze data
data.genres.nunique()
data3 = data.ix[data["budget"] <= 4e8 ]
data3 = data3.ix[data["title_year"] >= 1980]
plt.scatter(data3["title_year"], data3["budget"])
plt.xlabel("title_year")
plt.ylabel("budget")
plt.show()

#using grouping to find out the actor whose average movie rating is more than 8.5
data_actors = data.groupby('actor_1_name').mean().reset_index()
plt.hist(data["imdb_score"])
plt.xlabel("imdb_score")
plt.show()

data_actor2 = data_actors[data_actors["imdb_score"] >= 8.5]
print colored("Actors with average imdb movie rating more than 8.5", 'red') 
print data_actor2["actor_1_name"]

data_year = data.groupby('title_year').mean().reset_index()
plt.scatter(data_year['title_year'], data_year['duration'])
plt.xlabel("released year ")
plt.ylabel("average_running_time")
plt.show()

data_director = data.groupby('director_name').mean().reset_index()
data_director = data_director[data_director["imdb_score"] >= 8.5]
print colored("Directors with average imdb movie rating more than 8.5", 'red') 
print data_director.director_name

