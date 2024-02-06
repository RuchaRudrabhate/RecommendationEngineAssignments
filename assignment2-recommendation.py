# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 13:12:51 2023

@author: arudr
"""

'''
Problem Statement: -

The Entertainment Company, which is an online movie watching 
platform, wants to improve its collection of movies and showcase 
those that are highly rated and recommend those movies to its
 customer by their movie watching footprint. For this, the company 
 has collected the data and shared it with you to provide some 
 analytical insights and also to come up with a recommendation 
 algorithm so that it can automate its process for effective 
 recommendations. The ratings are between -9 and +9.
'''
#Business objective
#minimize - wrong recommendation 
#maximize - number views for movies by recommendations
#costraints - accuracy of predictions and recommendations

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/5-Recommendation/Assignment recommendation/Entertainment.csv') 

df

df.describe()

'''
                Id    Reviews
count    51.000000  51.000000
mean   6351.196078  36.289608
std    2619.679263  49.035042
min    1110.000000  -9.420000
25%    5295.500000  -4.295000
50%    6778.000000   5.920000
75%    8223.500000  99.000000
max    9979.000000  99.000000
'''
#data dictionary
df.dtypes

'''
columns   |   data type |       type        |       relevence 

Id            int64         quantitative            irrelevent
Titles       object         categorical             relevent
Category     object         categorical,nominal     relevent
Reviews     float64         quantitative            relevent
'''
df.columns
# Index(['Id', 'Titles', 'Category', 'Reviews'], dtype='object')

df = df.drop(['Id'],axis = 1)
df.columns
#Index(['Titles', 'Category', 'Reviews'], dtype='object')

#pdf and cdf for reviews column
counts, bin_edges = np.histogram(df['Reviews'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#from pdf
'''
from pdf we can say that for about  40 % data reviews decreases 
remains constant for 10 % of data
and afterwards it increases
'''
#outliers treatment

sns.boxplot(df['Reviews'])

#there are no outliers

df.describe()
'''
        Reviews
count  51.000000
mean   36.289608
std    49.035042
min    -9.420000
25%    -4.295000
50%     5.920000
75%    99.000000
max    99.000000

'''


### recommendation system for entertainment dataset

df.to_csv('C:/5-Recommendation/Assignment recommendation/assignment_entertainment.csv')


df = pd.read_csv('C:/5-Recommendation/Assignment recommendation/assignment_entertainment.csv',encoding = 'utf8')
df.shape
df.columns
df = df.drop(['Unnamed: 0'],axis=1)
df.columns
df.shape
#(51, 3)
from sklearn.feature_extraction.text import TfidfVectorizer
#This is term frequency inverse document
#each row is treated as doc
tfidf = TfidfVectorizer(stop_words = 'english')
#it is going to create TfidfVectorizer to separate all stop words
#it ios going to separate
#out all words from the row
#now let us check is there any null value
df['Titles'].isnull().sum()
#there are 0 null value so no need to treat them

tfidf_matrix=tfidf.fit_transform(df.Titles)
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
#this is for measuring simarity 
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared
#with each element of tfidf_matrix only

df_index =  pd.Series(df.index,index = df['Titles']).drop_duplicates()
df_id = df_index['Toy Story (1995)']
df_id

def get_recommendations(name,TopN):
    df_id = df_index[name]
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    #the cosine scores captured we want to arrange in descending order
    cosine_scores_N=cosine_scores[0:TopN+1]
    #get the scores top N similar movies
    #to capture top n movies you need to give
    #To capture topN movies,you need to give topN+1
    df_idx=[i[0] for i in cosine_scores_N]
    df_scores=[i[1] for i in cosine_scores_N]
    movie_similar_show = pd.DataFrame(columns = ['Titles','Reviews'])
    movie_similar_show['Titles'] = df.loc[df_idx,'Titles']
    #assign score to score col
    movie_similar_show['Reviews'] = df_scores
    #while assigning value it is by default capturing original index of the 
    #we want to reset the index
    movie_similar_show.reset_index(inplace = True)
    print(movie_similar_show)
    
    
get_recommendations('Usual Suspects, The (1995)', 10)

