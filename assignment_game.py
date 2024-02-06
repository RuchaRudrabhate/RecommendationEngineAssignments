# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 15:42:12 2023

@author: arudr
"""
'''
Q) Build a recommender system with the given data using UBCF.

This dataset is related to the video gaming industry and a survey was
conducted to build a recommendation engine so that the store can
 improve the sales of its gaming DVDs. Snapshot of the dataset is 
 given below. Build a Recommendation Engine and suggest top selling 
 DVDs to the store customers.

'''
#Business objective

#maximize --> increase the sales of the store by recommending games to users
#minimize --> minimize wrong or recommendations
#constraints-->accuracy of predictions and recommendations
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/5-Recommendation/Assignment recommendation/game.csv') 

df

df.describe()

#data dictionary
df.dtypes
'''
name of  |   datatype of  |      Type   |   relevence 
feature  |   feature      |             |

userId      int64           quantitative    irrelevent
game       object           nominal         relevent
rating    float64           quantitative    relevent
'''

df = df.drop(['userId'],axis=1)
df.columns

df.describe() #only rating col is of int type
'''
            rating
count  5000.000000
mean      3.592500
std       0.994933
min       0.500000
25%       3.000000
50%       4.000000
75%       4.000000
max       5.000000
'''
#pdf and cdf for rating column
counts, bin_edges = np.histogram(df['rating'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

##inference
'''
from cdf we say that about 60 % of data or games have rating 
between 3.5 to 4.0 , between 1 to 2% of games have rating between 1 to 1.5
most of the data have rating value= 5
'''

#finding outliers
import seaborn as sns
sns.boxplot(df['rating'])
##have outliers
iqr = df['rating'].quantile(0.75)-df['rating'].quantile(0.25)
iqr
q1=df['rating'].quantile(0.25)
q3=df['rating'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
df['rating'] =  np.where(df['rating']>u_limit,u_limit,np.where(df['rating']<l_limit,l_limit,df['rating']))
sns.boxplot(df['rating'])

#now there are no outliers
#describe data set
df.describe()
'''         rating
count  5000.000000
mean      3.611100
std       0.947991
min       1.500000
25%       3.000000
50%       4.000000
75%       4.000000
max       5.000000'''

#dataset is consist of values between 1 to 5 so no need to normalise data

df

#now data is perfect for recommendation sys for games

df.to_csv('C:/5-Recommendation/Assignment recommendation/assignment_game.csv')

df = pd.read_csv('C:/5-Recommendation/Assignment recommendation/assignment_game.csv',encoding = 'utf8')
df.shape
df.columns
df = df.drop(['Unnamed: 0'],axis=1)
# 3 rows and 5000 cols
from sklearn.feature_extraction.text import TfidfVectorizer
#This is term frequency inverse document
#each row is treated as doc
tfidf = TfidfVectorizer(stop_words = 'english')
#it is going to create TfidfVectorizer to separate all stop words
#it ios going to separate
#out all words from the row
#now let us check is there any null value
df['game'].isnull().sum()
#there are 0 null value so no need to treat them

tfidf_matrix=tfidf.fit_transform(df.game)
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
#this is for measuring simarity 
cosine_sim_matrix = linear_kernel(tfidf_matrix,tfidf_matrix)
#each element of tfidf_matrix is compared
#with each element of tfidf_matrix only

df_index =  pd.Series(df.index,index = df['game']).drop_duplicates()
df_id = df_index['Tony Hawk\'s Pro Skater 2']
df_id

def get_recommendations(name,TopN):
    df_id = df_index[name]
    cosine_scores = list(enumerate(cosine_sim_matrix[df_id]))
    cosine_scores = sorted(cosine_scores, key = lambda x:x[1], reverse = True)
    #the cosine scores captured we want to arrange in descending order
    cosine_scores_N=cosine_scores[0:TopN+1]
    #get the scores top N similar games
    #to capture top n movies you need to give
    #To capture topN movies,you need to give topN+1
    df_idx=[i[0] for i in cosine_scores_N]
    df_scores=[i[1] for i in cosine_scores_N]
    game_similar_show = pd.DataFrame(columns = ['game','rating'])
    game_similar_show['game'] = df.loc[df_idx,'game']
    #assign score to score col
    game_similar_show['rating'] = df_scores
    #while assigning value it is by default capturing original index of the 
    #we want to reset the index
    game_similar_show.reset_index(inplace = True)
    print(game_similar_show)
    
    
get_recommendations('The Legend of Zelda: Ocarina of Time', 10)
