#!/usr/bin/env python
# coding: utf-8

# # Review of Text Visualisation techniques

# In[1]:


##=>   You will find the data in this link : https://catalog.data.gov/dataset/consumer-complaint-database

import pandas as pd # we need this package to manipulate data structues for data analysis and statistics
# upload the data to a pandas dataframe
df = pd.read_csv("MulticlassTextDataset.csv", delimiter=',', dtype='unicode')


# ###########################################   GENERAL INFORMATION   ############################################### 

# In[7]:


df.info() # give a full view about the caracteristics of the dataset: number of rows, columns, rows by columns that are not empty, the memory storage 
print(df.shape) # give the number of rows and number of columns of the dataframe
df.head(2) # give you the look to the first 2 rows of the dataset
df.head(2).T # give you an overview of how data is shown in the file (.T is to transpose the columns so you can see easily the data)


# In[6]:


#deleting the rows with no consumer complanit naratif
df = df[pd.notnull(df['Consumer complaint narrative'])]
print(df.shape)
#give you the nomber of token (words, digit, ponctuation, ... ) splited using the blnak space
df['Consumer complaint narrative'].apply(lambda x: len(x.split(' '))).sum()


# In[8]:



# to count the number of words in a tweet
df['word_count'] = df['Consumer complaint narrative'].apply(lambda x: len(str(x).split())) 
# to count the length of a tweet
df['review_len'] = df['Consumer complaint narrative'].astype(str).apply(len)


# ########################################### DATA EXPLORATION ###############################################

# In[10]:


### Those lines make sure that the df will accept to be a pandas dataframe type instead of being recognized as a series times . in order to use the iplot function ##
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
#####################################################################################################################################################################

#A histogram of  the length of text in the dataset
df['review_len'].iplot(
    kind='hist',
    bins=100,
    xTitle='text length',
    yTitle='count of element',
    title='Review text length distribution')
##################################################
#A histogram of text word count  in the dataset
df['word_count'].iplot(
    kind='hist',
    bins=100,
    xTitle='word count',
    linecolor='black',
    yTitle='count of element',
    title='Review text word count distribution')
##################################################


# In[11]:


##############       Top 20 words in consumer complaint before removing stop words      ##############    
from sklearn.feature_extraction.text import CountVectorizer

def get_top_n_words(corpus, n=None): 
    # this function calculate the frequencies of each word in the dataset and return the n mose
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus) # vectorize the text 
    sum_words = bag_of_words.sum(axis=0)  #use the bag of word technique to segment the text to token
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(df['Consumer complaint narrative'], 20)
#print the words with their frequencies
print("List of the top 20 most frequent words and their frequencies, before removing stop words:")
for word, freq in common_words:
    print(word, freq)
# print a plot showing the frequencies of the top 2 à words in the dataset
df_topw = pd.DataFrame(common_words, columns = ['Text' , 'count'])
df_topw.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in consumer complaint before removing stop words')
#################################################################################################################


# In[12]:


##############       Top 20 words in consumer complaint after removing stop words      ##############   
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = 'english').fit(corpus) # here we extract the predefined stop words present the the package countvectorizer
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(df['Consumer complaint narrative'], 20)
print("List of the top 20 most frequent words and their frequencies, after removing stop words:")
for word, freq in common_words:
    print(word, freq)
df_after_topw = pd.DataFrame(common_words, columns = ['Text' , 'count'])
df_after_topw.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 words in consumer complaint after removing predefined stop words')
#################################################################################################################


# In[14]:


##############       Top 20 bigram (2 consecutive tokens) in consumer complaint before removing stop words      ##############   
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus) # here we tokenize the text into bigram using the ngram technique
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Consumer complaint narrative'], 20)
for word, freq in common_words:
    print(word, freq)
df_top_bi = pd.DataFrame(common_words, columns = ['Text' , 'count'])
df_top_bi.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in consumer complaint before removing stop words')
#################################################################################################################


# In[15]:


##############       Top 20 bigram (2 consecutive tokens) in consumer complaint after removing stop words      ##############   

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Consumer complaint narrative'], 20)
for word, freq in common_words:
    print(word, freq)
df_after_top_bi = pd.DataFrame(common_words, columns = ['Text' , 'count'])
df_after_top_bi.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in consumer complaint after removing stop words')

#################################################################################################################


# In[16]:



def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(1, 2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(df['Consumer complaint narrative'], 20)
for word, freq in common_words:
    print(word, freq)
df_after_top_bi = pd.DataFrame(common_words, columns = ['Text' , 'count'])
df_after_top_bi.groupby('Text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# In[17]:


##############       Distribution of the data by product      ##############   
df.groupby("Product").count()['Complaint ID'].iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Distribution of the data by product', xTitle='product')
#################################################################################################################


# In[18]:


##############       Distribution of the data by issue   ##############  
df.groupby("Issue").count()['Complaint ID'].iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Distribution of the data by issue', xTitle='issue')
#################################################################################################################


# In[25]:


############################        Consumer complaint text length Boxplot by Product        ############################
import plotly.plotly as py
import plotly.graph_objs as go
#######################those lines make sure that the df will accept to be a pandas dataframe type instead of being recognized as a series times . in order to use the iplot function
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)
####################
y0 = df.loc[df['Product'] == 'Virtual currency']['review_len']
y1 = df.loc[df['Product']  == 'Vehicle loan or lease']['review_len']
y2 = df.loc[df['Product']  == 'Student loan']['review_len']
y3 = df.loc[df['Product']  == 'Prepaid card']['review_len']
y4 = df.loc[df['Product']  == 'Payday loan, title loan, or personal loan']['review_len']
y5 = df.loc[df['Product']  == 'Payday loan']['review_len']
y6 = df.loc[df['Product']  == 'Other financial service']['review_len']
y7 = df.loc[df['Product']  == 'Mortgage']['review_len']
y8 = df.loc[df['Product'] == 'Money transfers']['review_len']
y9 = df.loc[df['Product']  == 'Money transfer, virtual currency, or money service']['review_len']
y10 = df.loc[df['Product']  == 'Debt collection']['review_len']
y11 = df.loc[df['Product']  == 'Credit reporting, credit repair services, or other personal consumer reports']['review_len']
y12 = df.loc[df['Product'] == 'Credit reporting']['review_len']
y13 = df.loc[df['Product'] == 'Credit card or prepaid card']['review_len']
y14 = df.loc[df['Product'] == 'Credit card']['review_len']
y15 = df.loc[df['Product'] == 'Consumer Loan']['review_len']
y16 = df.loc[df['Product'] == 'Checking or savings account']['review_len']
y17 = df.loc[df['Product'] == 'Bank account or service']['review_len']
trace0 = go.Box(
    y=y0,
    name = 'Virtual currency',
    marker = dict(
        color = 'rgb(214, 12, 140)',
    )
)
trace1 = go.Box(
    y=y1,
    name = 'Vehicle loan or lease',
    marker = dict(
        color = 'rgb(0, 128, 128)',
    )
)
trace2 = go.Box(
    y=y2,
    name = 'Student loan',
    marker = dict(
        color = 'rgb(10, 140, 208)',
    )
)
trace3 = go.Box(
    y=y3,
    name = 'Prepaid card',
    marker = dict(
        color = 'rgb(0, 0, 2)',
    )
)
trace4 = go.Box(
    y=y4,
    name = 'Payday loan, title loan, or personal loan',
    marker = dict(
        color = 'rgb(12, 102, 14)',
    )
)
trace5 = go.Box(
    y=y5,
    name = 'Payday loan',
    marker = dict(
        color = 'rgb(10, 0, 100)',
    )
)
trace6 = go.Box(
    y=y6,
    name = 'Other financial service',
    marker = dict(
        color = 'rgb(100, 0, 10)',
    )
)

trace7 = go.Box(
    y=y7,
    name = 'Mortgage',
    marker = dict(
        color = 'rgb(255, 12, 140)',
    )
)
trace8 = go.Box(
    y=y8,
    name = 'Money transfers',
    marker = dict(
        color = 'rgb(0, 228, 0)',
    )
)

trace9 = go.Box(
    y=y9,
    name = 'Money transfer, virtual currency, or money service',
    marker = dict(
        color = 'rgb(0, 0, 128)',
    )
)

trace10 = go.Box(
    y=y10,
    name = 'Debt collection',
    marker = dict(
        color = 'rgb(180, 14, 208)',
    )
)
trace11 = go.Box(
    y=y11,
    name = 'Credit reporting, credit repair services, or other personal consumer reports',
    marker = dict(
        color = 'rgb(212, 102, 214)',
    )
)
trace12 = go.Box(
    y=y12,
    name = 'Credit reporting',
    marker = dict(
        color = 'rgb(130, 90, 100)',
    )
)
trace13 = go.Box(
    y=y13,
    name = 'Credit card or prepaid card',
    marker = dict(
        color = 'rgb(100, 70, 100)',
    )
)

trace14 = go.Box(
    y=y14,
    name = 'Credit card',
    marker = dict(
        color = 'rgb(214, 120, 140)',
    )
)
trace15 = go.Box(
    y=y15,
    name = 'Consumer Loan',
    marker = dict(
        color = 'rgb(0, 200, 128)',
    )
)
trace16 = go.Box(
    y=y16,
    name = 'Checking or savings account',
    marker = dict(
        color = 'rgb(10, 14, 208)',
    )
)
trace17 = go.Box(
    y=y17,
    name = 'Bank account or service',
    marker = dict(
        color = 'rgb(120, 102, 14)',
    )
)
data = [trace0, trace1, trace2, trace3, trace4, trace5,trace6, trace7, trace8, trace9, trace10, trace11,trace12, trace13, trace14, trace15, trace16, trace17]
layout = go.Layout(
    title = "Consumer complaint text length Boxplot by Product"
)

fig = go.Figure(data=data,layout=layout).iplot(filename = "Consumer complaint text length Boxplot by Product")
#################################################################################################################


# In[ ]:




