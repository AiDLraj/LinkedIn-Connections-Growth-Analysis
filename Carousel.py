#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd


# In[3]:


st.write("""# My LinkedIn journey!""")



import pandas as pd
import matplotlib.pyplot as plt
#from plot import plot_weekly_connection, plot_cumsum, plot_violins, plot_bar_column, plot_nlp_cv

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

FIGSIZE = (16,9)
FONT = {"family": "DejaVu Sans", "weight": "normal", "size": 20}
tds = "#0072b1"
week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def plot_weekly_connection(df):
    weekly = df[["added"]].resample("W").sum()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    plt.plot(weekly.index, weekly.added, c=tds)
    

    plt.title("LinkedIn connections evolution", fontdict=FONT)
    plt.ylabel("Nb weekly connections", fontdict=FONT)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)

    ax.set_frame_on(False)
    plt.grid(True)
    #plt.savefig("x.pdf",bbox_inches='tight')
    #plt.show()
    st.pyplot(fig)
    


def plot_cumsum(df):
    cumsum = df.added.cumsum()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    plt.plot(cumsum.index, cumsum.values, c=tds)

    plt.title("LinkedIn connections evolution (cumulated)", fontdict=FONT)
    plt.ylabel("Nb connections", fontdict=FONT)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)

    ax.set_frame_on(False)
    plt.grid(True)
    #plt.savefig("x.pdf",bbox_inches='tight')
    #plt.show()
    st.pyplot(fig)


def violins_prep(tmp):
    tmp = tmp.resample("D").sum()
    tmp = tmp.assign(dow=tmp.index.dayofweek.values).sort_values("dow")
    return tmp.assign(dow_str=tmp.dow.apply(lambda d: week[d]))


def plot_violins(df):
    violins = violins_prep(df[["added"]])
    

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax = sns.violinplot(x="dow_str", y="added", data=violins, color=tds, cut=0, ax=ax)

    plt.title("LinkedIn connections distribution per day of week", fontdict=FONT)
    plt.xlabel("Week day", fontdict=FONT)
    plt.ylabel("Nb daily connections", fontdict=FONT)
    plt.xticks(fontsize= 20)
    plt.yticks(fontsize= 20)
    

    ax.set_frame_on(False)
    plt.grid(True)
    #plt.savefig("x.pdf",bbox_inches='tight')
    #plt.show()
    st.pyplot(fig)
    


def plot_bar_column(df, col):
    fnames = df[col].value_counts().head(30)
    plot_fnames(fnames,col)


def plot_nlp_cv(df):
    tfidf = CountVectorizer(ngram_range=(1, 3), stop_words='english')
    cleaned_positions = list(df["Position"].fillna(""))
    res = tfidf.fit_transform(cleaned_positions)
    res = res.toarray().sum(axis=0)

    fnames = pd.DataFrame(
        list(sorted(zip(res, tfidf.get_feature_names())))[-30:],
        columns=["Position by Words Freq", "Words"]
    )[::-1] 
    plot_fnames(fnames, "Position by Words Freq", "Words")


def plot_fnames(fnames, col, index="index"):
    fnames = fnames.reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    
    plt.bar(
        x=fnames.index,
        height=fnames[col],
        color=tds,
        alpha=0.5
    )

    plt.title("{} distribution".format(col), fontdict=FONT)
    plt.xticks(
        fnames.index,
        fnames[index].str.capitalize(),
        rotation=65,
        ha="right",
        size=FONT["size"],
    )

    plt.ylabel("Nb occurences", fontdict=FONT)
    plt.yticks(fontsize= 20)#[0, 5, 10, 15, 20])
    ax.set_frame_on(False)
    plt.grid(True)
    st.pyplot(fig)
    #plt.savefig("x.pdf",bbox_inches='tight')
    #plt.show()
    


# In[2]:


path = './data/'
file = 'Connections.csv'


# ### Load Connections Data

# In[3]:


df = pd.read_csv(f'{path}{file}')
df['Connected On'] = pd.to_datetime(df['Connected On'])
df.set_index('Connected On', inplace=True, drop=True)
df.sort_index(inplace=True)
df = df.assign(added=1)


# ### Data Exploration

# In[4]:


df.head(1)


# #### Let's now plot my gained weekly connections.

# In[5]:


p1 = plot_weekly_connection(df)


# In[6]:


p2 = plot_cumsum(df)


# In[7]:


p3 = plot_violins(df)


# #### Let's now plot the counts of first names

# In[8]:


p4 = plot_bar_column(df, "First Name")


# #### Let's now plot where my contacts works the most

# In[9]:


p5 = plot_bar_column(df, "Company")


# #### What do my connections do?

# In[10]:


p6 = plot_bar_column(df, "Position")


# In[11]:


p7 = plot_nlp_cv(df)






