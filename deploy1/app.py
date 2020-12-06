import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import plotly.express as px
from os import path
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import warnings
from pickle import dump, load
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

import nltk



dataset_loc = "SMSSpamCollection"
image_loc = "img/spam_img.png"

########################################## Side bar of the webapp ##########################################
def load_sidebar():
    st.sidebar.subheader("SPAM Message Detection")

    st.sidebar.success("Analyse wheather the message is SPAM or HAM .")

    st.sidebar.info("The dataset contains both Spam and Ham messages and we train this dataset\
        and make Predictions to get perfect analysis of the messages.")

    # Radio button for choices in sidebar
    choice=st.sidebar.radio("Enter your choice: ",('Data Description','Exploratory Data Analysis','Model Evaluation','Predictions'))
    return choice

# Dataset loading ####################################################################################################################
def load_data(dataset_loc):
    df=pd.read_csv(dataset_loc,sep='\t',names=['target','message'])
    df['length'] = df['message'].apply(len)
    return df


# description ###########################################################################################################################
def load_description(df):
    st.image(image_loc, use_column_width = True)
    st.header("DATA-SET Overview")

    if(st.checkbox("About the Data-set")):
        st.info('''The use of mobile phones has skyrocketed in the last decade leading to a new area for junk promotions from disreptable marketers. People innocently give out their mobile phone numbers while utilizing day to day services and are then flooded with spam promotional messages. ''')
        st.success('''In this Data-set, we have total 3 columns having the information as:\n
1) target :- Shows the type of message.(Ham :smile: or Spam :angry:)\n
2) text :- show the exact message .\n
3) Length :- denotes the length of the text .\n''')

    # display the whole dataset
    if(st.checkbox("Show Dataset")):
        h_t=st.radio("",('COMPLETE','TOP','BOTTOM'))
        if h_t =='TOP':
            st.table(df.head())
        elif h_t=='COMPLETE':
            st.write(df)
        else:
            st.table(df.tail())

    # Show shape
    if(st.checkbox("Shape of Data-set")):
        dim = st.radio("Rows/Columns?", ("Rows", "Columns"))
        if(dim == "Rows"):
            st.write("Number of Rows", df.shape[0])
        if(dim == "Columns"):
            st.write("Number of Columns", df.shape[1])

    # show value counts
    if(st.checkbox('Counts of Unique values')):
        st.table(df.target.value_counts())

    # show info    
    if(st.checkbox("Data-set Description")):
        st.table(df.describe(include='object'))
    
    # group-by
    if(st.checkbox('Describe using Group Labels')):
        st.table((df.groupby('target').describe(include='object')).T)


# Plots ###################################################################################################################3333333
def graph(df):
    st.header("DATA-SET Visualization")
    
    dim = st.radio("chhose any one graph below :", ("Bar Graph", "Pie Chart","Word-Cloud"))
    if(dim == "Bar Graph"):
        sns.countplot(x="target", data=df)
        plt.xlabel("TARGET")
        plt.ylabel("FREQUENCY")
        plt.title(" BAR PLOT OF : TARGET")
        st.pyplot(plt)

        if(st.checkbox("OBSERVATION 1.1 :")):
            st.write('''
                1. Tagret column have 2 unique vaules (ham and spam).
                2. There are 4825 (86.6%) of ham messages.
                3. There are 747 (13.4%) of spam messages.       
                ''')        

        st.subheader("Histogram of 'HAM' and 'SPAM' with respect to Length :")
        df.hist(column='length',by='target', bins=50)
        st.pyplot(plt)

        if(st.checkbox("OBSERVATION 1.2 :")):
            st.write('''
                1. Looks like spam messages are generally longer than ham messages.
                2. Bulk of ham has length below 100, for spam it is above 100.
                3. We will check if this feature is useful for the classification task.
                ''')       

    elif(dim == "Pie Chart"):
        plt.title(" PIE CHART OF : TARGET")
        df["target"].value_counts().plot(kind = 'pie', explode = [0, 0.1], autopct = '%1.1f%%', shadow = True)
        plt.ylabel("Spam vs Ham")
        plt.legend(["Ham", "Spam"])
        st.pyplot(plt)
        if(st.checkbox("OBSERVATION 2.1 :")):
            st.write('''
                1. Looks like spam messages are generally longer than ham messages.
                2. Bulk of ham has length below 100, for spam it is above 100.
                3. We will check if this feature is useful for the classification task.
                ''')  

    else:
        word_cloud(df)

# word cloud ##################################################################################################################  
def word_cloud(df):
    st.subheader("Treating 'SPAM / HAM' messages")
    dim = st.radio("Spam/Ham?", ("Spam", "Ham"))

    if(dim == "Ham"):
        df_ham = df.loc[df['target']=='ham', :]
        st.write(df_ham.head())                                           
        if (path.exists("img/wc_ham.png")):
            st.image("img/wc_ham.png", use_column_width = True)

    else:
        df_spam = df.loc[df['target']=='spam', :]
        st.write(df_spam.head())
        if (path.exists("img/wc_spam.png")):
            st.image("img/wc_spam.png", use_column_width = True)


# clean word #####################################################################################################################
def cleaned(df_sms):

# Join all messages to make one paragraph. 
  words_ = ' '.join(df_sms['message'])
  
# change all data into lower case.
  word_=words_.lower()

#  removes all word like(https,www.)
  c_word = " ".join([word for word in word_.split()
                            if 'http' not in word
                         and 'www.' not in word
                            ])
  
# removes all special characters and digits.
  word_sms=''
  letters_only_sms = re.sub("[^a-zA-Z]", " ",c_word)
 
# removes all stopwords like (the,we,are,it,if......)
  words = letters_only_sms.split()
  words = [w for w in words if not w in stopwords.words("english")]
  
# removes all words which have length less than 2.
  for a in words:
    if len(a)<3:
      words.remove(a)

# again make all words into paragraph.
  for i in words:
    word_sms=word_sms+" "+i

# return that paragraph.
  return word_sms


# preprocess ###############################################################################################################33333
def preprocess(raw_msg):

    stemmer = PorterStemmer()
    # Removing words like (http,www.)
    cleaned = " ".join([word for word in raw_msg.split()
                            if 'http' not in word
                          and 'www.' not in word
                            ])

    # Removing special characters and digits
    letters_only = re.sub("[^a-zA-Z]", " ",cleaned)

    # change sentence to lower case
    letters_only = letters_only.lower()

    # tokenize into words
    words = letters_only.split()
    
    # remove stop words                
    words = [w for w in words if not w  in stopwords.words("english")]

    # Stemming
    words = [stemmer.stem(word) for word in words]

    clean_sent = " ".join(words)
    
    return clean_sent

def Model():
    st.header("Model")
    choice=st.radio("Enter your ML model: ",('Logistic Regression','Decision Tree','Support Vector Classifier'))
    
    if choice=="Logistic Regression":
        st.header("Logistic Regression")
        st.subheader("Confusion Matrix")
        st.image("img/LRC.png", use_column_width = True)
        st.subheader("Precision Table")
        st.image("img/LR.png", use_column_width = True)
        
    elif choice=="Decision Tree":
        st.header("Decision Tree")
        st.subheader("Confusion Matrix")
        st.image("img/DTC.png", use_column_width = True)
        st.subheader("Precision Table")
        st.image("img/DT.png", use_column_width = True)
        
    else:
        st.header("Support Vector Classifier")
        st.subheader("Confusion Matrix")
        st.image("img/SVCC.png", use_column_width = True)
        st.subheader("Precision Table")
        st.image("img/SVC.png", use_column_width = True)
        
# predict #####################################################################################################333
def predict(msg):
    
    # Loading pretrained CountVectorizer from pickle file
    vectorizer = load(open('pickle/countvectorizer.pkl', 'rb'))
    
    # Loading pretrained logistic classifier from pickle file
    classifier = load(open('pickle/logit_model.pkl', 'rb'))
    
    # Preprocessing the tweet
    clean_msg = preprocess(msg)
    
    # Converting text to numerical vector
    clean_msg_encoded = vectorizer.transform([clean_msg])
    
    # Converting sparse matrix to dense matrix
    msg_input = clean_msg_encoded.toarray()
    
    # Prediction
    prediction = classifier.predict(msg_input)
    
    return prediction

# Tese ##########################################################################################
def test():

    st.header("Prediction")
    st.image(image_loc, use_column_width = True)
    msg = st.text_input('Enter your Message : ')

    prediction = predict(msg)

    if(msg):
        st.subheader("Prediction:")
        if(prediction == 0):
            st.warning("SPAM MESSAGE :angry: ")
        else:
            st.success("HAM MESSAGE :smile: ")



# Main ###############################################################################################
def main():
    
    st.title('SMS Spam Collection Data Set')
   
    choice=load_sidebar()
    df = load_data(dataset_loc)
    if choice=='Data Description':
        load_description(df)
    elif choice=='Exploratory Data Analysis':
        graph(df)
    elif choice=='Model Evaluation':
        Model()
    else:
        test()

#### calling main ########################
if(__name__ == '__main__'):
    main()
