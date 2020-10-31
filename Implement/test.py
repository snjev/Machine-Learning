#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:06:32 2019

@author: sanjeevkumar
"""


from bs4 import BeautifulSoup
import requests
import csv

from googlesearch import search
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize


def getTextFromURL(url):
    raw_html = requests.get(url)
    if raw_html.status_code == 200:
        page_text = ''
        soup = BeautifulSoup(raw_html.content, 'html.parser')
        for sent in soup.find_all('p'):
            page_text += sent.get_text() 
        return page_text         
    else:
        print('Error downloding page  '+ url)
        
def extract_words(url):        
    page_text = getTextFromURL(url)
    stop_words = set(stopwords.words('english')).union(set(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '#', '*', '-', '<', '>']))
    word_tokens = [w.lower() for w in word_tokenize(page_text) if not w in stop_words]
    #print(word_tokens)
    return word_tokens

def createDataSetUsingHTML(query):
    documentList = []
    #link_visited = set()
    for link in search(query, tld='com', lang='en', num=20, start=0, stop=1, pause=2.0):
        print(link)
        document = extract_words(link)
        if document is not None:
            documentList.append(document)

    return documentList

query = "what is the name of man who was sitting in garden and An apple dropped on head"    
dataset = createDataSetUsingHTML(query)


with open('datasetOutput.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(dataset)
