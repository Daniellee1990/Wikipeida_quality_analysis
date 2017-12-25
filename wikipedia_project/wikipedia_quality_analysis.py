#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 16:28:55 2017

@author: lixiaodan
"""
import pandas as pd 
import re
from collections import Counter
import nltk

### https://stackoverflow.com/questions/405161/detecting-syllables-in-a-word
def countSyllables(word):
    vowels = "aeiouy"
    numVowels = 0
    lastWasVowel = False
    for wc in word:
        foundVowel = False
        for v in vowels:
            if v == wc:
                if not lastWasVowel: 
                    numVowels += 1   #don't count diphthongs
                    foundVowel = True
                    lastWasVowel = True
                    break
                else:
                    foundVowel = True
                    lastWasVowel = True
                    break
        if not foundVowel:  #If full cycle and no vowel found, set lastWasVowel to false
            lastWasVowel = False
    if len(word) > 2 and word[-2:] == "es": #Remove es - it's "usually" silent (?)
        numVowels-=1
    elif len(word) > 1 and word[-1:] == "e":    #remove silent e
        numVowels-=1
    if numVowels == 0:
        numVowels = 1
    return numVowels

data_FA = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/Featured_articles.csv', encoding='latin-1')
data_GA = pd.read_csv('/Users/lixiaodan/Desktop/wikipedia_project/dataset/Good_articles.csv')

# combine featured articles and good articles as the benign dataset
FA = data_FA['Article']
FA_set = set(FA)
GA = data_GA['Article']
GA_set = set(GA)
good_article_names = FA_set | GA_set

## input the texts
f = open('/Users/lixiaodan/Desktop/wikipedia_project/dataset/AA.txt')
bodies = list()
labels = list()
body = list()
titles = list()
flag = False

###### compute the number of paragraphs ######################
paragraphs = f.read().split("\n\n")
title2paraNums = dict()
para_num = 0
cur_title = ''
body_start = False
for parag in paragraphs:
    if '</doc>' in parag:
        body_start = False
        title2paraNums[cur_title] = para_num
        para_num = 0
    if body_start == True and parag != "":
        para_num = para_num + 1
    if '<doc' in parag:
        temp = parag.split("\n")
        cur_title = temp[-1]
        body_start = True

f = open('/Users/lixiaodan/Desktop/wikipedia_project/dataset/AA.txt')        
### process the paragraphs and transform them into lines        
for line in f:
    if line == '\n':
        continue
    if line.startswith('<doc'):
        flag = True
        continue
    if line.startswith('</doc'):
        flag = False
    if flag == True:
        body.append(line)
    if flag == False:
        if len(body) > 1:    
            bodies.append(body)
            body = list()
        
for cont in bodies:
    title = cont[0]
    del cont[0] # delete titles in body.
    titles.append(title)
    if title in good_article_names:
        labels.append(1)
    else:
        labels.append(0)

#### find the FA and add its documents to texts
chars_numbers = list()
average_word_lengths = list()
complex_word_rates = list()
words_counts = list()
long_word_rates = list()
one_syllable_word_rates = list()
average_syllable_nums = list()
average_sent_lens = list()
large_sent_rates = list()
short_sent_rates = list()
average_sent_lens = list()
max_sent_lengths = list()
min_sent_lengths = list()
syllables_sums = list()

#bodies = bodies[:1]
#### get the content features
for body in bodies:
    chars_number = 0  
    bodystr = ""
    for paragraph in body:
        bodystr = bodystr + paragraph 
    # get word related features
    # get character count
    temps = re.split('\,|\.|\ |\n|\:|\;',bodystr)
    chars = [c for c in temps if c.isalpha()]
    counts = Counter(chars)
    syllables_num = 0
    syllables_sum = 0
    complex_word_number = 0
    long_word_number = 0
    one_syllable_word_cnt = 0
    for char in chars:
        chars_number = chars_number + len(char)
        if len(char) >= 6:
            long_word_number = long_word_number + 1
        syllables_num = countSyllables(char)
        syllables_sum = syllables_sum + syllables_num
        if syllables_num == 1:
            one_syllable_word_cnt = one_syllable_word_cnt + 1
        if syllables_num >= 3:
            complex_word_number = complex_word_number + 1
    words_count = len(chars)
    average_word_length = float(chars_number) / words_count
    long_word_rate = float(long_word_number) / words_count
    one_syllable_word_rate = float(one_syllable_word_cnt) / words_count
    average_syllable_num = float(syllables_sum) / words_count
    complex_word_rate = float(complex_word_number) / words_count
    # add character count into features
    chars_numbers.append(chars_number)
    # add average word length into features
    average_word_lengths.append(average_word_length)
    # get the complex word rate
    complex_word_rates.append(complex_word_rate)
    words_counts.append(words_count)
    long_word_rates.append(long_word_rate)
    one_syllable_word_rates.append(one_syllable_word_rate)
    average_syllable_nums.append(average_syllable_num)
    
    # get sentences related features.
    # split body into sentences
    sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenize.tokenize(bodystr)
    sum_len = 0
    sent_num = len(sentences)
    max_length = 0
    min_length = 10000
    large_sent_number = 0
    short_sent_number = 0
    average_sent_len = 0
    for sentence in sentences:
        sent_len = len(sentence)
        sum_len = sum_len + sent_len
        if sent_len > max_length:
            max_length = sent_len
        if sent_len < min_length:
            min_length = sent_len
        if sent_len >= 30:
            large_sent_number = large_sent_number + 1
        if sent_len <= 15:
            short_sent_number = short_sent_number + 1
    average_sent_len = float(sum_len) / sent_num
    large_sent_rate = float(large_sent_number) / sent_num
    short_sent_rate = float(short_sent_number) / sent_num
    max_sent_lengths.append(max_length)
    min_sent_lengths.append(min_length)
    large_sent_rates.append(large_sent_rate)
    short_sent_rates.append(short_sent_rate)
    average_sent_lens.append(average_sent_len)
    #print('Max sentence length is ' + str(max_length))
    #print('Min sentence length is ' + str(min_length))
    #print('large sentence rate is ' + str(large_sent_rate))
    #print('short sentence rate is ' + str(short_sent_rate))
    #print('Average sentence length is ' + str(average_sent_len))  
     
        
        
        

    
        