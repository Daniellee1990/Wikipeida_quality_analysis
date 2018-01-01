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
from nltk import word_tokenize,Text,pos_tag
from nltk.stem import SnowballStemmer

be_set = {
          "is",
          "been",
          "are", 
          "was",
          "were",
          "be"
          }

be_not_set = {
            "isn't",
            "aren't",
            "wasn't",
            "weren't"
            }

do_set = {
          "do", 
          "did",
          "does"
          }

do_not_set = {
            "don't",
            "didn't",
            "doesn't"
            }

have_set = {
            "have",
            "had",
            "has"
            }

prepositions_set_one_word = {
                    "aboard", "about", "above", "after", "against", "alongside", 
                    "amid", "among", "around", "at", "before", "behind", "below", 
                    "beneath", "beside", "besides", "between", "beyond", "but", 
                    "concerning", "considering", "despite", "down", "during", 
                    "except", "inside", "into", "like", "off",
                    "onto", "on", "opposite", "out", "outside", "over",
                    "past", "regarding", "round", "since", 
                    "together", "with", "throughout", "through", 
                    "till", "toward", "under", "underneath", "until", "unto", "up", 
                    "up to", "upon", "with", "within", "without", "across", 
                    "along", "by", "of", "in", "to", "near", "of", "from"
                    }

prepositions_set_two_words = {
                    "according to", "across from", "alongside of", "along with",
                    "apart from", "aside from", "away from", "back of", "because of", 
                    "by means of", "down from", "except for", "excepting for", "from among", 
                    "from between", "from under", "inside of", "instead of", "near to", "out of", 
                    "outside of", "over to", "owing to", "prior to", "round about", "subsequent to"
                    }

prepositions_set_three_words = {
                    "in addition to", "in behalf of", "in spite of",
                    "in front of", "in place of", "in regard to",
                    "on account of", "on behalf of", "on top of"
                    }

subordinate_conjunction_set_one_word = {
                                "after", "because", "lest", "till", "’til", "although", 
                                "before", "unless", "as", "provided", "until", "since",
                                "whenever", "if", "than", "inasmuch", "though", "while"
                               }

subordinate_conjunction_set_two_word = {
                                "now that", "even if", "provided that", 
                                "as if", "even though", "so that"
                               }

subordinate_conjunction_set_three_word = {
                                         "as long as", "as much as", 
                                         "as soon as", "in order that"   
                                         }       

def isToBeWord(word):
    if word in be_set or word in be_not_set:
        return True
    return False
    
def SentenceBeginWithSubConj(sentence):
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    if len(tokens) >= 3:
        first_three = tokens[0] + " " + tokens[1] + " " + tokens[2]
        if first_three in prepositions_set_three_words:
            return False
    if len(tokens) >= 2:
        first_two = tokens[0] + " " + tokens[1]
        if first_two in prepositions_set_two_words:
            return False
    for three_words in subordinate_conjunction_set_three_word:
        if sentence.startswith(three_words):
            return True
    for two_words in subordinate_conjunction_set_two_word:
        if sentence.startswith(two_words):
            return True
    if tokens[0] in subordinate_conjunction_set_one_word and tokens[0] not in prepositions_set_one_word:
        return True
    return False

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

def hasAuxiliaryVerb(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    for index, token in enumerate(tokens):
        # if sentence has "don't", "doesn't", "didn't", it must have ausiliary verb
        if token in do_not_set:
            return True
        
        if token in be_set:
            # are done
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # are not done
            if index != len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True
            # are doing
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBG':
                return True
            # are not doing
            if index != len(tokens) - 2 and tags[index + 2][1] == 'VBG':
                return True
            
        if token in be_not_set:
            # aren't done
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # aren't doing
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBG':
                return True
            
        if token in do_set:
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VB':
                return True
            if index != len(tokens) - 2 and tags[index + 2][1] == 'VB':
                return True
            
        if token in have_set:
            # have done
            if index != len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                # has limited powers XXXXXX
                if index != len(tokens) - 2 and tags[index + 2][1] == 'NN' or index != len(tokens) - 2 and tags[index + 2][1] == 'NNS':
                    return False
                else: 
                    return True
            # have not done
            if index !=len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True

    for tag in tags:
        if tag[0] == 'MD':
            return True
    return False
    
def getConjunctionCount(sentence):
    conjunction_number = 0
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)  
    for tag in tags:
        if tag[1] == "CC" or tag[1] == "IN":
            conjunction_number = conjunction_number + 1
    return conjunction_number

def getPrepositionCount(sentence):
    preposition_number = 0
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)  
    for tag in tags:
        if tag[1] == "IN":
            preposition_number = preposition_number + 1
    return preposition_number

def getPronounCount(sentence):
    pronoun_number = 0
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    for tag in tags:
        if tag[1] == "PRP" or tag[1] == "PRP$":
            pronoun_number = pronoun_number + 1
    return pronoun_number
        
def SentenceBeginWithConj(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    if tags[0][1] == "CC" or tags[0][1] == "IN":
        return True
    return False

def SentenceBeginWithPrep(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    if tags[0][1] == "IN" and tags[0][0].lower() not in subordinate_conjunction_set_one_word:
        return True
    return False

def SentenceBeginWithInterrogativePronoun(sentence):
    tokens = word_tokenize(sentence)
    interrogative_pronoun = {'What', 'Which', 'Who', 'Whom', 
                             'Whose', 'Whatever', 'Whatsoever',
                             'Whichever','Whoever','Whosoever', 
                             'Whomever', 'Whomsoever', 'Whosever',
                             'Why', 'where', 'When', 'How'
                             }
    if tokens[0] in interrogative_pronoun:
        return True
    return False

def SentenceBeginWithPronoun(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    if tags[0][1] == "PRP" or tags[0][1] == "PRP$":
        return True
    return False 

def isWordNominalization(word):
    if len(word) <= 4:
        return False
    suffix = word[-4:]
    nominalization_suffix = {'tion', 'ment', 'ence', 'ance'}
    snow = SnowballStemmer('english') 
    stem = snow.stem(word)
    # check whether it is suffix. For example, 'ance' in France is not suffix. 
    if suffix in nominalization_suffix and len(word) - len(stem) >= 3:
        return True
    return False

def SentencePassiveVoice(sentence):
    tokens = word_tokenize(sentence)
    text = Text(tokens)
    tags = pos_tag(text)
    for index, token in enumerate(tokens):
        if token in be_set:
            # are done
            if index < len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # are not done
            if index < len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True
            # are not properly done
            if index < len(tokens) - 3 and tags[index + 3][1] == 'VBN':
                return True
        if token in be_not_set:
            # isn't done
            if index < len(tokens) - 1 and tags[index + 1][1] == 'VBN':
                return True
            # isn't properly done
            if index < len(tokens) - 2 and tags[index + 2][1] == 'VBN':
                return True
    return False
    
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

#### Get the paragraph related features such as average paragraph length, paragraph numbers##
### compute the average length of paragraphs in sentences ###
paragraphs = f.read().split("\n\n")
title2paraNums = dict()
title2paraLen = dict()
para_num = 0
total_para_len = 0
cur_title = ''
last_title = ''
para_sizes = 0
body_start = False
# preprocess the paragraph
for index, parag in enumerate(paragraphs):
    if '</doc>' in parag and '<doc' in parag:
        temp = parag.split('\n')
        newParag = list()
        for sent in temp:
            if sent == '':
                continue
            newParag.append(sent)
        paragraphs[index] = newParag[0]
        for i in range(1, 3):
            paragraphs.insert(index + i, newParag[i])
            
processed_parags = list()
for parag in paragraphs:
    if parag == '':
        continue
    processed_parags.append(parag)

first_line_in_body = False        
for parag in processed_parags:
    # if it is the start of body
    if '<doc' in parag:
        body_start = True
        first_line_in_body = True
        continue
    # if it is the end of body
    if '</doc>' in parag:
        body_start = False
        title2paraNums[cur_title] = para_num
        title2paraLen[cur_title] = float(total_para_len) / para_num
        total_para_len = 0
        para_num = 0
        continue
    if first_line_in_body == True:
        cur_title = parag
        first_line_in_body = False 
    # if the paragraph is in the body
    if body_start == True and parag != "" and '<doc' not in parag and '</doc>' not in parag:
        para_num = para_num + 1
        temp_para = parag
        sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_tokenize.tokenize(temp_para)
        total_para_len = total_para_len + len(sentences)

##### Compute other features.
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
question_rates = list()
question_nums = list()
Article_sentence_rates = list()
Auxiliary_verb_rates = list()
Conjunction_rates = list()
Conjunction_sent_rates = list()
Interrogative_pronoun_sent_rates = list()
nominalization_word_rates = list()
Passive_voice_sent_rates = list()
Preposition_rates = list()
Preposition_sent_rates = list()
Pronoun_sent_rates = list()
Pronoun_rates = list()
SubConj_sent_rates = list()
to_be_word_rates = list()

bodies = bodies[-200:]
for body in bodies:
    chars_number = 0  
    bodystr = ""
    for paragraph in body:
        bodystr = bodystr + paragraph 
        
    #### get the text statistics
    # get word related features
    # get character count
    temps = re.split('\,|\.|\ |\n|\:|\;',bodystr)
    chars = [c for c in temps if c.isalpha()]
    counts = Counter(chars)
    words_count = len(chars)
    syllables_num = 0
    syllables_sum = 0
    complex_word_number = 0
    long_word_number = 0
    one_syllable_word_cnt = 0
    number_nominalization_word = 0
    number_to_be_word = 0
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
        if isWordNominalization(char) == True:
            number_nominalization_word = number_nominalization_word + 1
        if isToBeWord(char) == True:
            #print(char)
            number_to_be_word = number_to_be_word + 1
        
    average_word_length = float(chars_number) / words_count
    long_word_rate = float(long_word_number) / words_count
    one_syllable_word_rate = float(one_syllable_word_cnt) / words_count
    average_syllable_num = float(syllables_sum) / words_count
    complex_word_rate = float(complex_word_number) / words_count
    nominalization_word_rate = float(number_nominalization_word) / words_count
    to_be_word_rate = float(number_to_be_word) / words_count

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
    nominalization_word_rates.append(nominalization_word_rate)
    to_be_word_rates.append(to_be_word_rate)
    
    # get sentences related features.
    # split body into sentences
    sent_tokenize = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_tokenize.tokenize(bodystr)
    question_sent_number = 0
    sentence_number_with_article = 0
    sentence_with_auxiliary_verb = 0
    conjunction_number = 0
    number_sent_begin_with_InterrogativePronoun = 0
    preposition_number = 0
    pronoun_number = 0
    for sentence in sentences:
    # find the number of question sentences    
        if '?' in sentence:
            question_pos = sentence.find('?')
            # if "?" is not at end of sentence, do not count it.
            if question_pos != len(sentence) - 1:
                continue
            question_sent_number = question_sent_number + 1
    # find the number of sentences starting with article         
        if sentence.startswith('A ') or sentence.startswith('An ') or sentence.startswith('The '):
            sentence_number_with_article = sentence_number_with_article + 1
    # find the number of Auxiliary Verbs
        if hasAuxiliaryVerb(sentence) == True:
            #print(sentence + "\n")
            sentence_with_auxiliary_verb = sentence_with_auxiliary_verb + 1
    # find conjunction rate
        current_conj_number = getConjunctionCount(sentence)
        conjunction_number = conjunction_number + current_conj_number
    # Get proposition rate
        preposition_number = preposition_number + getPrepositionCount(sentence)
    # Get pronoun rate
        pronoun_number = pronoun_number + getPronounCount(sentence)
    sum_len = 0
    sent_num = len(sentences)
    max_length = 0
    min_length = 10000
    large_sent_number = 0
    short_sent_number = 0
    average_sent_len = 0
    number_sent_begin_with_conjuction = 0
    number_passive_voice_sent = 0
    number_sent_begin_with_preposition = 0
    number_sent_begin_witn_pronoun = 0
    number_sent_begin_with_subConj = 0
    
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
        if True == SentenceBeginWithConj(sentence):
            number_sent_begin_with_conjuction = number_sent_begin_with_conjuction + 1
        if True == SentenceBeginWithInterrogativePronoun(sentence):
            number_sent_begin_with_InterrogativePronoun = number_sent_begin_with_InterrogativePronoun + 1
        if True == SentencePassiveVoice(sentence):
            #print("Passive")
            #print(sentence + "\n")
            number_passive_voice_sent = number_passive_voice_sent + 1
        if True == SentenceBeginWithPrep(sentence):
            #print("Prep")
            #print(sentence + "\n")
            number_sent_begin_with_preposition = number_sent_begin_with_preposition + 1  
        if True == SentenceBeginWithPronoun(sentence):
            #print("Pronoun")
            #print(sentence + "\n")
            number_sent_begin_witn_pronoun = number_sent_begin_witn_pronoun + 1
        if True == SentenceBeginWithSubConj(sentence):
            #print("Sub conj")
            #print(sentence + "\n")
            number_sent_begin_with_subConj = number_sent_begin_with_subConj + 1

    average_sent_len = float(sum_len) / sent_num
    large_sent_rate = float(large_sent_number) / sent_num
    short_sent_rate = float(short_sent_number) / sent_num
    question_rate = float(question_sent_number) / sent_num
    Article_sentence_rate = float(sentence_number_with_article) / sent_num
    Auxiliary_verb_rate = float(sentence_with_auxiliary_verb) / sent_num
    Conjunction_rate = float(conjunction_number) / words_count
    Preposition_rate = float( preposition_number ) / words_count
    Conjunction_sent_rate = float(number_sent_begin_with_conjuction) / sent_num
    Interrogative_pronoun_sent_rate = float(number_sent_begin_with_InterrogativePronoun) / sent_num
    Passive_voice_sent_rate = float(number_passive_voice_sent) / sent_num
    Preposition_sent_rate = float(number_sent_begin_with_preposition) / sent_num
    Pronoun_sent_rate = float(number_sent_begin_witn_pronoun) / sent_num
    SubConj_sent_rate = float(number_sent_begin_with_subConj) / sent_num
    Pronoun_rate = float(pronoun_number) / words_count
    
    max_sent_lengths.append(max_length)
    min_sent_lengths.append(min_length)
    large_sent_rates.append(large_sent_rate)
    short_sent_rates.append(short_sent_rate)
    average_sent_lens.append(average_sent_len)
    question_rates.append(question_rate)
    question_nums.append(question_sent_number)
    Article_sentence_rates.append(Article_sentence_rate)
    Auxiliary_verb_rates.append(Auxiliary_verb_rate)
    Conjunction_rates.append(Conjunction_rate)
    Preposition_rates.append(Preposition_rate)
    Conjunction_sent_rates.append(Conjunction_sent_rate)
    Interrogative_pronoun_sent_rates.append(Interrogative_pronoun_sent_rate)
    Passive_voice_sent_rates.append(Passive_voice_sent_rate)
    Preposition_sent_rates.append(Preposition_sent_rate)
    Pronoun_sent_rates.append(Pronoun_sent_rate)
    Pronoun_rates.append(Pronoun_rate)
    SubConj_sent_rates.append(SubConj_sent_rate)
    
    #print('Max sentence length is ' + str(max_length))
    #print('Min sentence length is ' + str(min_length))
    #print('large sentence rate is ' + str(large_sent_rate))
    #print('short sentence rate is ' + str(short_sent_rate))
    #print('Average sentence length is ' + str(average_sent_len)) 
    
## get part of speech features
## Article sentence rate
    
    
    
     
        
        
        

    
        