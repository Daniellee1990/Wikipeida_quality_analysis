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
import readability_formulas
import text_stat_pos
    
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
aris = list() ## automated readability index
BIs = list() ## Bormuth index
CLIndexes = list() ## Coleman Liau index
eduYears = list() ## education years
GunningFogIndexes = list() ## Gunning fog index
FlecshReadings = list() ## Flesch reading ease
FlecshKincaids = list() ## Flesch Kincaids
LIXes = list() ## Lasbarhedsindex
MiyazakiEFLs = list() ## Miyazaki EFL readability index
newDaleChalls = list() ## new dale chall
smogGradings = list() ## smog grading
common_word_rates = list()
difficult_word_rates = list()
Peacock_word_rates = list()
Stop_word_rates = list()
weasel_word_rates = list()

#bodies = bodies[0:50]
for body in bodies:
    chars_number = 0  
    bodystr = ""
    for paragraph in body:
        bodystr = bodystr + paragraph 
    #print(bodystr)
    #print("\n")
    
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
    number_difficult_word = 0
    number_peacock_word = 0
    number_stop_word = 0
    number_weasel_word = 0
    
    for char in chars:
        chars_number = chars_number + len(char)
        #print("chars_number")
        #print(chars_number)
        if len(char) >= 6:
            long_word_number = long_word_number + 1
        syllables_num = text_stat_pos.countSyllables(char)
        syllables_sum = syllables_sum + syllables_num
        if syllables_num == 1:
            one_syllable_word_cnt = one_syllable_word_cnt + 1
        if syllables_num >= 3:
            complex_word_number = complex_word_number + 1
        if text_stat_pos.isWordNominalization(char) == True:
            number_nominalization_word = number_nominalization_word + 1
        if text_stat_pos.isToBeWord(char) == True:
            number_to_be_word = number_to_be_word + 1
        if readability_formulas.isDifficultWord(char) == True:
            number_difficult_word = number_difficult_word + 1
        if readability_formulas.isPeacockWord(char) == True:
            number_peacock_word = number_peacock_word + 1
            #print("Peacock word")
            #print(char)
        if readability_formulas.isStopWord(char) == True:
            number_stop_word = number_stop_word + 1 
            #print("Stop word")
            #print(char)
        if readability_formulas.isWeaselWord(char) == True:
            number_weasel_word = number_weasel_word + 1 
            #print("weasel word")
            #print(char)
        
    average_word_length = float(chars_number) / words_count
    long_word_rate = float(long_word_number) / words_count
    one_syllable_word_rate = float(one_syllable_word_cnt) / words_count
    average_syllable_num = float(syllables_sum) / words_count
    complex_word_rate = float(complex_word_number) / words_count
    nominalization_word_rate = float(number_nominalization_word) / words_count
    to_be_word_rate = float(number_to_be_word) / words_count
    difficult_word_rate = float(number_difficult_word) / words_count
    common_word_rate = 1 - difficult_word_rate
    stop_word_rate = float(number_stop_word) / words_count

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
    common_word_rates.append(common_word_rate)
    difficult_word_rates.append(difficult_word_rate)
    Stop_word_rates.append(stop_word_rate)
    
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
        if text_stat_pos.hasAuxiliaryVerb(sentence) == True:
            #print(sentence + "\n")
            sentence_with_auxiliary_verb = sentence_with_auxiliary_verb + 1
        if readability_formulas.hasPeacockPhrase(sentence) == True:
            number_peacock_word = number_peacock_word + 1
            #print(sentence)
        if readability_formulas.hasWeaselPhrase(sentence) == True:
            number_weasel_word = number_weasel_word + 1
            #print(sentence)
            
    # find conjunction rate
        current_conj_number = text_stat_pos.getConjunctionCount(sentence)
        conjunction_number = conjunction_number + current_conj_number
    # Get proposition rate
        preposition_number = preposition_number + text_stat_pos.getPrepositionCount(sentence)
    # Get pronoun rate
        pronoun_number = pronoun_number + text_stat_pos.getPronounCount(sentence)
        
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
        if True == text_stat_pos.SentenceBeginWithConj(sentence):
            number_sent_begin_with_conjuction = number_sent_begin_with_conjuction + 1
        if True == text_stat_pos.SentenceBeginWithInterrogativePronoun(sentence):
            number_sent_begin_with_InterrogativePronoun = number_sent_begin_with_InterrogativePronoun + 1
        if True == text_stat_pos.SentencePassiveVoice(sentence):
            #print("Passive")
            #print(sentence + "\n")
            number_passive_voice_sent = number_passive_voice_sent + 1
        if True == text_stat_pos.SentenceBeginWithPrep(sentence):
            #print("Prep")
            #print(sentence + "\n")
            number_sent_begin_with_preposition = number_sent_begin_with_preposition + 1  
        if True == text_stat_pos.SentenceBeginWithPronoun(sentence):
            #print("Pronoun")
            #print(sentence + "\n")
            number_sent_begin_witn_pronoun = number_sent_begin_witn_pronoun + 1
        if True == text_stat_pos.SentenceBeginWithSubConj(sentence):
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
    Peacock_word_rate = float(number_peacock_word) / words_count
    weasel_word_rate = float(number_weasel_word) / words_count
    
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
    Peacock_word_rates.append(Peacock_word_rate)
    weasel_word_rates.append(weasel_word_rate)
    
    ################### Readability Formulas ##################################
    # Automated readability index
    #print(sentences)
    """
    print("characters number")
    print(chars_number)
    print("\n")
    print("words count")
    print(words_count)
    print("\n")
    print("sentence number")
    #sent_num = len(sentences)
    print(sent_num)
    print("\n")
    """
    ari = readability_formulas.getARI(chars_number, words_count, sent_num)
    #print("Automated readability index")
    #print(ari)
    aris.append(ari)
    #print("Difficult word count \n")
    #print(number_difficult_word)
    BI = readability_formulas.BormuthIndex(chars_number, words_count, sent_num, number_difficult_word)
    BIs.append(BI)
   # if BI >= 0:
    #    print(BI)
    
    CLIndex = readability_formulas.getColemanLaurIndex(chars_number, words_count, sent_num)
    #print("Coleman-liau index")
    #print(CLIndex)
    CLIndexes.append(CLIndex)
    
    # forcast readability
    eduYear = readability_formulas.getEduYears(one_syllable_word_cnt)
    eduYears.append(eduYear)
    #print("Forcast readability")
    #print(eduYear)
    
    # Flesch reading ease
    FlecshReading = readability_formulas.getFleschReading(words_count, sent_num, one_syllable_word_cnt)
    FlecshReadings.append(FlecshReading)
    #print("Flesch Reading ease")
    #print(FlecshReading)
    
    # Flesch Kincaid
    FlecshKincaid = readability_formulas.getFleschKincaid(words_count, sent_num, one_syllable_word_cnt)
    FlecshKincaids.append(FlecshKincaid)
    #print("Flesch Kincaid")
    #print(FlecshKincaid)
    
    # Gunning fog index
    GunningFogIndex = readability_formulas.getGunningFogIndex(words_count, sent_num, complex_word_number)
    GunningFogIndexes.append(GunningFogIndex)
    #print("Gunning fog index")
    #print(GunningFogIndex)
    
    # Lasbarhedsindex
    LIX = readability_formulas.getLIX(words_count, sent_num, long_word_number)
    LIXes.append(LIX)
    #print("LIX")
    #print(LIX)
    
    # Miyazaki EFL readability index
    MiyazakiEFL = readability_formulas.getMiyazakiEFL(chars_number, words_count, sent_num)
    MiyazakiEFLs.append(MiyazakiEFL)
    #print("Miyazaki EFL readability index")
    #print(MiyazakiEFL)
    
    # New Dale Chall
    newDaleChall = readability_formulas.getNewDaleChall(words_count, sent_num, number_difficult_word)
    newDaleChalls.append(newDaleChall)
    #print("new Dale chall")
    #print(newDaleChall)
    
    # Smog grading
    smogGrading = readability_formulas.getSmogGrading(sent_num, complex_word_number)
    smogGradings.append(smogGrading)
    print("Smog grading")
    print(smogGrading)

    
    