#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:25:00 2018

@author: lixiaodan
This file includes the function for text statistics and part of speech features.
"""
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