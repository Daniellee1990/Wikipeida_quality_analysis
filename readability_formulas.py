#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:50:38 2018

@author: lixiaodan
This file describes readability formulas.
"""
import math
from nltk.corpus import stopwords

peacock_word_set = {'acclaimed', 'amazing', 'astonishing', 'authoritative', 'beautiful', 'best', 'brilliant', 
                    'canonical', 'celebrated', 'charismatic', 'classic', 'cutting-edged', 'defining', 'definitive', 
                    'eminent', 'enigma', 'exciting', 'extraordinary', 'fabulous', 'famous', 'infamous', 'fantastic', 
                    'fully', 'genius', 'global', 'great', 'greatest', 'iconic', 'immensely', 'impactful', 'incendiary', 
                    'indisputable', 'influential', 'innovative', 'inspired', 'intriguing', 'leader', 'leading', 
                    'legendary', 'major', 'masterly', 'mature', 'memorable', 'notable', 'outstanding', 'pioneer', 
                    'popular', 'prestigious', 'remarkable', 'renowned', 'respected', 'seminal', 
                    'significant', 'skillful', 'solution', 'single-handedly', 'staunch', 'talented',
                    'top', 'transcendent', 'undoubtedly', 'unique', 'visionary', 'virtually', 'virtuoso', 'well-known', 
                    'well-established', 'world-class', 'worst'}

peacock_phrase = { 'the most',
                   'really good'
                    }

weasel_word_set = { 
                   'about', 'adequate', 'and', 'or', 'appropriate', 'approximately', 'basically', 'clearly', 
                   'completely', 'exceedingly', 'excellent', 'extremely', 'fairly', 'few', 'frequently', 'good', 
                   'huge', 'indicated', 'interestingly', 'largely', 'major', 'many', 'maybe', 'mostly', 'normally', 'often',
                   'perhaps', 'primary', 'quite', 'relatively', 'relevant', 'remarkably', 'roughly', 'significantly', 'several', 
                   'sometimes', 'substantially', 'suitable', 'surprisingly', 'tentatively', 'tiny', 'try', 'typically', 'usually', 
                   'valid', 'various', 'vast', 'very'
                   }
weasel_phrase = {  'are a number', 'as applicable', 'as circumstances dictate', 'as much as possible', 'as needed', 'as required', 
                   'as soon as possible', 'at your earliest convenience', 'critics say', 'depending on', 'experts declare', 
                   'if appropriate', 'if required', 'if warranted', 'is a number', 'in a timely manner', 'in general', 'in most cases', 
                   'in our opinion', 'in some cases', 'in most instances', 'it is believed', 'it is often reported', 'it is our understanding', 'it is widely thought', 
                   'it may', 'it was proven', 'make an effort to', 'many are of the opinion', 'many people think', 'more or less', 
                   'most feel', 'on occasion', 'research has shown', 'science says', 'should be', 'some people say', 'striving for', 'we intend to', 
                   'when necessary', 'when possible'
                   }

lines = open('/Users/lixiaodan/Desktop/wikipedia_project/dataset/dale_chall_word_list.txt', 'r').readlines()
dale_chall_word_list = set()
for line in lines:
    temp = line.split(' ')
    res = temp[0]
    if temp[0][-1] == '\n':
        res = temp[0][0:-1]
    dale_chall_word_list.add(res)

## Automated readability index
def getARI(charCnt, wordCnt, sentCnt):
    ari = 4.71 * charCnt / wordCnt + 0.5 * wordCnt / sentCnt - 21.43
    if ari < 0:
        ari = 0
    return ari

def BormuthIndex(charCnt, wordCnt, sentCnt, diffwordCnt):
    BI = 0.886593 - 0.0364 * charCnt / wordCnt + 0.161911 * (1 - diffwordCnt / wordCnt) - 0.21401 * wordCnt / sentCnt - 0.000577 * wordCnt / sentCnt - 0.000005 * wordCnt / sentCnt
    return BI

def isDifficultWord(word):
    if word in dale_chall_word_list:
        return False
    return True

def isPeacockWord(word):
    if word in peacock_word_set:
        return True
    return False

def hasPeacockPhrase(sentence):
    for phrase in peacock_phrase:
        res = sentence.find(phrase)
        if res != -1:
            #print("peacock")
            #print(phrase)
            return True
    return False

def isWeaselWord(word):
    if word in weasel_word_set:
        return True
    return False

def hasWeaselPhrase(sentence):
    for phrase in weasel_phrase:
        res = sentence.find(phrase)
        if res != -1:
            #print("weasel")
            #print(phrase)
            return True
    return False

def isStopWord(word):
    if word in stopwords.words('english'):
        return True
    return False

## Coleman-Liar index
def getColemanLaurIndex(charCnt, wordCnt, sentCnt):
    CLIndex = 5.89 * charCnt / wordCnt - 30.0 * sentCnt / wordCnt - 15.8
    return CLIndex

## FORCAST readability
def getEduYears(oneSyllableCnt):
    return 20 - oneSyllableCnt / 10.0

## Flesch reading ease
def getFleschReading(wordCnt, sentCnt, oneSyllableCnt):
    return 206.835 - 1.015 * wordCnt / sentCnt - 84.6 * oneSyllableCnt / wordCnt

## Flesch-Kincaid
def getFleschKincaid(wordCnt, sentCnt, oneSyllableCnt):
    return 0.39 * wordCnt / sentCnt + 11.8 * oneSyllableCnt / wordCnt - 15.59

## Gunning Fog Index
def getGunningFogIndex(wordCnt, sentCnt, complexWordCnt):
    return 0.4 * ((wordCnt / sentCnt) + 100.0 * (float(complexWordCnt) / wordCnt))

## Lasbarhedsindex
def getLIX(wordCnt, sentCnt, longWordCnt):
    return float(wordCnt) / sentCnt + 100 * float(longWordCnt) / wordCnt

## Miyazaki EFL readability index
def getMiyazakiEFL(charCnt, wordCnt, sentCnt):
    return 164.935 - 18.792 * charCnt / wordCnt - 1.916 * wordCnt / sentCnt

## new Dale Chall
def getNewDaleChall(wordCnt, sentCnt, diffwordCnt):
    return 0.1579 * diffwordCnt / wordCnt + 0.0496 * wordCnt / sentCnt

## SMOG grading:
def getSmogGrading(sentCnt, complexWordCnt):
    return math.sqrt(30.0 * complexWordCnt / sentCnt) + 3