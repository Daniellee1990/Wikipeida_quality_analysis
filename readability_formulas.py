#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 10:50:38 2018

@author: lixiaodan
This file describes readability formulas.
"""
import math

## Automated readability index
def getARI(charCnt, wordCnt, sentCnt):
    ari = 4.71 * charCnt / wordCnt + 0.5 * wordCnt / sentCnt - 21.43
    if ari < 0:
        ari = 0
    return ari

def BormuthIndex(charCnt, wordCnt, sentCnt, diffwordCnt):
    BI = 0.886593 - 0.0364 * charCnt / wordCnt + 0.161911 * (1 - diffwordCnt / wordCnt) - 0.21401 * wordCnt / sentCnt - 0.000577 * wordCnt / sentCnt - 0.000005 * wordCnt / sentCnt
    return BI

lines = open('/Users/lixiaodan/Desktop/wikipedia_project/dataset/dale_chall_word_list.txt', 'r').readlines()
dale_chall_word_list = set()
for line in lines:
    temp = line.split(' ')
    res = temp[0]
    if temp[0][-1] == '\n':
        res = temp[0][0:-1]
    dale_chall_word_list.add(res)

def isDifficultWord(word):
    if word in dale_chall_word_list:
        return False
    return True

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
    
    
    
    

    



    
