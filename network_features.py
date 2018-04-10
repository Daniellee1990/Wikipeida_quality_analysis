#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 21:45:04 2018

@author: lixiaodan
get data set : https://dumps.wikimedia.org/enwiki/latest/
reference : https://github.com/sharnett/wiki_pagerank
http://snap.stanford.edu/data/wiki-meta.html
"""


"THis file is to process network features"
filename = "/Users/lixiaodan/Desktop/wikipedia_project/dataset/wiki_pageLinks/tlwiki-latest-pagelinks.sql"
fd = open(filename, 'r', encoding="ISO-8859-1")
sqlFile = fd.read()
sqlCommands = sqlFile.split(';')
for pageLink in fd:
    print(pageLink)
fd.close()