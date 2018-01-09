#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:02:24 2018

@author: lixiaodan
"""

##### Extraction of structure features ###
path = "/Users/lixiaodan/Desktop/wikipedia_project/dataset/enwiki-20180101-pages-articles1.xml"
file = open(path)
pages = list()
startPage = False
curPage = list()
## split the content by <page> and </page>
for line in file:
    if line.find("<page>") != -1:
        startPage = True
    if line.find("</page>") != -1:
        startpage = False
        curPage.append(line)
        pages.append(curPage)
        curPage = list()
        continue
    if startPage == True:
        curPage.append(line)
        
structure_features = dict()
leads = list()
#number_lead = 0
for page in pages:
    #print(page)
    #print("\n")
    file_cnt = 0
    cate_cnt = 0
    heading_cnt = 0
    image_cnt = 0
    section_cnt = 0
    infobox_cnt = 0
    list_cnt = 0
    structure_feature_per_page = list()
    hasLead = 0 # does not have lead paragraph
    title = ""
    lead_section = list()
    #first_heading = True
    for line in page:
        line = line.lower()
        lead_start = False
        if line.find("<title>") != -1:
           nameEnd = line.find("</title>")
           nameStart = line.find("<title>") + 7
           title = line[nameStart:nameEnd]
           #print(title)
        if line.find("[[file") != -1:
           file_cnt = file_cnt + 1
           #print(line)
        if line.find("[[category") != -1:
           cate_cnt = cate_cnt + 1
           #print(line)
        if line.find("==") != -1:    
           heading_cnt = heading_cnt + 1
           #print(line)
        if line.find("[[image") != -1:
           image_cnt = image_cnt + 1
           #print(line)
        if line.find("infobox") != -1:
           infobox_cnt = infobox_cnt + 1
           #print(line)
        curName = "'''" + title + "'''"
        #print(curName)
        # get lead section
        if line.find(curName) != -1 and heading_cnt == 0:
            lead_start = True
            #print(line)
        if heading_cnt >= 1:
            lead_start = False
            #print(line)
        if lead_start == True:
            lead_section.append(line)
        # get lists
        if line.startswith( '*' ):
            list_cnt = list_cnt + 1
           # print(line)
    #print(title)
    #print(lead_section)
    if len(lead_section) != 0:
        #number_lead = number_lead + 1
        hasLead = 1
    leads.append(lead_section)
    lead_section = list()
    section_cnt = heading_cnt
    if section_cnt != 0:
        images_per_section = float(image_cnt) / section_cnt
    else:
        images_per_section = 0
    structure_feature_per_page.append(file_cnt)
    structure_feature_per_page.append(cate_cnt)
    structure_feature_per_page.append(heading_cnt)
    structure_feature_per_page.append(section_cnt)
    structure_feature_per_page.append(image_cnt)
    structure_feature_per_page.append(infobox_cnt)
    structure_feature_per_page.append(images_per_section)
    structure_feature_per_page.append(hasLead)
    structure_feature_per_page.append(list_cnt)
    structure_features[title] = structure_feature_per_page
    