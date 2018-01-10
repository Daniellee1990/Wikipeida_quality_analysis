#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 09:02:24 2018

@author: lixiaodan
"""
ref_sec_names = {
            "references", "notes", "footnotes", "sources", "citations", 
            "bibliography", "works cited", "external references", "reference notes", 
            "references cited", "bibliographical references", "cited references", 
            "notes, references", "sources, references, external links", 
            "sources, references, external links, quotations", "notes & references", 
            "references & notes", "external links & references", "references & external links", 
            "references & footnotes", "footnotes & references", "citations & notes", "notes & sources", 
            "sources & notes", "notes & citations", "footnotes & citations", "citations & footnotes", 
            "reference & notes", "footnotes & sources", "note & references", "notes & reference", 
            "sources & footnotes", "notes & external links", "references & further reading", 
            "sources & references", "references & sources", "references & links", "links & references", 
            "references & bibliography", "references & resources", "bibliography & references", 
            "external articles & references", "references & citations", "citations & references", 
            "references & external link", "external link & references", "further reading & references", 
            "notes, sources & references", "sources, references & external links", "references/notes", 
            "notes/references", "notes/further reading", "references/links", "external links/references", 
            "references/external links", "references/sources", "external links / references", 
            "references / sources", "references / external links"
                }

citations = {
            "web", "book", "news", "journal"
            }

trivia_sections = {
           "facts", "miscellanea", "other facts", "other information", "trivia"
                    }

def isRefSec(headline):
    for ref_sec_name in ref_sec_names:
        if headline.find(ref_sec_name) != -1 and headline.find("resources") == -1:
            return True
    return False

def isCitation(line):
    target = line.lower()
    tp = "{{" + "cite "
    for cite in citations:
        tp = tp + cite
        if target.find(tp) != -1:
            return True
    return False

def strCompare(str1, str2):
    str1 = str1.strip( ' ' )
    str2 = str2.strip( ' ' )
    if len(str1) != len(str2):
        return False
    for id in range(len(str1)):
        if str1[id] != str2[id]:
            return False
    return True
##################################
""""
def isTriviaSec(headline):
    hdln = headline.strip('=')
    hdln = hdln.strip(' ')
    hdln = hdln[0:-4]
    for ts in trivia_sections:
        if ts.find(hdln) != -1 :
            return True
    return False
"""

def getCitationsCnt(line):
    citation_cnt = 0
    target = line.lower()
    for cite in citations:
        tp = "{{" + "cite " + cite
        if target.find(tp) != -1:
            citation_cnt = citation_cnt + target.count(tp)
    return citation_cnt

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
plain_texts = list()
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
    ref_cnt = 0
    ref_sec_cnt = 0
    ref_per_sect = 0
    table_cnt = 0
    trivia_sec_cnt = 0
    title = ""
    lead_section = list()
    toDelete = False
    plain_text = list()
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
           toDelete = True
           #print(line)
        if line.find("[[category") != -1:
           cate_cnt = cate_cnt + 1
           toDelete = True
           #print(line)
        if line.find("==") != -1:    
           heading_cnt = heading_cnt + 1
           if isRefSec(line) == True:
               ref_sec_cnt = ref_sec_cnt + 1
               #print("Reference sections")
               #print(line)
           if isTriviaSec(line) == True:
               trivia_sec_cnt = trivia_sec_cnt + 1
               print(line)
           #toDelete = True
           #print(line)
        if line.find("[[image") != -1:
           image_cnt = image_cnt + 1
           toDelete = True
           #print(line)
        if line.find("infobox") != -1:
           infobox_cnt = infobox_cnt + 1
           #toDelete = True
        lt_ref_start = False
        if line.find("&lt;ref") != -1:
            ref_cnt = ref_cnt + line.count("&lt;ref")
            lt_ref_start = True
            #toDelete = True
            #print("&lt;ref")
            #print(ref_cnt)
            #print(line)
        if isCitation(line) == True and lt_ref_start == False:
            ref_cnt = ref_cnt + getCitationsCnt(line)
            #print(getCitationsCnt(line))
            #print("citations")
            #print(ref_cnt)
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
            #toDelete = True
           # print(line)
        # get the table numbers
        if line.startswith("{|"):
            table_cnt = table_cnt + 1
            #print(line)
        """
        if toDelete == False:
            #print("Lines to be deleted")
            #print(line)
            plain_text.append(line)
        """
    if len(lead_section) != 0:
        #number_lead = number_lead + 1
        hasLead = 1
    leads.append(lead_section)
    lead_section = list()
    plain_texts.append(plain_text)
    plain_text = list()
    section_cnt = heading_cnt
    if section_cnt != 0:
        images_per_section = float(image_cnt) / section_cnt
    else:
        images_per_section = 0

    if section_cnt != 0:
        ref_per_sect = float(ref_cnt) / section_cnt
    structure_feature_per_page.append(file_cnt)
    structure_feature_per_page.append(cate_cnt)
    structure_feature_per_page.append(heading_cnt)
    structure_feature_per_page.append(section_cnt)
    structure_feature_per_page.append(image_cnt)
    structure_feature_per_page.append(infobox_cnt)
    structure_feature_per_page.append(images_per_section)
    structure_feature_per_page.append(hasLead)
    structure_feature_per_page.append(list_cnt)
    structure_feature_per_page.append(ref_cnt)
    structure_feature_per_page.append(ref_sec_cnt)
    structure_feature_per_page.append(ref_per_sect)
    structure_feature_per_page.append(table_cnt)
    structure_feature_per_page.append(trivia_sec_cnt)
    structure_features[title] = structure_feature_per_page
    