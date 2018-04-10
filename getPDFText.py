#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 13:53:40 2018

@author: lixiaodan
"""
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfpage import PDFTextExtractionNotAllowed
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfdevice import PDFDevice
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
import unicodedata, codecs
from cStringIO import StringIO

def getPDFText(pdfFilenamePath):
    retstr = StringIO()
    parser = PDFParser(open(pdfFilenamePath,'r'))
    try:
        document = PDFDocument(parser)
    except Exception as e:
        print(pdfFilenamePath,'is not a readable pdf')
        return ''
    if document.is_extractable:
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr,retstr, codec='ascii' , laparams = LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(document):
            interpreter.process_page(page)
        return retstr.getvalue()
    else:
        print(pdfFilenamePath,"Warning: could not extract text from pdf file.")
        return ''

if __name__ == '__main__':
    path = '/Users/lixiaodan/Desktop/wikipedia_project/Dale_Chall_Word_List.pdf'
    words = getPDFText(path)