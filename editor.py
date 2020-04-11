from __future__ import print_function
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tree import Tree
from nltk import pos_tag,ne_chunk
import json
import math
import re
from nltk import pos_tag,word_tokenize,ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet,stopwords

import nltk
import networkx as nx
import itertools

from scipy.spatial.distance import cosine as cosine_similarity 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
import string
from nltk.tokenize import sent_tokenize, word_tokenize
import pandas as pd

from sklearn.metrics.pairwise import pairwise_kernels

from nltk.stem.wordnet import WordNetLemmatizer


import re
import string


numbers = "(^a(?=\s)|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand)"
day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
week_day = "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
month = "(january|february|march|april|may|june|july|august|september|october|november|december)"
dmy = "(year|day|week|month)"
rel_day = "(today|yesterday|tomorrow|tonight|tonite)"
exp1 = "(before|after|earlier|later|ago)"
exp2 = "(this|next|last)"
iso = "\d+[/-]\d+[/-]\d+ \d+:\d+:\d+\.\d+"
year = "((?<=\s)\d{4}|^\d{4})"
regxp1 = "((\d+|(" + numbers + "[-\s]?)+) " + dmy + "s? " + exp1 + ")"
regxp2 = "(" + exp2 + " (" + dmy + "|" + week_day + "|" + month + "))"

date = "([012]?[0-9]|3[01])"
regxp3 = "(" + date + " " + month + " " + year + ")"
regxp4 = "(" + month + " " + date + "[th|st|rd]?[,]? " + year + ")"

reg1 = re.compile(regxp1, re.IGNORECASE)
reg2 = re.compile(regxp2, re.IGNORECASE)
reg3 = re.compile(rel_day, re.IGNORECASE)
reg4 = re.compile(iso)
reg5 = re.compile(year)
reg6 = re.compile(regxp3, re.IGNORECASE)
reg7 = re.compile(regxp4, re.IGNORECASE)

def extractDate(text):

    
    timex_found = []

  
    found = reg1.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)


    found = reg2.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

 
    found = reg3.findall(text)
    for timex in found:
        timex_found.append(timex)

 
    found = reg4.findall(text)
    for timex in found:
        timex_found.append(timex)

  
    found = reg6.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)

    found = reg7.findall(text)
    found = [a[0] for a in found if len(a) > 1]
    for timex in found:
        timex_found.append(timex)


    found = reg5.findall(text)
    for timex in found:
        timex_found.append(timex)


    return timex_found


lemtz = WordNetLemmatizer()

def normalize_and_tokenize(text, stemmer = lemtz.lemmatize):

    tokens = word_tokenize(text)
    try: 
        outv = [stemmer(t).translate(None, string.punctuation) for t in tokens] 
        return outv
    except: 
        translator = str.maketrans(dict.fromkeys(string.punctuation))
        return [stemmer(t).translate(translator) for t in tokens]


def vectorize(sentences, tfidf=True, ngram_range=None):
   
    if ngram_range is None:
        ngram_range = (1,1)
    vect = TfidfVectorizer(sublinear_tf=True, max_df=0.5, analyzer='word', 
                   stop_words='english', lowercase=True, ngram_range=ngram_range)
    return vect.fit_transform(sentences)
    

    
def similarity_graph_from_term_document_matrix(sp_mat):
    dx = pairwise_kernels(sp_mat, metric='cosine')
    g = nx.from_numpy_matrix(dx)
    
    return g
    
def summarize(text=None, term_doc_matrix=None,  n=10, tfidf=False, ngram_range=None, verbose=False):
 
    if term_doc_matrix is None:
        if verbose: print("Reading document...")
        sentences = sent_tokenize(text)
        if verbose: print("Fitting vectorizer...")
        term_doc_matrix = vectorize(sentences, tfidf=tfidf, ngram_range=ngram_range)
    if verbose: print("Building similarity graph...")
    g = similarity_graph_from_term_document_matrix(term_doc_matrix)
    if verbose: print("Calculating sentence pagerank (lexrank)...")
    scores = pd.Series(nx.pagerank(g, weight='weight'))
    
    scores.sort_values(ascending=False, inplace=True)
    ix = pd.Series(scores.index[:n])
  
    ix.sort_values(inplace=True)
    summary = [sentences[i] for i in ix]
    return {'summary':summary}



class DocumentRetrievalModel:
    def __init__(self,paragraphs,removeStopWord = False,useStemmer = False):
        self.idf = {}              
        self.paragraphInfo = {}     
        self.paragraphs = paragraphs
        self.totalParas = len(paragraphs)
        self.stopwords = stopwords.words('english')
        self.removeStopWord = removeStopWord
        self.useStemmer = useStemmer
        self.vData = None
        self.stem = lambda k:k.lower()
        if useStemmer:
            ps = PorterStemmer()
            self.stem = ps.stem
            
        
        self.computeTFIDF()
        
    def getTermFrequencyCount(self,paragraph):
        sentences = sent_tokenize(paragraph)
        wordFrequency = {}
        for sent in sentences:
            for word in word_tokenize(sent):
                if self.removeStopWord == True:
                    if word.lower() in self.stopwords:
                       
                        continue
                    if not re.match(r"[a-zA-Z0-9\-\_\\/\.\']+",word):
                        continue
            
                if self.useStemmer:
                    word = self.stem(word)
                    
                if word in wordFrequency.keys():
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1
        return wordFrequency
    
    def computeTFIDF(self):
        
        self.paragraphInfo = {}
        for index in range(0,len(self.paragraphs)):
            wordFrequency = self.getTermFrequencyCount(self.paragraphs[index])
            self.paragraphInfo[index] = {}
            self.paragraphInfo[index]['wF'] = wordFrequency
        
        wordParagraphFrequency = {}
        for index in range(0,len(self.paragraphInfo)):
            for word in self.paragraphInfo[index]['wF'].keys():
                if word in wordParagraphFrequency.keys():
                    wordParagraphFrequency[word] += 1
                else:
                    wordParagraphFrequency[word] = 1
        
        self.idf = {}
        for word in wordParagraphFrequency:
            
            self.idf[word] = math.log((self.totalParas+1)/wordParagraphFrequency[word])
        
       
        for index in range(0,len(self.paragraphInfo)):
            self.paragraphInfo[index]['vector'] = {}
            for word in self.paragraphInfo[index]['wF'].keys():
                self.paragraphInfo[index]['vector'][word] = self.paragraphInfo[index]['wF'][word] * self.idf[word]
    

    def query(self,pQ):
        
        
        relevantParagraph = self.getSimilarParagraph(pQ.qVector)

        
        sentences = []
        for tup in relevantParagraph:
            if tup != None:
                p2 = self.paragraphs[tup[0]]
                sentences.extend(sent_tokenize(p2))
        
       
        if len(sentences) == 0:
            return "Oops! Unable to find answer"

       
        relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)

       
        aType = pQ.aType
        
        
        answer = relevantSentences[0][0]

        ps = PorterStemmer()
        
        if aType == "PERSON":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "PERSON":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "LOCATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "GPE":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "ORGANIZATION":
            ne = self.getNamedEntity([s[0] for s in relevantSentences])
            for entity in ne:
                if entity[0] == "ORGANIZATION":
                    answer = entity[1]
                    answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                   
                    qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                    if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                        continue
                    break
        elif aType == "DATE":
            allDates = []
            for s in relevantSentences:
                allDates.extend(extractDate(s[0]))
            if len(allDates)>0:
                answer = allDates[0]
        elif aType in ["NN","NNP"]:
            candidateAnswers = []
            ne = self.getContinuousChunk([s[0] for s in relevantSentences])
            for entity in ne:
                if aType == "NN":
                    if entity[0] == "NN" or entity[0] == "NNS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                       
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
                elif aType == "NNP":
                    if entity[0] == "NNP" or entity[0] == "NNPS":
                        answer = entity[1]
                        answerTokens = [ps.stem(w) for w in word_tokenize(answer.lower())]
                        qTokens = [ps.stem(w) for w in word_tokenize(pQ.question.lower())]
                        
                        if [(a in qTokens) for a in answerTokens].count(True) >= 1:
                            continue
                        break
        elif aType == "DEFINITION":
            relevantSentences = self.getMostRelevantSentences(sentences,pQ,1)
            answer = relevantSentences[0][0]
        return answer
        
    def getSimilarParagraph(self,queryVector):
        queryVectorDistance = 0
        for word in queryVector.keys():
            if word in self.idf.keys():
                queryVectorDistance += math.pow(queryVector[word]*self.idf[word],2)
        queryVectorDistance = math.pow(queryVectorDistance,0.5)
        if queryVectorDistance == 0:
            return [None]
        pRanking = []
        for index in range(0,len(self.paragraphInfo)):
            sim = self.computeSimilarity(self.paragraphInfo[index], queryVector, queryVectorDistance)
            pRanking.append((index,sim))
        
        return sorted(pRanking,key=lambda tup: (tup[1],tup[0]), reverse=True)[:3]
    
    def computeSimilarity(self, pInfo, queryVector, queryDistance):
        
        pVectorDistance = 0
        for word in pInfo['wF'].keys():
            pVectorDistance += math.pow(pInfo['wF'][word]*self.idf[word],2)
        pVectorDistance = math.pow(pVectorDistance,0.5)
        if(pVectorDistance == 0):
            return 0

        
        dotProduct = 0
        for word in queryVector.keys():
            if word in pInfo['wF']:
                q = queryVector[word]
                w = pInfo['wF'][word]
                idf = self.idf[word]
                dotProduct += q*w*idf*idf
        
        sim = dotProduct / (pVectorDistance * queryDistance)
        return sim
    
    def getMostRelevantSentences(self, sentences, pQ, nGram=3):
        relevantSentences = []
        for sent in sentences:
            sim = 0
            if(len(word_tokenize(pQ.question))>nGram+1):
                sim = self.sim_ngram_sentence(pQ.question,sent,nGram)
            else:
                sim = self.sim_sentence(pQ.qVector, sent)
            relevantSentences.append((sent,sim))
        
        return sorted(relevantSentences,key=lambda tup:(tup[1],tup[0]),reverse=True)
    
    def sim_ngram_sentence(self, question, sentence,nGram):
        
        ps = PorterStemmer()
        getToken = lambda question:[ ps.stem(w.lower()) for w in word_tokenize(question) ]
        getNGram = lambda tokens,n:[ " ".join([tokens[index+i] for i in range(0,n)]) for index in range(0,len(tokens)-n+1)]
        qToken = getToken(question)
        sToken = getToken(sentence)

        if(len(qToken) > nGram):
            q3gram = set(getNGram(qToken,nGram))
            s3gram = set(getNGram(sToken,nGram))
            if(len(s3gram) < nGram):
                return 0
            qLen = len(q3gram)
            sLen = len(s3gram)
            sim = len(q3gram.intersection(s3gram)) / len(q3gram.union(s3gram))
            return sim
        else:
            return 0
    
    def sim_sentence(self, queryVector, sentence):
        sentToken = word_tokenize(sentence)
        ps = PorterStemmer()
        for index in range(0,len(sentToken)):
            sentToken[index] = ps.stem(sentToken[index])
        sim = 0
        for word in queryVector.keys():
            w = ps.stem(word)
            if w in sentToken:
                sim += 1
        return sim/(len(sentToken)*len(queryVector.keys()))
    
    def getNamedEntity(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            nc = ne_chunk(pos_tag(answerToken))
            entity = {"label":None,"chunk":[]}
            for c_node in nc:
                if(type(c_node) == Tree):
                    if(entity["label"] == None):
                        entity["label"] = c_node.label()
                    entity["chunk"].extend([ token for (token,pos) in c_node.leaves()])
                else:
                    (token,pos) = c_node
                    if pos == "NNP":
                        entity["chunk"].append(token)
                    else:
                        if not len(entity["chunk"]) == 0:
                            chunks.append((entity["label"]," ".join(entity["chunk"])))
                            entity = {"label":None,"chunk":[]}
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["label"]," ".join(entity["chunk"])))
        return chunks
    
    def getContinuousChunk(self,answers):
        chunks = []
        for answer in answers:
            answerToken = word_tokenize(answer)
            if(len(answerToken)==0):
                continue
            nc = pos_tag(answerToken)
            
            prevPos = nc[0][1]
            entity = {"pos":prevPos,"chunk":[]}
            for c_node in nc:
                (token,pos) = c_node
                if pos == prevPos:
                    prevPos = pos       
                    entity["chunk"].append(token)
                elif prevPos in ["DT","JJ"]:
                    prevPos = pos
                    entity["pos"] = pos
                    entity["chunk"].append(token)
                else:
                    if not len(entity["chunk"]) == 0:
                        chunks.append((entity["pos"]," ".join(entity["chunk"])))
                        entity = {"pos":pos,"chunk":[token]}
                        prevPos = pos
            if not len(entity["chunk"]) == 0:
                chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    def getqRev(self, pq):
        if self.vData == None:
            
            self.vData = json.loads(open("validatedata.py","r").readline())
        revMatrix = []
        for t in self.vData:
            sent = t["q"]
            revMatrix.append((t["a"],self.sim_sentence(pq.qVector,sent)))
        return sorted(revMatrix,key=lambda tup:(tup[1],tup[0]),reverse=True)[0][0]
        
    def __repr__(self):
        msg = "Total Paras " + str(self.totalParas) + "\n"
        msg += "Total Unique Word " + str(len(self.idf)) + "\n"
        msg += str(self.getMostSignificantWords())
        return msg




class ProcessedQuestion:
    def __init__(self, question, useStemmer = False, useSynonyms = False, removeStopwords = False):
        self.question = question
        self.useStemmer = useStemmer
        self.useSynonyms = useSynonyms
        self.removeStopwords = removeStopwords
        self.stopWords = stopwords.words("english")
        self.stem = lambda k : k.lower()
        if self.useStemmer:
            ps = PorterStemmer()
            self.stem = ps.stem
        self.qType = self.determineQuestionType(question)
        self.searchQuery = self.buildSearchQuery(question)
        self.qVector = self.getQueryVector(self.searchQuery)
        self.aType = self.determineAnswerType(question)
    

    def determineQuestionType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTags = []
        for token in qPOS:
            if token[1] in questionTaggers:
                qTags.append(token[1])
        qType = ''
        if(len(qTags)>1):
            qType = 'complex'
        elif(len(qTags) == 1):
            qType = qTags[0]
        else:
            qType = "None"
        return qType
    
    def determineAnswerType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTag = None

        for token in qPOS:
            if token[1] in questionTaggers:
                qTag = token[0].lower()
                break
        
        if(qTag == None):
            if len(qPOS) > 1:
                if qPOS[1][1].lower() in ['is','are','can','should']:
                    qTag = "YESNO"
        
        if qTag == "who":
            return "PERSON"
        elif qTag == "where":
            return "LOCATION"
        elif qTag == "when":
            return "DATE"
        elif qTag == "what":
            
            qTok = self.getContinuousChunk(question)
           
            if(len(qTok) == 4):
                if qTok[1][1] in ['is','are','was','were'] and qTok[2][0] in ["NN","NNS","NNP","NNPS"]:
                    self.question = " ".join([qTok[0][1],qTok[2][1],qTok[1][1]])
                   
                    return "DEFINITION"

            
            for token in qPOS:
                if token[0].lower() in ["city","place","country"]:
                    return "LOCATION"
                elif token[0].lower() in ["company","industry","organization"]:
                    return "ORGANIZATION"
                elif token[1] in ["NN","NNS"]:
                    return "FULL"
                elif token[1] in ["NNP","NNPS"]:
                    return "FULL"
            return "FULL"
        elif qTag == "how":
            if len(qPOS)>1:
                t2 = qPOS[2]
                if t2[0].lower() in ["few","great","little","many","much"]:
                    return "QUANTITY"
                elif t2[0].lower() in ["tall","wide","big","far"]:
                    return "LINEAR_MEASURE"
            return "FULL"
        else:
            return "FULL"
    
    def buildSearchQuery(self, question):
        qPOS = pos_tag(word_tokenize(question))
        searchQuery = []
        questionTaggers = ['WP','WDT','WP$','WRB']
        for tag in qPOS:
            if tag[1] in questionTaggers:
                continue
            else:
                searchQuery.append(tag[0])
                if(self.useSynonyms):
                    syn = self.getSynonyms(tag[0])
                    if(len(syn) > 0):
                        searchQuery.extend(syn)
        return searchQuery
    
    def getQueryVector(self, searchQuery):
        vector = {}
        for token in searchQuery:
            if self.removeStopwords:
                if token in self.stopWords:
                    continue
            token = self.stem(token)
            if token in vector.keys():
                vector[token] += 1
            else:
                vector[token] = 1
        return vector
    
    def getContinuousChunk(self,question):
        chunks = []
        answerToken = word_tokenize(question)
        nc = pos_tag(answerToken)

        prevPos = nc[0][1]
        entity = {"pos":prevPos,"chunk":[]}
        for c_node in nc:
            (token,pos) = c_node
            if pos == prevPos:
                prevPos = pos       
                entity["chunk"].append(token)
            elif prevPos in ["DT","JJ"]:
                prevPos = pos
                entity["pos"] = pos
                entity["chunk"].append(token)
            else:
                if not len(entity["chunk"]) == 0:
                    chunks.append((entity["pos"]," ".join(entity["chunk"])))
                    entity = {"pos":pos,"chunk":[token]}
                    prevPos = pos
        if not len(entity["chunk"]) == 0:
            chunks.append((entity["pos"]," ".join(entity["chunk"])))
        return chunks
    
    def getSynonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name().lower()
                synonyms.extend(w.split("_"))
        return list(set(synonyms))
    
    def __repr__(self):
        msg = "Q: " + self.question + "\n"
        msg += "QType: " + self.qType + "\n"
        msg += "QVector: " + str(self.qVector) + "\n"
        return msg




print("Answer> Please wait, while I am loading my dependencies")

import warnings
warnings.filterwarnings("ignore")
import re
import sys
import os
fn = sys.argv[1]
print("Answer>Welcome to Advanced lex&python-supported text editor.\nChoose your options here\n")
while True:
    print("Press 1-->find a word\nPress 2-->Display file details\nPress 3-->To make a copy of the current file\nPress 4-->To find and replace any word\nPress 5-->To summarize\nPress 6-->To ask questions")
    print("Press Ctrl-D to terminate a option")
    print("\nChoice>")
    ch=int(input())
    if ch==1:
        os.system("./find"+" "+fn)
    elif ch==2:
        print("The file details is as follows:\n")
        os.system("./count"+" "+fn)
    elif ch==3:
        os.system("./copy"+" "+fn)
        print("File content copied to copy.txt")
    elif ch==4:
        x=input("Enter word to find :\t")
        y=input("Enter word to replace :\t")
        os.system("./repl"+" "+fn+" "+x+" "+y)
        print("File content replaced in copy.txt")
    elif ch==5:
        with open(fn, 'r') as f:
            text = f.read()    
            test = summarize(text)
        print(test)
    elif ch==6:
        break


print("Answer> Ask me questions based on facts!")
print("Answer> You can say me Bye to end!")

if len(sys.argv) == 1:
	print("Answer> Please! Rerun me using following syntax")
	print("\t\t$ python P2.py <datasetName>")
	exit()

datasetName = sys.argv[1]

try:
	datasetFile = open(datasetName,"r")
except FileNotFoundError:
	print("Answer> Oops! I am unable to locate \"" + datasetName + "\"")
	exit()


paragraphs = []
for para in datasetFile.readlines():
	if(len(para.strip()) > 0):
		paragraphs.append(para.strip())

drm = DocumentRetrievalModel(paragraphs,True,True)



greetPattern = re.compile("^\ *((hi+)|((good\ )?morning|evening|afternoon)|(he((llo)|y+)))\ *$",re.IGNORECASE)

isActive = True
while isActive:
    userQuery = input("You>    ")
    if(not len(userQuery)>0):
        print("Answer> You need to ask something")
    elif userQuery.strip().lower() == "bye":
        response = "Bye Bye"
        isActive = False
    elif greetPattern.findall(userQuery):
        response = "Hello!"        
  
    else:
        pq = ProcessedQuestion(userQuery,True,False,True)
        response =drm.query(pq)
    print("Answer> "+response+"\n")
