from __future__ import print_function
import struct
import numpy as np
import time
import os
import re
import pdb
from functools import partial
import random
import io
import codecs


def generateDict(mapping):
  '''
  Creates a dictionary from a mapping file containing
  links between italian syncons and its respective value in
  any other language.

  Parameters
  ----------
  mapping:
    Mapping file.

  Returns
  -------
  dictionary
    Italian syncons: key
    Other syncons: value
  '''
  i = 0
  d = {}
  for line in mapping:
    (val,key,other) = line.split()
    pair = d.setdefault(key, [])
    pair.append(val)
  for v in d.itervalues():
    if len(v) > 1: i += 1
  print(('Valores repetidos:',i))
  return d


def mapSynconsLang(tsv,lang):
  """Creates a dictionary containing the italian syncons - X language syncons
  pairs taking into account the english syncons that appear in the vocabulary.

  Parameters
  ----------
  vocabulary
    File containing the english syncons and words to process.
  lang
    The language of the target syncons.
  Returns
  -------
  dictionary
    Dictionary containing the italian - X language pairs matching the vocabulary.
  """
  nf = 0
  totalsyncons = 0
  it_lang = {}
  result = {}
  if lang == 'en':
    it2entxt = open(os.getcwd()+'/word2vec/it2en.dump.txt', 'r')
    it_lang = generateDict(it2entxt)
  else:
    sp2ittxt = open(os.getcwd()+'/word2vec/sp2it.dump.txt','r')
    it_lang = generateDict(sp2ittxt)
  with open(tsv) as tsvf:
    with open(os.getcwd()+'/Mapping-set2/notfound_'+lang+'.txt','w') as nftxt:
      for line in tsvf:
        token = line.split()
        if lang+'#' in token[0]:
          totalsyncons += 1
          syncon = token[0].split(lang)[1]
          try:
            translation = it_lang[syncon]
            for t in translation:
              result[t] = syncon
          except KeyError:
            nf += 1
            nftxt.write(syncon+' '+ lang +'\n')
  return result



def mapSyncons(it_en,it_es):
  """Creates a file containing the english syncons - spanish syncons pair
  by making a logical AND between the it - en and it - es dictionaries.

  Parameters
  ----------
  it_en
    Dictionary containing the italian - english pairs.
  it_es
    Dictionary containing the italian - spanish pairs.
  """
  enk = set(it_en.keys())
  esk = set(it_es.keys())
  intersection = enk & esk
  symdiff = enk ^ esk
  with open(os.getcwd()+'/Mapping-set2/en_es.txt','w') as m:
    for value in intersection:
      m.write(it_en[value] + ' ' + it_es[value] +'\n')



def createEmbeddings(rowsf, colsf, lang, ttype):
  """Reads the rows tsv and columns tsv cointaining the vector embeddings
  and adds them.
  Checks if the tokens in the vocabulary are in the EN - ES mapping, and then
  creates a tuple containing the syncons and their embeddings.

  Parameters
  ----------
  rowsf
    Row embeddings.
  colsf
    Column embeddings.
  lang:
    Source language of the vocabulary.
  ttype:
    Type of the token.

  Returns
  -------
  tuple
    Contains two aligned vectors, one with tokens and the other one with
    their respective embeddings.
  """
  vec = []
  identifiers = []
  mapping_s = {}
  mapping_l = {}
  with open(os.getcwd()+'/dicts/en_es.txt','r') as m:
    if lang == 'en':
      for line in m:
        (key,val) = line.split() #EN -> SP
        mapping_s[key] = val
    else:
      for line in m:
        (val,key) = line.split() #SP -> EN
        mapping_s[key] = val
  with open(os.getcwd()+'/dicts/en2es-lemma-dict.txt','r') as m:
    if lang == 'en':
      for line in m:
        (key,val) = line.split(':') #EN -> SP
        mapping_l[key] = val.strip('\n')#.decode("utf-8")
    else:
      for line in m:
        (val,key) = line.split(':') #SP -> EN
        mapping_l[key.strip('\n')] = val
  with open(rowsf, 'r') as rows:
    with open(colsf, 'r') as cols:
      for row, col in zip(rows,cols):
        token = row.split('\t')[0]
        if lang+'#' in token: #Si el token cumple el formato de syncon
          token = token.split(lang)[1]
        if ttype == 'syncon':
          if token in mapping_s:
            identifiers.append(token)
            rowe = map(float, row.split('\t')[1:])
            cole = map(float, col.split('\t')[1:])
            vec.append([x+y for x,y in zip(rowe,cole)])
        elif ttype == 'lemma':
          if token in mapping_l:
            identifiers.append(token)
            rowe = map(float, row.split('\t')[1:])
            cole = map(float, col.split('\t')[1:])
            vec.append([x+y for x,y in zip(rowe,cole)])
        else:
          if token in mapping_s or token in mapping_l:
            identifiers.append(token)
            rowe = map(float, row.split('\t')[1:])
            cole = map(float, col.split('\t')[1:])
            vec.append([x+y for x,y in zip(rowe,cole)])
  return np.asarray(identifiers), np.asarray(vec)



def createFullEmbeddings(rowsf, colsf, lang):
  """Reads the rows tsv and columns tsv cointaining the vector embeddings
  and adds them.

  Parameters
  ----------
  rowsf
    Row embeddings.
  colsf
    Column embeddings.
  lang:
    Source language of the vocabulary.

  Returns
  -------
  tuple
    Contains two aligned vectors, one with tokens and the other one with
    their respective embeddings.
  """
  vec = []
  identifiers = []
  with open(rowsf, 'r') as rows:
    with open(colsf, 'r') as cols:
      for row, col in zip(rows,cols):
        token = row.split('\t')[0]
        if lang+'#' in token: #Si el token cumple el formato de syncon
          token = token.split(lang)[1]
        identifiers.append(token)
        rowe = map(float, row.split('\t')[1:])
        cole = map(float, col.split('\t')[1:])
        vec.append([x+y for x,y in zip(rowe,cole)])
  return np.asarray(identifiers), np.asarray(vec)



def generateVectors(binFile, vocabulary, embeddings, lang):
  """Reads the .bin file cointaining the vector embeddings and unpacks them.
  Checks if the tokens in the vocabulary are in the EN - ES mapping, and then
  creates a tuple containing the syncons and their embeddings.

  Parameters
  ----------
  binFile
    Binary file of the vector embeddings of the vocabulary.
  vocabulary
    File containing the tokens and words.
  embeddings
    Size of the embedding vectors.
  lang:
    Source language of the vocabulary.

  Returns
  -------
  tuple
    Contains two aligned vectors, one with syncons and the other one with
    their respective embeddings.
  """
  vec = []
  tokens = []
  syncons = {}
  lemmas = {}
  fmt = struct.Struct('%df' % embeddings)
  with open(binFile,'rb') as binf:
    with codecs.open(vocabulary,'r',encoding='utf-8') as vocab:
      for data, token in zip(iter(partial(binf.read, embeddings*4), ''), vocab):
        token = token.strip('\n')
        tokens.append(token)
        vec.append(fmt.unpack(data))
  return np.asarray(tokens), np.asarray(vec)



def paralellizeVectors(source,target,slang):
  """Lines up the target syncons with the source syncons according to their
  translations.

  Parameters
  ----------
  source
    Tuple containing the source syncons and their embeddings.
  target
    Tuple containing the target syncons and their embeddings.
  slang
    Language of the source syncons.

  Returns
  -------
  tuple
    New tuple with the syncons and embeddings aligned to the source pairs.
  """
  target_values = []
  target_tokens = []
  translation_s, translation_l = loadDictionaries('en', 'es', slang)
  target_embeddings = dict((x,y) for x,y in zip(target[0],target[1]))
  synCnt = 0
  lemCnt = 0
  otherCnt = 0
  for s_token in source[0]:
#    t_syn = translation_s[s_token]
#    t_lem = translation_l[s_token]
    if '#' in s_token and translation_s[s_token] in target_embeddings:
      synCnt = synCnt + 1
      t_token = translation_s[s_token]
      target_values.append(target_embeddings[t_token])
      target_tokens.append(t_token)
    elif translation_l[s_token] in target_embeddings:
      lemCnt = lemCnt + 1
      t_token = translation_l[s_token]
      target_values.append(target_embeddings[t_token])
      target_tokens.append(t_token)
    else:
      otherCnt = otherCnt + 1
  print('aligned %d syncons, %s lemmas and %s others' % (synCnt, lemCnt, otherCnt))
  return np.asarray(target_tokens), np.asarray(target_values)


def loadDictionaries(langA, langB, slang):
  """Loads the syncon and lemma dictionaries between srcLang and tgtLang

  Returns
  ---
  tuple 
    translation_s, translation_l
  """
  translation_s = {}
  translation_l = {}
  with open(os.getcwd()+'/dicts/'+langA+'_'+langB+'.txt','r') as syncons:
    if slang == langA:
      for line in syncons:
        (key,val) = line.split() #langA -> langB
        translation_s[key] = val
    else:
      for line in syncons:
        (val,key) = line.split() #langB -> langA
        translation_s[key] = val
  with codecs.open(os.getcwd()+'/dicts/en2es-lemma-dict.txt','r',encoding='utf-8') as lemmas:
    if slang == langA:
      for line in lemmas:
        (key,val) = line.split(':') #langA -> langB
        translation_l[key] = val.strip('\n')
    else:
      for line in lemmas:
        (val,key) = line.split(':') #langB -> langA
        translation_l[key.strip('\n')] = val
  return translation_s, translation_l

def translationMatrix(source,target):
  """Solves the matrix equation A*X=B, where X is the translation matrix between
  two matrices of embeddings.

  Parameters
  ----------
  source
    Matrix containing the embeddings of the source language. (A in the equation)
  target
    Matrix containing the embeddings of the target language. (B in the equation)

  Returns
  -------
  matrix
    Translation matrix between A and B. (X in the equation)
  """
  tm = np.linalg.pinv(source).dot(target)
  return tm



def getSimilars(embedding, size, targetSyncons):
  """Obtains the most similar translations for a token computing
  the cosine distance.

  Parameters
  ----------
  tm
    Translation matrix.
  vector
    Embedding of the syncon we want to analyze.
  size
    Number of similar translations to return.
  targetSyncons
    Tuple that contains the target syncons and their embeddings.

  Returns
  -------
  list
    Contains the most similar syncons.
  """
  distance = []
  target_dict = dict((x,y) for x,y in zip(targetSyncons[0],targetSyncons[1]))
  for syncon in target_dict:
    escalar = np.dot(embedding, target_dict[syncon])
    norma = np.linalg.norm(embedding)*np.linalg.norm(target_dict[syncon])
    if norma == 0.0:
      distance.append(0.0)
    else:
      distance.append(escalar/norma)
  distance = np.asarray(distance)
  targetSyncons = np.asarray(target_dict.keys())
  sortedDistanceIndexes = np.argsort(distance, axis=0)
  sortedDistanceIndexes = sortedDistanceIndexes[::-1]
  sortedDistanceValues = np.sort(distance, axis=0)
  sortedDistanceValues = sortedDistanceValues[::-1]
  t = targetSyncons[sortedDistanceIndexes]
  return t[0:size]




def getTop(translation, vector, size):
  """Generates a value from 1 to len(vector) depending on the position
  of the translation in the translation vector.


  Parameters
  ----------
  translation
    The true translation syncon.
  vector
    Values containing translations for a syncon.

  Returns
  -------
  int
    0 if the true translation is not in the vector, otherwise it returns
    a value depending on the position of the translation. len(vector) if
    the translation is the first value in the vector, len(vector)-1 if the
    translation is in the second position of the value...
  """
  i = 0
  found = False
  for i in range(len(vector)):
    if translation == vector[i]:
      found = True
      break
    i += 1
  if not found: return 0
  else:
    l = range(1,len(vector)+1)[::-1]
    return l[i]



def getFrequentSL(en, es, size, tokenType):
  syncons = 0
  lemmas = 0
  new_en_tokens = []
  new_en_embeddings = []
  new_es_tokens = []
  new_es_embeddings = []
  for token_en,embedding_en,token_es,embedding_es in zip(en[0],en[1],es[0],es[1]):
    if tokenType == 'syncon':
      if syncons == size: break
      if '#' in token_en:
        new_en_tokens.append(token_en)
        new_es_tokens.append(token_es)
        new_en_embeddings.append(embedding_en)
        new_es_embeddings.append(embedding_es)
        syncons += 1
    elif tokenType == 'lemma':
      if lemmas == size: break
      if '#' not in token_en:
        new_en_tokens.append(token_en)
        new_es_tokens.append(token_es)
        new_en_embeddings.append(embedding_en)
        new_es_embeddings.append(embedding_es)
        lemmas += 1
    else:
      if syncons == size/2 and lemmas == size/2: break
      if '#' in token_en and syncons < size/2:
        new_en_tokens.append(token_en)
        new_es_tokens.append(token_es)
        new_en_embeddings.append(embedding_en)
        new_es_embeddings.append(embedding_es)
        syncons += 1
      if '#' not in token_en and lemmas < size/2:
        new_en_tokens.append(token_en)
        new_es_tokens.append(token_es)
        new_en_embeddings.append(embedding_en)
        new_es_embeddings.append(embedding_es)
        lemmas += 1
  return (np.asarray(new_en_tokens),np.asarray(new_en_embeddings)),(np.asarray(new_es_tokens),np.asarray(new_es_embeddings))



def getSLSamples(en, size, maxsize):
  syncons = 0
  lemmas = 0
  sample_tokens = []
  sample_embeddings = []
  indexes = random.sample(range(maxsize),maxsize)
  for token,value in zip(en[0][indexes],en[1][indexes]):
    if syncons == size/2 and lemmas == size/2: break
    if '#' in token and syncons < size/2:
      sample_tokens.append(token)
      sample_embeddings.append(value)
      syncons += 1
    if '#' not in token and lemmas < size/2:
      sample_tokens.append(token)
      sample_embeddings.append(value)
      lemmas += 1
  return np.asarray(sample_tokens),np.asarray(sample_embeddings)



def generateBin(embeddings,path):
  print(len(embeddings[1][0]))
  fmt = struct.Struct('%df' % len(embeddings[1][0]))
  print(fmt)
  with open(path+'/vecs.bin','wb') as binf:
    with open(path+'/vocab.txt','w') as vocabf:
       for token,value in zip(embeddings[0],embeddings[1]):
         print(token, file=vocabf)
         binf.write(fmt.pack(*value))



def generateDicts():
  d_en_es = {}
  d_es_en = {}
  with open(os.getcwd()+'/dicts/en2es-lemma-dict.txt','r') as lemmas:
    for line in lemmas:
      (key,val) = line.split(':') #EN -> SP
      d_en_es[key] = val.strip('\n')
      (val,key) = line.split(':') #SP -> EN
      d_es_en[key.strip('\n')] = val
  with open(os.getcwd()+'/dicts/en_es.txt','r') as syncons:
    for line in syncons:
      (key,val) = line.split()
      d_en_es[key] = val
      (val,key) = line.split()
      d_es_en[key] = val
  return d_en_es,d_es_en



def main():
  ### Mapping de traduccion EN -> ES ###
  '''
  print '--- Generando mapping con vocabulario ingles ---'
  start = time.time()
  it_en = mapSynconsLang(en_rows, 'en')
  end = time.time() - start
  print 'Fin: ' + str(end)

  print '--- Generando mapping con vocabulario espanol ---'
  start = time.time()
  it_es = mapSynconsLang(es_rows,'es')
  end = time.time() - start
  print 'Fin: ' + str(end)

  mapSyncons(it_en,it_es)
  '''
  
  ### Obtencion de embeddings para EN y ES ###
  
  print('Generando full embeddings en ingles')
  start = time.time()
  en_rows = os.getcwd()+'/regul_vec/en/original/row_embedding.tsv'
  en_cols = os.getcwd()+'/regul_vec/en/original/col_embedding.tsv'
  en_f = createFullEmbeddings(en_rows, en_cols, 'en')#,'sl') # Tamano maximo 147456
  print('Fin:', time.time() - start)
  print('Generando full embeddings en espanol')
  start = time.time()
  es_rows = os.getcwd()+'/regul_vec/es/original/row_embedding.tsv'
  es_cols = os.getcwd()+'/regul_vec/es/original/col_embedding.tsv'
  es_f = createFullEmbeddings(es_rows, es_cols, 'es')#,'sl') # Tamano maximo 143360
  print('Fin:', time.time() - start)
  print('Generando embeddings en ingles y espanol')
  en = createEmbeddings(en_rows, en_cols, 'en', 'sl')
  es = createEmbeddings(en_rows, en_cols, 'es', 'sl')
  
  ######################## MATRIZ DE TRADUCCION ############################
  #es = paralellizeVectors(en,es,'en')
  #with open(os.getcwd()+'/word2vec/new_en-es_en/vecs.txt','w') as txt:
  #  for x in en[1]:
  #    txt.write(str(x)+'\n')
  #with open(os.getcwd()+'/word2vec/new_en-es_es/vecs.txt','w') as txt:
  #  for x in es[1]:
  #    txt.write(str(x)+'\n')
  
  generateBin(en_f,os.getcwd()+'/regul_vec/en/bin_full')
  generateBin(es_f,os.getcwd()+'/regul_vec/es/bin_full')
  generateBin(en,os.getcwd()+'/regul_vec/en/bin')
  generateBin(es,os.getcwd()+'/regul_vec/es/bin')
  
  #samples = getSLSamples(en,500, len(es[0]))
  #d_en = dict((x,y) for x,y in zip(samples_en[0],samples_en[1])) #embeddings[0][0][0:500],embeddings[0][1][0:500]))
  #with open(os.getcwd()+'/samples.txt','w') as s:
  #  for sample in samples[0]:
  #    s.write(str(sample)+'\n')
  '''
  sbin = os.getcwd()+'/experiments/vecs_en.bin'
  svocab = os.getcwd()+'/experiments/vocab_en.txt'
  tbin = os.getcwd()+'/experiments/vecs_es.bin'
  tvocab = os.getcwd()+'/experiments/vocab_es.txt'

  en = generateVectors(sbin, svocab, 300, 'en')
  es = generateVectors(tbin, tvocab, 300, 'es')

  #embeddings = getFrequentSL(en,es,20000,'SL')
  tmen_es = translationMatrix(en[1][0:5000],es[1][0:5000])
  '''
  '''
  d_en_es = {}
  d_es_en = {}

  with open(os.getcwd()+'/word2vec/en2es-lemma-dict.txt','r') as lemmas:
    for line in lemmas:
      (key,val) = line.split(':') #EN -> SP
      d_en_es[key] = val.strip('\n')
      (val,key) = line.split(':') #SP -> EN
      d_es_en[key.strip('\n')] = val
  with open(os.getcwd()+'/Mapping-set2/en_es.txt','r') as syncons:
    for line in syncons:
      (key,val) = line.split()
      d_en_es[key] = val
      (val,key) = line.split()
      d_es_en[key] = val

  #samples_en = getSLSamples(en,500)
  d_en = {}
  #indexes = random.sample(range(15689),1500) #Para obtener syncons aleatorios
  #d_en = dict((x,y) for x,y in zip(samples_en[0],samples_en[1])) #embeddings[0][0][0:500],embeddings[0][1][0:500]))
  with open(os.getcwd()+'/samples.txt','r') as s:
    aux = dict((x,y) for x,y in zip(en[0], en[1])) #Creo un diccionario con TODOS los embeddings
    for line in s:
      d_en[line.strip('\n')] = aux[line.strip('\n')]
  #d_es = dict((x,y) for x,y in zip(es[0][0:20000],es[1][0:20000]))

  with open(os.getcwd()+'/samples.txt','w') as s:
    for k in d_en:
      s.write(str(k)+'\n')
  start = time.time()
  
  exists = 0
  i = 0
  with open(os.getcwd()+'/500-20000x20000.tsv','w') as tsv:
    with open(os.getcwd()+'/500-media.tsv','a') as m:
      for lemma in d_en:
        n = getSimilars(tmen_es, d_en[lemma], 5, es)
        i += getTop(d_en_es[lemma],n,5)
        if d_en_es[lemma] in n:
          exists += 1
        tsv.write(lemma+'\t'+d_en_es[lemma]+'\t'+str(getTop(d_en_es[lemma],n,5))+'\n')
        print i/float(500)
      m.write('20000x20000\t'+str(i/float(500))+'\t'+str(exists/float(500))+'\n')
  print 'Fin:', time.time() - start
  
  md = 0
  with open(os.getcwd()+'/distances.tsv','a') as d:
    for syncon in d_en:
      matrixTranslation = np.dot(d_en[syncon],tmen_es)
      realTranslation = d_es[d_en_es[syncon]]
      escalar = np.dot(matrixTranslation,realTranslation)
      norma = np.linalg.norm(matrixTranslation)*np.linalg.norm(realTranslation)
      md += escalar/norma
      print md
    d.write('20000x20000\t'+str(md/float(20000))+'\n')
'''
if __name__ == "__main__":
  main()
