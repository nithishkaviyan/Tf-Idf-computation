##Tf-Idf computation
import nltk
nltk.download("punkt")
import numpy as np


class tfidf:
  def __init__(self):
    self.tf = {}
    self.idf = {}
    self.tfidf ={}

  ##Term frequency  
  def term_freq(self,input):
    '''
    input : list where each element is a string containing a review
    '''
       
    for n,i in enumerate(input):
      for j in nltk.word_tokenize(i.lower()):
        if j not in self.tf.keys():
          self.tf[j] = {n:1}
        else:
          if n not in self.tf[j].keys():
            self.tf[j][n] = 1
          else:
            self.tf[j][n] += 1        
    
    
  ##Inverse Document Frequency
  def inv_df(self,input):
    '''
    input : list where each element is a string containing a review
    '''
    
    N = len(input)
    for k,v in self.tf.items():
      self.idf[k] = np.log(N / len(v))
     
      
  ##TF-IDF
  def tf_idf(self,input):
    '''
    input : list where each element is a string containing a review
    '''
    
    self.term_freq(input)
    self.inv_df(input)
    
    for i,j in self.tf.items():
      if i not in self.tfidf.keys():
        self.tfidf[i] = {}
      for k in j:
        self.tfidf[i][k] = self.tf[i][k] * self.idf[i]    
    
    return self.tfidf
  
  
  if __name__ =="main":
    input = ['This is a sample sentence',/
             'This code computes term frequency inverse document frequency', /
             'The features generated from the given text can be fed as input to a machine learning algorithm'] 
    tf_idf_comput = tfidf()
    print('Sample tf-idf computation')
    print(tf_idf_comput.tf_idf(input))
