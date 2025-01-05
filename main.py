#importing necessary libraries
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#connecting to g drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive')


# function to extract the text of testcases from the given folder
def extract_text(directory):
    text = []
    files = []
    for filename in os.listdir(directory):
        files.append(filename)
        files.sort()
    for filename in files:
        with open(os.path.join(directory, filename), 'r') as f:
            data = f.readlines()
            f.close()
            test_case = ''
            for d in data:
                test_case += d
            text.append(test_case)

    return text

#extracting the text
text = extract_text(r'MFDS_PROJECT/TC')

#converting text to data frame
text_df=pd.DataFrame(text,columns=['text'])

# removing special characters and stop words from the text
stopwords_list=stopwords.words('english')
text_df['text_cleaned']=text_df.text.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stopwords_list) )

#forming the vectors
tfidfvectoriser=TfidfVectorizer()
tfidfvectoriser.fit(text_df.text_cleaned)
tfidf_vectors=tfidfvectoriser.transform(text_df.text_cleaned)

#finding the similarity using cosine angle
cosine_similarities=np.dot(tfidf_vectors[0],tfidf_vectors.T).toarray()

#printing the similarities of given test cases with the original template
print(cosine_similarities[0][1:])

#printing the similarities of given test cases with the original template
for i in range(1,11):
  print('The answer {} is {:.4f}% matching'.format(i,cosine_similarities[0][i]*100))

