                                                                    #Copywrite Warning: Owner of the code is Gulcheera Academy(Khosiyat Sabirova)
                                                        #This code can be used by anyone for free, but the name "Gulcheera Academy" must be acknowledged 
#Named Entity Recognition with NLTK

#nltk packages are imported
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

example_4Tagging1 = state_union.raw("2005-GWBush.txt")#create a variable to store a raw data which is in text format provided by the corpus of nltk package
example_4Tagging2 = state_union.raw("2006-GWBush.txt")

def namedChunk(sample_text,train_text):
    tokenized_trained = PunktSentenceTokenizer(train_text)
    tokenized = tokenized_trained.tokenize(sample_text)
    try:
        for lexUnit in tokenized[5:]:
            words = nltk.word_tokenize(lexUnit)
            taggedUnit = nltk.pos_tag(words)
            namedChunk = nltk.ne_chunk(taggedUnit, binary=True)
            #namedChunk.draw()
    except Exception as skip:
        print(str(skip))

#print the result
namedChunk(example_4Tagging1,example_4Tagging2)


#Lemmatizing with NLTK

#import the stem package of nltk
from nltk.stem import WordNetLemmatizer

def lemmatize(lexUnit):
    lemmatizer = WordNetLemmatizer()
    lemmatizer.lemmatize(lexUnit)

#print the result
lemmatize("ball")
