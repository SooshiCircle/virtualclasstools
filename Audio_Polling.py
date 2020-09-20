import numpy as np
import speech_recognition as sr
from sklearn.metrics.pairwise import cosine_similarity as sim
from gensim.models.keyedvectors import KeyedVectors
from collections import Counter

# Speech to text using speech recognition API

# +
def loadGlove(path):
    """
    Load pretrained GloVe embeddings that map word -> embedding
    """
    glove = KeyedVectors.load_word2vec_format(path, binary=False)
    return glove

glove = loadGlove("glove.6B.300d.txt.w2v")


# -

def speechToText():
    """
    Perform speech to text transcription using SpeechRecognition API
    """
    with sr.Microphone() as source:
        print("Listening...")
        audio = sr.Recognizer().listen(source)
    try:
        speech = sr.Recognizer().recognize_google(audio).lower()
    except sr.UnknownValueError:
        print("Couldn't understand audio")
    return speech

# Clean responses based on general English stop words

def getStopWords(path):
    """
    Load stop words (common filler words that don't add meaning to sentences)
    """
    with open(path) as dat:
        stops = []
        for line in dat:
            stops += [word.strip() for word in line.split('\t')]
    return stops

def cleanDocs(docs, stops):
    """
    Clean documents (student responses) by removing stop words
    """
    cleaned_docs = []
    for doc in docs:
        doc = doc.split()
        cleaned_docs.append([word for word in doc if not(word in stops)])
    return cleaned_docs

# Assesing response content using TF-IDF values

def getVocab(cleaned_docs):
    """
    Get the vocab of all words across all docs (responses)
    """
    vocab = set()
    for doc in cleaned_docs:
        vocab = vocab.union(set(word for word in doc))
    vocab = sorted(list(vocab))
    return vocab

def tf(doc, vocab):
    """
    Compute the term frequency for the document given the vocab
    """
    count = Counter(doc)
    return np.array([count[word] for word in vocab]) # count[word] gives count of word in doc or 0 if word not in doc

def idf(docs, vocab):
    """
    Compute the idf for each word in the vocab
    """
    counts = []
    for word in vocab:
        t = 0
        for doc in docs:
            if word in doc: t += 1
        counts.append(t)
    counts = np.array(counts)
    return np.log(len(docs)/counts)


def tf_idf(vocab, cleaned_docs):
    """
    Compute the tf_idf values for each word in each document
    """
    tfs = np.array([tf(doc, vocab) for doc in cleaned_docs])
    idfs = idf(cleaned_docs, vocab)
    tf_idf = tfs * idfs
    return tf_idf

def printBestWord(tf_idf, vocab):
    """
    Prints the word with the largest tf_idf value for each document
    """
    indices = np.argmax(tf_idf, axis=1)
    indices
    print("most descriptive word")
    for i, idx in enumerate(indices):
        print(f"answer {i+1}: {vocab[idx]}")

# Assesing response similarity to other students using GloVe embeddings

def embedDocs(cleaned_docs, glove, glove_dim):
    """
    Create an embedding for each doc by summing the embeddings of it's words (excluding stop words)
    """
    embeddings = np.zeros((len(cleaned_docs), glove_dim))
    for i, doc in enumerate(cleaned_docs):
        emb = np.sum(np.array([glove[word] for word in doc]), axis=0)
        norm = np.linalg.norm(emb)
        embeddings[i] = emb/norm
    return embeddings

def computeSim(embeddings):
    """
    Compute the pairwise similarities for each response
    """
    pairwise_sims = sim(embeddings, embeddings)
    return pairwise_sims

def main(docs):
    """
    Display all statistics/evaluations on a set of responses
    """
    stops = getStopWords("stopwords.txt")
    cleaned_docs = cleanDocs(docs, stops)

    # TF-IDF assessment
    vocab = getVocab(cleaned_docs)
    tf_idfs = tf_idf(vocab, cleaned_docs)
    printBestWord(tf_idfs, vocab)

    # GloVe assessment
    embeddings = embedDocs(cleaned_docs, glove, 300)

    print()
    print("pairwise similarities")
    print(computeSim(embeddings))
