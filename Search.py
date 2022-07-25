import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import numpy as np
import string

string.punctuation += "«"
string.punctuation += "»"
string.punctuation += "—"


# TF function definition
def TF(x):
   #return np.log(x + 1)
   return x

# A function that creates from a sentence an array of words processed by a morphological analyzer.
def senttoterm(sent):
    terms = []
    temp = sent.split()
    for j in temp:
        fl = True
        while fl:
            fl = False
            for p in string.punctuation:
                if p in j:
                    j = j.replace(p, '')
                    fl = True
        if j != "":
            j = j.lower()
            q = morph.parse(j)[0].normal_form
            terms.append(q)
    return terms

# A function that creates a dictionary of words from a common array of words processed by the morphological analyzer. The key is the word, the value is the frequency of that word
def arrtodict(terms):
    word_list = {}
    for word in terms:
        if word in word_list:
            word_list[word] += 1
        else:
            word_list[word] = 1
    return word_list

# Paths of original Wikipedia articles
file1 = "C:\\Users\\Frederik\\Desktop\\1.txt"
file2 = "C:\\Users\\Frederik\\Desktop\\2.txt"
file3 = "C:\\Users\\Frederik\\Desktop\\3.txt"

# Read all files and collect them into one array
a = open(file1, encoding="utf-8")
a = a.readlines()
b = open(file2, encoding="utf-8")
b = b.readlines()
c = open(file3, encoding="utf-8")
c = c.readlines()
a = a + b + c
# Remove empty strings from sentence array
a = [s for s in a if s != "\n"]
# Total number of sentences
N = len(a)
s = []
col = []
# Loop creating a common array of words and dictionaries for each sentence (Doc1, Doc2 ...)
for i in a:
    t = senttoterm(i)
    col.append(arrtodict(t))
    s += senttoterm(i)
df = arrtodict(s)
# Create a set of all words
dict = set(s)
# Creating a dictionary of words idf, in which the key is the word, and the value of the idf of this word, calculated by the formula N / df[i]
idf = {}
for i in df:
    idf[i] = N / df[i]

# Create a tf.idf matrix where each row is a sentence and each column is a word
Docs = np.zeros((N, len(dict)))

for i in range(N):
    k = 0
    for j in dict:
        if j in col[i]:
            Docs[i][k] = idf[j] * TF(col[i][j])
        else:
            Docs[i][k] = 0
        k += 1

# Normalization of the resulting vectors
for i in range(N):
    Docs[i] = Docs[i] / np.linalg.norm(Docs[i])

# Line with request

#Ask = "Академический словарь литовского языка помогали составлять два президента, архиепископ, офтальмолог и ещё несколько сотен человек."
#Ask = "Один из главных злодеев трилогии приквелов «Звёздных войн» говорит с русским акцентом."
Ask = "Воспоминания Агаты Кристи об участии в археологических экспедициях в Ираке и Сирии долго не хотели печатать."

# Splitting the query into terms and creating a dictionary from them
Ask = arrtodict(senttoterm(Ask))
# Create a request vector
A = []
for i in dict:
    if i in Ask:
        A.append(idf[i] * TF(Ask[i]))
    else:
        A.append(0)

# Normalization of the query vector
A = np.array(A)
A = A / np.linalg.norm(A)

# Array containing all weights
cos = []

# Calculate weights for each sentence from articles
for i in range(N):
    cos.append(np.dot(A, Docs[i])/(np.linalg.norm(A) * np.linalg.norm(Docs[i])))

# Output 1/5 of all sentences in order of decreasing weight
for i in range(10):
    print(a[cos.index(max(cos))], max(cos))
    a.pop(cos.index(max(cos)))
    cos.pop(cos.index(max(cos)))
