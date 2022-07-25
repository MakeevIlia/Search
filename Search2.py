import pymorphy2
import pandas as pd

morph = pymorphy2.MorphAnalyzer()
import numpy as np
import string

string.punctuation += "«"
string.punctuation += "»"
string.punctuation += "—"


def TF(x):
    #return np.log(x + 1)
    return x


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


def arrtodict(terms):
    word_list = {}
    for word in terms:
        if word in word_list:
            word_list[word] += 1
        else:
            word_list[word] = 1
    return word_list


def prob(a, Ask, Lambda):
    N = len(a)
    s = []
    col = []
    for i in a:
        t = senttoterm(i)
        col.append(arrtodict(t))
        s += senttoterm(i)
    df = arrtodict(s)
    W = []
    for i in range(N):
        w = 1
        for j in Ask:
            if j in col[i]:
                w *= ((1 - Lambda) * df[j] / len(s) + Lambda * col[i][j] / sum(col[i].values()))
            elif j in df:
                w *= ((1 - Lambda) * df[j] / len(s))
            else:
                w *= 0.000001
        W.append(w)
    return W


def vec(a, Ask):
    N = len(a)
    s = []
    col = []
    for i in a:
        t = senttoterm(i)
        col.append(arrtodict(t))
        s += senttoterm(i)
    df = arrtodict(s)
    dict = set(s)
    idf = {}
    for i in df:
        idf[i] = N / df[i]
    Docs = np.zeros((N, len(dict)))
    for i in range(N):
        k = 0
        for j in dict:
            if j in col[i]:
                Docs[i][k] = idf[j] * TF(col[i][j])
            else:
                Docs[i][k] = 0
            k += 1
    for i in range(N):
        Docs[i] = Docs[i] / np.linalg.norm(Docs[i])
    A = []
    for i in dict:
        if i in Ask:
            A.append(idf[i] * TF(Ask[i]))
        else:
            A.append(0)
    A = np.array(A)
    A = A / np.linalg.norm(A)
    cos = []
    for i in range(N):
        cos.append(np.dot(A, Docs[i]) / (np.linalg.norm(A) * np.linalg.norm(Docs[i])))
    return cos


file1 = "C:\\Users\\Frederik\\Desktop\\1.txt"
file2 = "C:\\Users\\Frederik\\Desktop\\2.txt"
file3 = "C:\\Users\\Frederik\\Desktop\\3.txt"

Lambda = 0.01
a = open(file1, encoding="utf-8")
a = a.readlines()
b = open(file2, encoding="utf-8")
b = b.readlines()
c = open(file3, encoding="utf-8")
c = c.readlines()
a = a + b + c
a = [s for s in a if s != "\n"]

file = pd.read_excel('C:/Users/Frederik/Desktop/Result.xlsx', index_col=None, header=None)
file = file.set_axis(['ideal1', 'ideal2', 'ideal3', 'Value'], axis='columns')

# Ask = "Академический словарь литовского языка помогали составлять два президента, архиепископ, офтальмолог и ещё несколько сотен человек."
# ideal = list(file["ideal1"])

# Ask = "Один из главных злодеев трилогии приквелов «Звёздных войн» говорит с русским акцентом."
# ideal = list(file["ideal2"])

Ask = "Воспоминания Агаты Кристи об участии в археологических экспедициях в Ираке и Сирии долго не хотели печатать."
ideal = list(file["ideal3"])
print(Ask)
Ask = arrtodict(senttoterm(Ask))

result1 = vec(a, Ask)
result2 = prob(a, Ask, Lambda)

b = a.copy()
ar1 = []
ideal1 = ideal.copy()
for i in range(len(a)):
    ar1.append(ideal1[result1.index(max(result1))])
    ideal1.pop(result1.index(max(result1)))
    b.pop(result1.index(max(result1)))
    result1.pop(result1.index(max(result1)))


b = a.copy()
ar2 = []
ideal2 = ideal.copy()
for i in range(len(a)):
    ar2.append(ideal2[result2.index(max(result2))])
    ideal2.pop(result2.index(max(result2)))
    b.pop(result2.index(max(result2)))
    result2.pop(result2.index(max(result2)))

DCG1 = 0
for i in range(len(ar1)):
    DCG1 += ar1[i] / np.log(i + 2)

DCG2 = 0
for i in range(len(ar2)):
    DCG2 += ar2[i] / np.log(i + 2)

ideal.sort(reverse=True)
IDCG = 0
for i in range(len(ideal)):
    IDCG += ideal[i] / np.log(i + 2)


print(DCG1 / IDCG, DCG2 / IDCG)
