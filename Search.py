import pymorphy2
morph = pymorphy2.MorphAnalyzer()
import numpy as np
import string

string.punctuation += "«"
string.punctuation += "»"
string.punctuation += "—"


# Определение функции TF
def TF(x):
   #return np.log(x + 1)
   return x

# Функция, создающая из предложения массив слов, обработанных морфологическим анализатором.
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

# Функция, создающая словарь слов из общего массива слов, обработанных морфологическим анализатором. Ключом является слово, значением является частота этого слова
def arrtodict(terms):
    word_list = {}
    for word in terms:
        if word in word_list:
            word_list[word] += 1
        else:
            word_list[word] = 1
    return word_list

# Пути исходных статей Википедии
file1 = "C:\\Users\\Frederik\\Desktop\\1.txt"
file2 = "C:\\Users\\Frederik\\Desktop\\2.txt"
file3 = "C:\\Users\\Frederik\\Desktop\\3.txt"

# Считывание всех файлов и сбор их в один массив
a = open(file1, encoding="utf-8")
a = a.readlines()
b = open(file2, encoding="utf-8")
b = b.readlines()
c = open(file3, encoding="utf-8")
c = c.readlines()
a = a + b + c
# Удаление пустых строк из массива предложений
a = [s for s in a if s != "\n"]
# Общее число предложений
N = len(a)
s = []
col = []
# Цикл создающий общий массив слов и словари для каждого предложения (Doc1, Doc2 ...)
for i in a:
    t = senttoterm(i)
    col.append(arrtodict(t))
    s += senttoterm(i)
df = arrtodict(s)
# Создание множества всех слов
dict = set(s)
# Создание словаря слов idf, в котором ключом является слово, а значением idf этого слова, посчитанного по формуле N / df[i]
idf = {}
for i in df:
    idf[i] = N / df[i]

# Создание матрицы tf.idf, в котором каждая строка - это предложение, а стобец - слово
Docs = np.zeros((N, len(dict)))

for i in range(N):
    k = 0
    for j in dict:
        if j in col[i]:
            Docs[i][k] = idf[j] * TF(col[i][j])
        else:
            Docs[i][k] = 0
        k += 1

# Нормировка полученных векторов
for i in range(N):
    Docs[i] = Docs[i] / np.linalg.norm(Docs[i])

# Строка с запросом

#Ask = "Академический словарь литовского языка помогали составлять два президента, архиепископ, офтальмолог и ещё несколько сотен человек."
#Ask = "Один из главных злодеев трилогии приквелов «Звёздных войн» говорит с русским акцентом."
Ask = "Воспоминания Агаты Кристи об участии в археологических экспедициях в Ираке и Сирии долго не хотели печатать."

# Разбиение запроса на термы и создание из них словаря
Ask = arrtodict(senttoterm(Ask))
# Создание вектора запроса
A = []
for i in dict:
    if i in Ask:
        A.append(idf[i] * TF(Ask[i]))
    else:
        A.append(0)

# Нормировка вектора запроса
A = np.array(A)
A = A / np.linalg.norm(A)

# Массив содержащий все веса
cos = []

# Подсчет весов для каждого предложения из статей
for i in range(N):
    cos.append(np.dot(A, Docs[i])/(np.linalg.norm(A) * np.linalg.norm(Docs[i])))

# Вывод 1/5 всех предложений в порядке уменьшения веса
for i in range(10):
    print(a[cos.index(max(cos))], max(cos))
    a.pop(cos.index(max(cos)))
    cos.pop(cos.index(max(cos)))
