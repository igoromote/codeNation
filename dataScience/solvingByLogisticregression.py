import pandas as pd
import numpy as np
import requests
import json
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, linear_model
from sklearn.linear_model import LogisticRegression

def openCSV(link):
    df = pd.read_csv(link)
    return df

def createDataFrameToRL(df):
    gradeWithoutMT = pd.DataFrame()
    gradeWithoutMT['LC'], gradeWithoutMT['CN'], gradeWithoutMT['CH'], gradeWithoutMT['RED'] = df['NU_NOTA_LC'] ,df['NU_NOTA_CH'], df['NU_NOTA_CN'], df['NU_NOTA_REDACAO']
    try:
        gradeMT = df['NU_NOTA_MT']
        return gradeWithoutMT, gradeMT
    except:
        return gradeWithoutMT


def createDatagrama(output):
    data = {
      "token": "1413af7a468c1b3c08e4644af0cc0bdb3cca2816",
      "email": "igor.omote@gmail.com",
      "answer": output
    }
    data = json.dumps(data).encode("utf-8")
    return data

def sendDatagram(link, datagram):
    r = requests.post(link, data = datagram)
    print(r.text)

def getFirst40Answers (df):
    inputToPredict = pd.DataFrame()
    i = 1
    while (i < 41):
        inputToPredict[str(i)] = df['TX_RESPOSTAS_MT'].astype(str).str[i - 1]
        i = i + 1
    return inputToPredict

def getLast5Answers(df):
    a41 = df['TX_RESPOSTAS_MT'].astype(str).str[40]
    a42 = df['TX_RESPOSTAS_MT'].astype(str).str[41]
    a43 = df['TX_RESPOSTAS_MT'].astype(str).str[42]
    a44 = df['TX_RESPOSTAS_MT'].astype(str).str[43]
    a45 = df['TX_RESPOSTAS_MT'].astype(str).str[44]
    return a41, a42, a43, a44, a45

def transformIndexToLetter(indexArray):
    for i in range(0, indexArray.size):
        indexArray[i] = (ord('A') + indexArray[i])
    indexArray = map(chr, indexArray)
    return indexArray

def createAnswer(dfSubscription, arrayResults):
    output = []
    i = 0
    for index in dfSubscription.index:
        increase = {}
        increase['NU_INSCRICAO'] = dfSubscription[index]
        increase['TX_RESPOSTAS_MT'] = arrayResults[i][0] + arrayResults[i][1] + arrayResults[i][2] + arrayResults[i][3] + arrayResults[i][4]
        i = i + 1
        output.append(increase)
    return output

def filterDotsAndAsterics(df):
    toDrop = []
    print(df.shape[0])
    for i in range(0,df.shape[0]):
        try:
            array = df.iloc[[i]]['TX_RESPOSTAS_MT'].to_string()
            if array.find('.') != -1:
                toDrop.append(i)
            elif array.find('*') != -1:
                toDrop.append(i)
        except:
            break
    print("Finish")
    df = df.drop(df.index[toDrop])
    return df

def transformLettersIntoNumbers(df):
    arrayPredict = df.values
    arrayPredict1D = arrayPredict.ravel()
    i = 0
    j = 0
    while i < arrayPredict1D.size:
        try:
            if arrayPredict1D[i] == 'A':
                arrayPredict1D[i] = 1
            elif arrayPredict1D[i] == 'B':
                arrayPredict1D[i] = 2
            elif arrayPredict1D[i] == 'C':
                arrayPredict1D[i] = 3
            elif arrayPredict1D[i] == 'D':
                arrayPredict1D[i] = 4
            elif arrayPredict1D[i] == 'E':
                arrayPredict1D[i] = 5
            elif arrayPredict1D[i] == '.':
                arrayPredict1D[i] = 0.3
            elif arrayPredict1D[i] == '*':
                arrayPredict1D[i] = -0.3
        except:
            j = j + 1
        finally:
            i = i + 1
    arrayPredict2D = np.reshape(arrayPredict1D, (-1,40))
    return arrayPredict2D

# Find grade of MT
df = openCSV('data/train.csv')
df = df.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_MT','NU_NOTA_REDACAO'])
gradeWithoutMT, gradeMT = createDataFrameToRL(df)

lm = LinearRegression()
lm.fit(gradeWithoutMT.values, gradeMT.values)

# Train Neural Network
dftr = openCSV('data/train.csv')
dftr = dftr.dropna(subset= ['NU_NOTA_MT', 'TX_RESPOSTAS_MT'])
# dftr = filterDotsAndAsterics(dftr)
dfToTrain = getFirst40Answers(dftr)
# dfToTrain['Nota'] = dftr['NU_NOTA_MT']
a41, a42, a43, a44, a45 = getLast5Answers(dftr)

#parameters
C = 1.0
penalt = 'l1'
stopFit = 0.0001
solverMode = 'liblinear'
maxIter = 200
multiClass = 'ovr'
talk = False
reuse = False


arrayToFit = transformLettersIntoNumbers(dfToTrain)
LR41 = LogisticRegression(C=C, penalty=penalt, tol = stopFit, solver = solverMode, max_iter = maxIter, multi_class = multiClass, verbose = talk, warm_start = reuse)
LR42 = LogisticRegression(C=C, penalty=penalt, tol = stopFit, solver = solverMode, max_iter = maxIter, multi_class = multiClass, verbose = talk, warm_start = reuse)
LR43 = LogisticRegression(C=C, penalty=penalt, tol = stopFit, solver = solverMode, max_iter = maxIter, multi_class = multiClass, verbose = talk, warm_start = reuse)
LR44 = LogisticRegression(C=C, penalty=penalt, tol = stopFit, solver = solverMode, max_iter = maxIter, multi_class = multiClass, verbose = talk, warm_start = reuse)
LR45 = LogisticRegression(C=C, penalty=penalt, tol = stopFit, solver = solverMode, max_iter = maxIter, multi_class = multiClass, verbose = talk, warm_start = reuse)

print("go fit")
LR41.fit(arrayToFit, a41)
LR42.fit(arrayToFit, a42)
LR43.fit(arrayToFit, a43)
LR44.fit(arrayToFit, a44)
LR45.fit(arrayToFit, a45)
print("finish fit")

# Data to Predict
df2 = openCSV('data/test3.csv')
df2 = df2.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO','NU_INSCRICAO'])

dataToPredict = createDataFrameToRL(df2)
dfSubscription = df2['NU_INSCRICAO']

dfResults = lm.predict(dataToPredict)

inputToPredict = getFirst40Answers(df2)
# inputToPredict['Nota'] = dfResults
arrayToPredict = transformLettersIntoNumbers(inputToPredict)


answer41 = LR41.predict(arrayToPredict)
answer42 = LR42.predict(arrayToPredict)
answer43 = LR43.predict(arrayToPredict)
answer44 = LR44.predict(arrayToPredict)
answer45 = LR45.predict(arrayToPredict)

print("Score 41: "+ str(LR41.score(arrayToFit, a41)))
print("Score 42: "+ str(LR42.score(arrayToFit, a42)))
print("Score 43: "+ str(LR43.score(arrayToFit, a43)))
print("Score 44: "+ str(LR44.score(arrayToFit, a44)))
print("Score 45: "+ str(LR45.score(arrayToFit, a45)))

results = np.stack((answer41, answer42, answer43, answer44, answer45), axis = -1)
output = createAnswer(dfSubscription, results)
data = createDatagrama(output)
# print(data)

sendDatagram('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit', data)
