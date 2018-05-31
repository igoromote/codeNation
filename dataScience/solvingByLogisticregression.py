import pandas as pd
import numpy as np
import requests
import json
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import datasets, neighbors, linear_model

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

def transformLettersIntoNumbers(df, letter, rowSize):
    arrayPredict = df.values
    arrayPredict1D = arrayPredict.ravel()
    i = 0
    j = 0
    while i < arrayPredict1D.size:
        try:
            # batata = arrayPredict1D[i]
            # arrayPredict1D[i] = (ord(arrayPredict1D[i]) - ord('A') + 1)
            # if (arrayPredict1D[i] < 0):
            #     print(arrayPredict1D[i])
            #     print(i)
            #     print(batata)
            if arrayPredict1D[i] == letter:
                arrayPredict1D[i] = 1
            else:
                arrayPredict1D[i] = 0
        except:
            j = j + 1
        finally:
            i = i + 1
    if rowSize > 1:
        arrayPredict2D = np.reshape(arrayPredict1D, (-1,rowSize))
        return arrayPredict2D
    else:
        return arrayPredict1D

def getLast5Answers(df):
    all5 = df['TX_RESPOSTAS_MT'].astype(str).str[40:]
    answer41 = df['TX_RESPOSTAS_MT'].astype(str).str[40]
    answer42 = df['TX_RESPOSTAS_MT'].astype(str).str[41]
    answer43 = df['TX_RESPOSTAS_MT'].astype(str).str[42]
    answer44 = df['TX_RESPOSTAS_MT'].astype(str).str[43]
    answer45 = df['TX_RESPOSTAS_MT'].astype(str).str[44]
    return all5,answer41,answer42,answer43,answer44,answer45

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
dfToTrain['Nota'] = dftr['NU_NOTA_MT']
all5A,a41A,a42A,a43A,a44A,a45A = getLast5Answers(dftr)
all5B,a41B,a42B,a43B,a44B,a45B = getLast5Answers(dftr)
all5C,a41C,a42C,a43C,a44C,a45C = getLast5Answers(dftr)
all5D,a41D,a42D,a43D,a44D,a45D = getLast5Answers(dftr)
all5E,a41E,a42E,a43E,a44E,a45E = getLast5Answers(dftr)
dfToTrainA = transformLettersIntoNumbers(dfToTrain, 'A', 41)
dfToTrainB = transformLettersIntoNumbers(dfToTrain, 'B', 41)
dfToTrainC = transformLettersIntoNumbers(dfToTrain, 'C', 41)
dfToTrainD = transformLettersIntoNumbers(dfToTrain, 'D', 41)
dfToTrainE = transformLettersIntoNumbers(dfToTrain, 'E', 41)
df41A = transformLettersIntoNumbers(a41A, 'A', 1).astype('float')
df41B = transformLettersIntoNumbers(a41B, 'B', 1).astype('float')
df41C = transformLettersIntoNumbers(a41C, 'C', 1).astype('float')
df41D = transformLettersIntoNumbers(a41D, 'D', 1).astype('float')
df41E = transformLettersIntoNumbers(a41E, 'E', 1).astype('float')
df42A = transformLettersIntoNumbers(a42A, 'A', 1).astype('float')
df42B = transformLettersIntoNumbers(a42B, 'B', 1).astype('float')
df42C = transformLettersIntoNumbers(a42C, 'C', 1).astype('float')
df42D = transformLettersIntoNumbers(a42D, 'D', 1).astype('float')
df42E = transformLettersIntoNumbers(a42E, 'E', 1).astype('float')
df43A = transformLettersIntoNumbers(a43A, 'A', 1).astype('float')
df43B = transformLettersIntoNumbers(a43B, 'B', 1).astype('float')
df43C = transformLettersIntoNumbers(a43C, 'C', 1).astype('float')
df43D = transformLettersIntoNumbers(a43D, 'D', 1).astype('float')
df43E = transformLettersIntoNumbers(a43E, 'E', 1).astype('float')
df44A = transformLettersIntoNumbers(a44A, 'A', 1).astype('float')
df44B = transformLettersIntoNumbers(a44B, 'B', 1).astype('float')
df44C = transformLettersIntoNumbers(a44C, 'C', 1).astype('float')
df44D = transformLettersIntoNumbers(a44D, 'D', 1).astype('float')
df44E = transformLettersIntoNumbers(a44E, 'E', 1).astype('float')
df45A = transformLettersIntoNumbers(a45A, 'A', 1).astype('float')
df45B = transformLettersIntoNumbers(a45B, 'B', 1).astype('float')
df45C = transformLettersIntoNumbers(a45C, 'C', 1).astype('float')
df45D = transformLettersIntoNumbers(a45D, 'D', 1).astype('float')
df45E = transformLettersIntoNumbers(a45E, 'E', 1).astype('float')
#
knn41A = neighbors.KNeighborsClassifier()
knn41A.fit(dfToTrainA, df41A)
knn41B = neighbors.KNeighborsClassifier()
knn41B.fit(dfToTrainB, df41B)
knn41C = neighbors.KNeighborsClassifier()
knn41C.fit(dfToTrainC, df41C)
knn41D = neighbors.KNeighborsClassifier()
knn41D.fit(dfToTrainD, df41D)
knn41E = neighbors.KNeighborsClassifier()
knn41E.fit(dfToTrainE, df41E)

knn42A = neighbors.KNeighborsClassifier()
knn42A.fit(dfToTrainA, df42A)
knn42B = neighbors.KNeighborsClassifier()
knn42B.fit(dfToTrainB, df42B)
knn42C = neighbors.KNeighborsClassifier()
knn42C.fit(dfToTrainC, df42C)
knn42D = neighbors.KNeighborsClassifier()
knn42D.fit(dfToTrainD, df42D)
knn42E = neighbors.KNeighborsClassifier()
knn42E.fit(dfToTrainE, df42E)

knn43A = neighbors.KNeighborsClassifier()
knn43A.fit(dfToTrainA, df43A)
knn43B = neighbors.KNeighborsClassifier()
knn43B.fit(dfToTrainB, df43B)
knn43C = neighbors.KNeighborsClassifier()
knn43C.fit(dfToTrainC, df43C)
knn43D = neighbors.KNeighborsClassifier()
knn43D.fit(dfToTrainD, df43D)
knn43E = neighbors.KNeighborsClassifier()
knn43E.fit(dfToTrainE, df43E)

knn44A = neighbors.KNeighborsClassifier()
knn44A.fit(dfToTrainA, df44A)
knn44B = neighbors.KNeighborsClassifier()
knn44B.fit(dfToTrainB, df44B)
knn44C = neighbors.KNeighborsClassifier()
knn44C.fit(dfToTrainC, df44C)
knn44D = neighbors.KNeighborsClassifier()
knn44D.fit(dfToTrainD, df44D)
knn44E = neighbors.KNeighborsClassifier()
knn44E.fit(dfToTrainE, df44E)

knn45A = neighbors.KNeighborsClassifier()
knn45A.fit(dfToTrainA, df45A)
knn45B = neighbors.KNeighborsClassifier()
knn45B.fit(dfToTrainB, df45B)
knn45C = neighbors.KNeighborsClassifier()
knn45C.fit(dfToTrainC, df45C)
knn45D = neighbors.KNeighborsClassifier()
knn45D.fit(dfToTrainD, df45D)
knn45E = neighbors.KNeighborsClassifier()
knn45E.fit(dfToTrainE, df45E)

# Data to Predict
df2 = openCSV('data/test3.csv')
df2 = df2.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO','NU_INSCRICAO'])

dataToPredict = createDataFrameToRL(df2)
dfSubscription = df2['NU_INSCRICAO']

dfResults = lm.predict(dataToPredict)

inputToPredict = getFirst40Answers(df2)
inputToPredict['Nota'] = dfResults
inputToPredictA = transformLettersIntoNumbers(inputToPredict, 'A', 41)
inputToPredictB = transformLettersIntoNumbers(inputToPredict, 'B', 41)
inputToPredictC = transformLettersIntoNumbers(inputToPredict, 'C', 41)
inputToPredictD = transformLettersIntoNumbers(inputToPredict, 'D', 41)
inputToPredictE = transformLettersIntoNumbers(inputToPredict, 'E', 41)

is41A = knn41A.predict_proba(inputToPredictA)[:, 1]
is41B = knn41B.predict_proba(inputToPredictB)[:, 1]
is41C = knn41C.predict_proba(inputToPredictC)[:, 1]
is41D = knn41D.predict_proba(inputToPredictD)[:, 1]
is41E = knn41E.predict_proba(inputToPredictE)[:, 1]

is42A = knn42A.predict_proba(inputToPredictA)[:, 1]
is42B = knn42B.predict_proba(inputToPredictB)[:, 1]
is42C = knn42C.predict_proba(inputToPredictC)[:, 1]
is42D = knn42D.predict_proba(inputToPredictD)[:, 1]
is42E = knn42E.predict_proba(inputToPredictE)[:, 1]

is43A = knn43A.predict_proba(inputToPredictA)[:, 1]
is43B = knn43B.predict_proba(inputToPredictB)[:, 1]
is43C = knn43C.predict_proba(inputToPredictC)[:, 1]
is43D = knn43D.predict_proba(inputToPredictD)[:, 1]
is43E = knn43E.predict_proba(inputToPredictE)[:, 1]

is44A = knn44A.predict_proba(inputToPredictA)[:, 1]
is44B = knn44B.predict_proba(inputToPredictB)[:, 1]
is44C = knn44C.predict_proba(inputToPredictC)[:, 1]
is44D = knn44D.predict_proba(inputToPredictD)[:, 1]
is44E = knn44E.predict_proba(inputToPredictE)[:, 1]

is45A = knn45A.predict_proba(inputToPredictA)[:, 1]
is45B = knn45B.predict_proba(inputToPredictB)[:, 1]
is45C = knn45C.predict_proba(inputToPredictC)[:, 1]
is45D = knn45D.predict_proba(inputToPredictD)[:, 1]
is45E = knn45E.predict_proba(inputToPredictE)[:, 1]

results41 = np.stack((is41A, is41B, is41C,is41D, is41E), axis = -1)
results42 = np.stack((is42A, is42B, is42C,is42D, is42E), axis = -1)
results43 = np.stack((is43A, is43B, is43C,is43D, is43E), axis = -1)
results44 = np.stack((is44A, is44B, is44C,is44D, is44E), axis = -1)
results45 = np.stack((is45A, is45B, is45C,is45D, is45E), axis = -1)

result41 = results41.argmax(axis=1)
result42 = results42.argmax(axis=1)
result43 = results43.argmax(axis=1)
result44 = results44.argmax(axis=1)
result45 = results45.argmax(axis=1)

result41 = transformIndexToLetter(result41)
result42 = transformIndexToLetter(result42)
result43 = transformIndexToLetter(result43)
result44 = transformIndexToLetter(result44)
result45 = transformIndexToLetter(result45)

results = np.stack((result41, result42, result43, result44, result45), axis = -1)

output = createAnswer(dfSubscription, results)
data = createDatagrama(output)
print(data)

print("\n\n\nY1")
unique,counts = np.unique(result41,return_counts=True)
print np.asarray((unique, counts)).T
print("Y2")
unique,counts = np.unique(result42,return_counts=True)
print np.asarray((unique, counts)).T
print("Y3")
unique,counts = np.unique(result43,return_counts=True)
print np.asarray((unique, counts)).T
print("Y4")
unique,counts = np.unique(result44,return_counts=True)
print np.asarray((unique, counts)).T
print("Y5")
unique,counts = np.unique(result45,return_counts=True)
print np.asarray((unique, counts)).T

sendDatagram('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit', data)

# print(result41)
# print(result42)
# print(result43)
# print(result44)
# print(result45)
exit(1)


y1 = mlp1.predict(inputToPredict)
print("Predicted 1")
y2 = mlp2.predict(inputToPredict)
print("Predicted 2")
y3 = mlp3.predict(inputToPredict)
print("Predicted 3")
y4 = mlp4.predict(inputToPredict)
print("Predicted 4")
y5 = mlp5.predict(inputToPredict)
print("Predicted 5")
# y1 = ['.']*3173
# y2 = ['.']*3173
# y3 = ['.']*3173
# y4 = ['.']*3173
# y5 = ['.']*3173

results = np.stack((y1, y2, y3 ,y4, y5), axis = -1)
output = createAnswer(dfSubscription, results)
data = createDatagrama(output)

print("Y1")
unique,counts = np.unique(y1,return_counts=True)
print np.asarray((unique, counts)).T
print("Y2")
unique,counts = np.unique(y2,return_counts=True)
print np.asarray((unique, counts)).T
print("Y3")
unique,counts = np.unique(y3,return_counts=True)
print np.asarray((unique, counts)).T
print("Y4")
unique,counts = np.unique(y4,return_counts=True)
print np.asarray((unique, counts)).T
print("Y5")
unique,counts = np.unique(y5,return_counts=True)
print np.asarray((unique, counts)).T

sendDatagram('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit', data)
