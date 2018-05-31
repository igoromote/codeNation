import pandas as pd
import numpy as np
import requests
import json
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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

def transformLettersIntoNumbers(df):
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
            if arrayPredict1D[i] == 'A':
                arrayPredict1D[i] = -2
            elif arrayPredict1D[i] == 'B':
                arrayPredict1D[i] = -1
            elif arrayPredict1D[i] == 'C':
                arrayPredict1D[i] = 0
            elif arrayPredict1D[i] == 'D':
                arrayPredict1D[i] = 1
            elif arrayPredict1D[i] == 'E':
                arrayPredict1D[i] = 2
            elif arrayPredict1D[i] == '.' or arrayPredict1D[i] == '*':
                arrayPredict1D[i] = 0
                print(i)
        except:
            j = j + 1
        finally:
            i = i + 1
    arrayPredict2D = np.reshape(arrayPredict1D, (-1,41))
    return arrayPredict2D

def getLast5Answers(df):
    all5 = df['TX_RESPOSTAS_MT'].astype(str).str[40:]
    answer41 = df['TX_RESPOSTAS_MT'].astype(str).str[40]
    answer42 = df['TX_RESPOSTAS_MT'].astype(str).str[41]
    answer43 = df['TX_RESPOSTAS_MT'].astype(str).str[42]
    answer44 = df['TX_RESPOSTAS_MT'].astype(str).str[43]
    answer45 = df['TX_RESPOSTAS_MT'].astype(str).str[44]
    return all5,answer41,answer42,answer43,answer44,answer45

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
dftr = filterDotsAndAsterics(dftr)
dfToTrain = getFirst40Answers(dftr)
dfToTrain['Nota'] = dftr['NU_NOTA_MT']
all5,a41,a42,a43,a44,a45 = getLast5Answers(dftr)
dfToTrain = transformLettersIntoNumbers(dfToTrain)

activationMode = "logistic"
solverMode = "adam"
alphaParameter = 0.0001
learningRateInit = 0.001
maxIter = 2000
tolRate = 0.001
momentumRate = 0.99
batchSize = 10
learnRating = "adptive"
talkative = True

mlp1 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize, verbose = talkative)
mlp2 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize, verbose = talkative)
mlp4 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize, verbose = talkative)
mlp3 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize, verbose = talkative)
mlp5 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize, verbose = talkative)
mlp1.fit(dfToTrain, a41)
print("Fit 1")
mlp2.fit(dfToTrain, a42)
print("Fit 2")
mlp3.fit(dfToTrain, a43)
print("Fit 3")
mlp4.fit(dfToTrain, a44)
print("Fit 4")
mlp5.fit(dfToTrain, a45)
print("Fit 5")

# Data to Predict
df2 = openCSV('data/test3.csv')
df2 = df2.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO','NU_INSCRICAO'])

dataToPredict = createDataFrameToRL(df2)
dfSubscription = df2['NU_INSCRICAO']

dfResults = lm.predict(dataToPredict)

inputToPredict = getFirst40Answers(df2)
inputToPredict['Nota'] = dfResults
inputToPredict = transformLettersIntoNumbers(inputToPredict)


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
