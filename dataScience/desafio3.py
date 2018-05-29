import pandas as pd
import numpy as np
import requests
import json
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

df = pd.read_csv('data/train.csv')
df = df.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_MT','NU_NOTA_REDACAO'])

notaSemMT = (df['NU_NOTA_LC'] + df['NU_NOTA_CH'] + df['NU_NOTA_CN'] + df['NU_NOTA_REDACAO'])
notaMT = df['NU_NOTA_MT']

x = notaSemMT.mean()
y = notaMT.mean()

beta = np.cov(m = notaSemMT, y = notaMT)[0][1]/notaSemMT.var()
alfa = y - beta*x

df2 = pd.read_csv('data/test3.csv')
df2 = df2.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO','NU_INSCRICAO','TX_RESPOSTAS_MT'])

notaMAT = alfa + beta*(df2['NU_NOTA_LC'] + df2['NU_NOTA_CH'] + df2['NU_NOTA_CN'] + df2['NU_NOTA_REDACAO'])
numeroInscricao = df2['NU_INSCRICAO']

inputToPredict = pd.DataFrame()
i = 1
while (i < 41):
    inputToPredict[str(i)] = df2['TX_RESPOSTAS_MT'].astype(str).str[i - 1]
    i = i + 1

inputToPredict['Nota'] = notaMAT
arrayPredict = inputToPredict.values
arrayPredict1D = arrayPredict.ravel()
i = 0
j = 0
while i < arrayPredict1D.size:
    try:
        arrayPredict1D[i] = ord(arrayPredict1D[i]) - ord('A') + 1
    except:
        j = j + 1
    finally:
        i = i + 1
arrayPredict2D = np.reshape(arrayPredict1D, (-1,41))

dftraining = pd.read_csv('data/train.csv')
dftraining = dftraining.dropna(subset= ['NU_NOTA_MT', 'TX_RESPOSTAS_MT'])

separarTuru = dftraining['TX_RESPOSTAS_MT'].astype(str).str[40:]

separar41 = dftraining['TX_RESPOSTAS_MT'].astype(str).str[40]
separar42 = dftraining['TX_RESPOSTAS_MT'].astype(str).str[41]
separar43 = dftraining['TX_RESPOSTAS_MT'].astype(str).str[42]
separar44 = dftraining['TX_RESPOSTAS_MT'].astype(str).str[43]
separar45 = dftraining['TX_RESPOSTAS_MT'].astype(str).str[44]

separarInput = pd.DataFrame()

i = 1
while (i < 41):
    separarInput[str(i)] = dftraining['TX_RESPOSTAS_MT'].astype(str).str[i - 1]
    i = i + 1

separarNota = dftraining['NU_NOTA_MT']
separarInput['Nota'] = separarNota

array = separarInput.values
array1D = array.ravel()

i, j = 0, 0
while i < array1D.size:
    try:
        array1D[i] = ord(array1D[i]) - ord('A')
    except:
        j = j + 1
    finally:
        i = i + 1

array2D = np.reshape(array1D, (-1,41))

activationMode = "relu"
solverMode = "adam"
alphaParameter = 0.0001
learningRateInit = 0.001
maxIter = 2000
tolRate = 0.0001
momentumRate = 0.999
batchSize = 10
learnRating = "adptive"
#Nao estou usando learning_rate

mlp1 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize)
mlp2 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize)
mlp3 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize)
mlp4 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize)
mlp5 = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize)
mlp1.fit(array2D, separar41)
print("Fit 1")
mlp2.fit(array2D, separar42)
print("Fit 2")
mlp3.fit(array2D, separar43)
print("Fit 3")
mlp4.fit(array2D, separar44)
print("Fit 4")
mlp5.fit(array2D, separar45)
print("Fit 5")

y1 = mlp1.predict(arrayPredict2D)
print("Predicted 1")
y2 = mlp2.predict(arrayPredict2D)
print("Predicted 2")
y3 = mlp3.predict(arrayPredict2D)
print("Predicted 3")
y4 = mlp4.predict(arrayPredict2D)
print("Predicted 4")
y5 = mlp5.predict(arrayPredict2D)
print("Predicted 5")


# y1 = ['.']*3173
# y2 = ['.']*3173
# y3 = ['.']*3173
# y4 = ['.']*3173
# y5 = ['.']*3173

saida = np.stack((y1, y2, y3 ,y4, y5), axis = -1)
print(saida.shape[0])
print(type(saida))
print(saida.shape[1])


i = 0
output = []
for index in numeroInscricao.index:
    increase = {}
    increase['NU_INSCRICAO'] = numeroInscricao[index]
    increase['TX_RESPOSTAS_MT'] = saida[i][0] + saida[i][1] + saida[i][2] + saida[i][3] + saida[i][4]
    i = i + 1
    output.append(increase)

# print(output)


#
# mlp = MLPClassifier(activation=activationMode,solver=solverMode,alpha=alphaParameter, learning_rate_init = learningRateInit, max_iter = maxIter, tol = tolRate, momentum = momentumRate, batch_size=batchSize, verbose= True)
# mlp.fit(array2D, separarTuru)
# print("Fit!")
# y = mlp.predict(arrayPredict2D)
# print("Predicted!")
# i = 0
# output = []
# for index in numeroInscricao.index:
#     increase = {}
#     increase['NU_INSCRICAO'] = numeroInscricao[index]
#     increase['TX_RESPOSTAS_MT'] = y[i]
#     i = i + 1
#     print(increase['TX_RESPOSTAS_MT'])
#     output.append(increase)

data = {
    'token': '1413af7a468c1b3c08e4644af0cc0bdb3cca2816',
    'email': 'igor.omote@gmail.com',
    'answer': output
}

data = json.dumps(data).encode("utf-8")

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

r = requests.post('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-3/submit', data = data)

print(r.text)
