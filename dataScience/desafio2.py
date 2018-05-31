# IDEIAS PARA MELHORAR:
# ENCONTRAR PONTOS FORA DA CURVA E TIRAR (PLOTAR EM CADA EIXO OU FAZER FILTRO GROSSO)
import pandas as pd
import numpy as np
import requests
import json
from sklearn.linear_model import LinearRegression

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

def createAnswer(dfSubscription, dfResults):
    output = []
    i = 0
    for index in dfSubscription.index:
        increase = {}
        increase['NU_INSCRICAO'] = dfSubscription[index]
        increase['NU_NOTA_MT'] = dfResults[i]
        i = i + 1
        output.append(increase)
    return output

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

df = openCSV('data/train.csv')
df = df.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_MT','NU_NOTA_REDACAO'])
gradeWithoutMT, gradeMT = createDataFrameToRL(df)

lm = LinearRegression()
lm.fit(gradeWithoutMT.values, gradeMT.values)

df2 = openCSV('data/test2.csv')
df2 = df2.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO','NU_INSCRICAO'])

dataToPredict = createDataFrameToRL(df2)
dfSubscription = df2['NU_INSCRICAO']

dfResults = lm.predict(dataToPredict)

output = createAnswer(dfSubscription, dfResults)
data = createDatagrama(output)
sendDatagram("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-2/submit", data)
