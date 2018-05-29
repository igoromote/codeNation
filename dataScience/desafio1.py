import pandas as pd
import numpy as np
import requests
import json

def openCSV(link):
    df = pd.read_csv(link)
    return df

def get20HigherSubscription(df):
    subscription = []
    subscription = df['NU_INSCRICAO'].head(20)
    return subscription

def get20HigherFromDataFrame(df):
    higher = []
    higher = df['NOTA_FINAL'].head(20)
    return higher

# Uses as global variable
def calculateFinalGrade(df):
    df['NOTA_FINAL'] = ((1.5*df['NU_NOTA_LC'] + df['NU_NOTA_CH'] + 2*df['NU_NOTA_CN'] + 3*df['NU_NOTA_MT'] + 3*df['NU_NOTA_REDACAO'])/10.5)

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
calculateFinalGrade(df)
df = df.sort_values(by = 'NOTA_FINAL', ascending = False)
df['NOTA_FINAL'] = df['NOTA_FINAL'].round(decimals=1)
subscription = get20HigherSubscription(df)
topHigher = get20HigherFromDataFrame(df)
output = [{"NU_INSCRICAO": n, "NOTA_FINAL": m} for n, m in zip(subscription, topHigher)]
data = createDatagrama(output)
sendDatagram("https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-1/submit", data)
