import pandas as pd
import numpy as np
import requests
import json
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/train.csv')
df = df.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_MT','NU_NOTA_REDACAO'])
# df['NOTA_FINAL_MINUS_MT'] = ((1.5*df['NU_NOTA_LC'] + df['NU_NOTA_CH'] + 2*df['NU_NOTA_CN'] + 3*df['NU_NOTA_MT'] + 3*df['NU_NOTA_REDACAO'])/10.5)
notaSemMT = (df['NU_NOTA_LC'] + df['NU_NOTA_CH'] + df['NU_NOTA_CN'] + df['NU_NOTA_REDACAO'])
notaMT = df['NU_NOTA_MT']

lm = LinearRegression()
# print(type(notaSemMT.values))
lm.fit(notaSemMT.values, notaMT.values)

exit(1)

# x = notaSemMT.mean()
# print(df['NU_NOTA_MT'].head())
#
# print(x)
#
# y = notaMT.mean()
#
# print(y)
#
# print("wtf")
#
# beta = np.cov(m = notaSemMT, y = notaMT)[0][1]/notaSemMT.var()
# print (beta)
# alfa = y - beta*x
# print(alfa)

df2 = pd.read_csv('data/test2.csv')

df2 = df2.dropna(subset = ['NU_NOTA_LC','NU_NOTA_CH','NU_NOTA_CN','NU_NOTA_REDACAO','NU_INSCRICAO'])

s = alfa + beta*(df2['NU_NOTA_LC'] + df2['NU_NOTA_CH'] + df2['NU_NOTA_CN'] + df2['NU_NOTA_REDACAO'])

numeroInscricao = df2['NU_INSCRICAO'].dropna()

#
output = []
for index in s.index:
    increase = {}
    increase['NU_INSCRICAO'] = numeroInscricao[index]
    increase['NU_NOTA_MT'] = float(s[index])
    output.append(increase)

# print (output)

# output = [{"NU_INSCRICAO": n, "NU_NOTA_MT": m} for n, m in zip(numeroInscricao, s)]
#
# output = json.dumps(output).encode("utf-8")
#
data = {'token': '1413af7a468c1b3c08e4644af0cc0bdb3cca2816','email': 'igor.omote@gmail.com','answer': output
}

data = json.dumps(data).encode("utf-8")
# print(data)
r = requests.post('https://api.codenation.com.br/v1/user/acceleration/data-science/challenge/enem-2/submit', data = data)
#
print(r.text)
# print("Finish")
