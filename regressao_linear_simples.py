import numpy
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv ('dados_regressao_linear_simples.csv')

x = dataset.iloc[:, :-1].values
# print (x)
y = dataset.iloc[:, -1].values
# print (y)

x_treinamento, x_teste, y_treinamento, y_teste = train_test_split(x, y, test_size=0.2, random_state=0)

linearRegression = LinearRegression()
linearRegression.fit(x_treinamento, y_treinamento)

y_pred = linearRegression.predict(x_treinamento)

# plt.scatter(x_treinamento, y_treinamento, color="red")
# plt.plot (x_treinamento, y_pred, color="blue")
# plt.title ("Salário x Tempo de Experiência (Treinamento")
# plt.xlabel ("Anos de Experiência")
# plt.ylabel ("Salário")
# plt.show()


# plt.scatter (x_teste, y_teste, color="red")
# plt.plot (x_treinamento, y_pred, color="blue")
# plt.title("Salário x Tempo de Experiência (Teste")
# plt.xlabel ("Anos de Experiência")
# plt.ylabel ("Salário")
# plt.show()

# print (f"15.7 anos: {linearRegression.predict([ [15.7] ])}")
# print (f"10.5 anos: {linearRegression.predict([ [10.5] ])}")
# print (f"0 anos: {linearRegression.predict([ [0] ])}")
# print (f"5 anos: {linearRegression.predict([ [5] ])}")


print(f'y={linearRegression.coef_[0]:.2f}x + {linearRegression.intercept_:.2f}')

