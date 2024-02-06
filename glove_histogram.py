import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df= pd.read_csv("Ninapro_DB1.csv")
print(df)

#make a variable cols which contains all columns from glove_0 to glove_21
cols = [col for col in df.columns if 'glove' in col]

for label in cols[:-1]:
    plt.hist(df[df['exercise']==1][label],color='red',label='exercise1',alpha=0.4,density=True)
    plt.hist(df[df['exercise']==2][label],color='green',label='exercise2',alpha=0.4,density=True)
    plt.hist(df[df['exercise']==3][label],color='blue',label='exercise3',alpha=0.4,density=True)
    plt.title(label)
    plt.xlabel(label)
    plt.ylabel("probability")
    plt.legend()
    plt.show()





































