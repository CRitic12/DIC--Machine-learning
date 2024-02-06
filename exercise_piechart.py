import matplotlib.pyplot as plt
import pandas as pd
#import sns
from sklearn.decomposition import PCA

df=pd.read_csv("Ninapro_DB1.csv")

plot_column='exercise'

value_count=df[plot_column].value_counts()

plt.figure(figsize=(8,8))
plt.pie(value_count,labels=value_count.index,autopct='%1.1f%%',startangle=140)
plt.title(f'Pie chart for {plot_column}')
plt.show()


pca = PCA(n_components=2)
X_r = pca.fit_transform(df.drop([plot_column],axis=1))

x=X_r[:,0]
y=X_d=X_r[:,1]

plt.scatter(x, y, c=df[plot_column])
plt.colorbar()
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Principal Components Analysis of Ninapro DB1')
plt.show()


