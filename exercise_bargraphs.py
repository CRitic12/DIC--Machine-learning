import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Ninapro_DB1.csv")

plot_column = 'exercise'

#value counts
value_counts = df[plot_column].value_counts().sort_index()

#bar graph
plt.figure(figsize=(15, 8))
value_counts.plot(kind='bar', color='blue')
plt.xlabel(plot_column)
plt.ylabel('Count')
plt.title(f'Bar graph for {plot_column}')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.show()





