from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split
from sklearn import metrics
# from sklearn.cross_validation import train_test_split
import pandas as pd
import statistics as stat
import numpy as np

f1 = pd.read_csv("change_Cleah1.csv")
f2 = pd.read_csv("change_Cleah2.csv")
f3 = pd.read_csv("change_Cleah3.csv")
f4 = pd.read_csv("change_Cleah4.csv")
cf1 = pd.read_csv("change_Cailin1.csv")
cf2 = pd.read_csv("change_Cailin2.csv")
files = [f1, f2, f3, f4, cf1, cf2]
X = []
y = []

for f in files:	
	n = 0
	#list of lists
	tripleL = []
	#0 or 1 distracted
	binary = []
	for index, row in f.iterrows():
		# temp = list(tripleL.appen)		
		y.append(row['distracted?'])
		X.append((row["DELTA"], row["ALPHA"], row["BETA"], row["GAMMA"], row["DELTArel"], row["ALPHArel"], row["BETArel"], row["GAMMArel"]))
		if (row['distracted?'] == 1):
			y.append(row['distracted?'])
			X.append((row["DELTA"], row["ALPHA"], row["BETA"], row["GAMMA"], row["DELTArel"], row["ALPHArel"], row["BETArel"], row["GAMMArel"]))
			# y.append(row['distracted?'])
			# X.append((row["DELTA"], row["ALPHA"], row["BETA"], row["GAMMA"], row["DELTArel"], row["ALPHArel"], row["BETArel"], row["GAMMArel"]))
			# y.append(row['distracted?'])
			# X.append((row["DELTA"], row["ALPHA"], row["BETA"], row["GAMMA"], row["DELTArel"], row["ALPHArel"], row["BETArel"], row["GAMMArel"]))

		n = n + 1

# nsamples, nx, ny = X.shape
# X = X.reshape((nsamples, nx*ny))
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
# instantiate the model (using the default parameters)
logreg = LogisticRegression()
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
# %matplotlib inline
class_names = [0,1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu', fmt = 'g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print(logreg.score())