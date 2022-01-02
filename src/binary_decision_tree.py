# Load libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import seaborn as sns
sns.set()
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import accuracy_score
import statistics as stat 
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import numpy as np 

filesAdd = ['C2G1', 'C1A2', 'C2G2', 'C2E1', 'C1F2', 'C1E2','B2G2', 'B1A2', 'B2A2', 'B2A1','B1G3','B1E1','B2C1','B1A1','E2A2', 'E1A2', 'E3G4', 'E2G4', 'E3A2', 'E1G1', 'E3A3', 'E1A1', 'E2B2', 'E2G3', 'E3A4', 'E3G3', 'E1A4', 'E3F4', 'E2B5', 'E2A1', 'E1F5', 'F3A3', 'F2B2', 'F2B1', 'F3E4', 'F1A3', 'F3A2', 'F1E3', 'F1B5', 'F3F5', 'F1F3', 'F1E4', 'F1F4', 'F2F5', 'F2E4', 'F1F2', 'F2A4', 'F3B5', 'F2G2']
files = []
for file in filesAdd:
	df = pd.read_csv("Binary Data/" + file+".csv")
	df.isnull().any()
	df = df.fillna(method='ffill')
	files.append(df)

X = []
y = []

# make x and y
for f in files:	
	n = 0
	#list of lists
	tripleL = []
	#0 or 1 distracted
	binary = []
	fiveData = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]

	fiveY = []
	counter = 0
	for index, row in f.iterrows():
		counter = counter + 1
		if counter == 5:
			counter = 0
			tempArray = []
			for colData in fiveData:
				tempArray.append(stat.mean(colData))
			X.append(tempArray)
			fiveData = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
			y.append(stat.mode(fiveY)) 
			fiveY = []
		# temp = list(tripleL.appen)		
		fiveY.append(int(row['distracted?']))
		columns =  ['delta_relative_1','delta_relative_2', 'delta_relative_3', 'delta_relative_4', 'alpha_relative_1','alpha_relative_2', 'alpha_relative_3', 'alpha_relative_4', 'beta_relative_1','beta_relative_2', 'beta_relative_3', 'beta_relative_4', 'theta_relative_1','theta_relative_2', 'theta_relative_3', 'theta_relative_4', 'gamma_relative_1','gamma_relative_2', 'gamma_relative_3', 'gamma_relative_4']
		counter2 = 0 
		for col in columns:
			fiveData[counter2].append(row[col])
			counter2 = counter2 + 1 
		n = n + 1

accuracy = []
#add'
#for loopn in range(0, 100):
#undersample
sampling_strategy =1
rus = RandomUnderSampler(sampling_strategy = sampling_strategy)
X, y = rus.fit_resample(X, y)


# training our model 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

#### METHOD 1 #####
# # Create Decision Tree classifer object
# clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
# clf = clf.fit(X_train,y_train)

# #Predict the response for test dataset
# y_pred = clf.predict(X_test)

#### METHOD 2 #####
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=5)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy:",acc)
accuracy.append(acc)
print()
# 	# Plot the confusion matrix
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
class_names = [1,2,3,4,5]
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
# print("Average accuracy: ", stat.mean(accuracy))
# print("Standard deviation: ", stat.stdev(accuracy))
# Print the confusion matrix
