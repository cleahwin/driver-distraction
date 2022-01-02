import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
import statistics as stat

filesAdd = ['C2G1', 'C1A2', 'C2G2', 'C2E1', 'C1F2', 'C1E2','B2G2', 'B1A2', 'B2A2', 'B2A1','B1G3','B1E1','B2C1','B1A1','E2A2', 'E1A2', 'E3G4', 'E2G4', 'E3A2', 'E1G1', 'E3A3', 'E1A1', 'E2B2', 'E2G3', 'E3A4', 'E3G3', 'E1A4', 'E3F4', 'E2B5', 'E2A1', 'E1F5', 'F3A3', 'F2B2', 'F2B1', 'F3E4', 'F1A3', 'F3A2', 'F1E3', 'F1B5', 'F3F5', 'F1F3', 'F1E4', 'F1F4', 'F2F5', 'F2E4', 'F1F2', 'F2A4', 'F3B5', 'F2G2']
files = []
for file in filesAdd:
	df = pd.read_csv("Binary Data/" + file+".csv")
	df.isnull().any()
	df = df.fillna(method='ffill')
	files.append(df)
X = []
y = []

for f in files:	
	n = 0
	#list of lists
	tripleL = []
	#0 or 1 distracted
	binary = []
	fiveData = []
	#columns =  ['delta_relative_2', 'beta_relative_3', 'beta_relative_4']
	#columns =  ['delta_relative_3', 'gamma_relative_1', 'gamma_relative_3']
	columns =  ['delta_relative_1','delta_relative_2', 'delta_relative_3', 'delta_relative_4', 'alpha_relative_1','alpha_relative_2', 'alpha_relative_3', 'alpha_relative_4', 'beta_relative_1','beta_relative_2', 'beta_relative_3', 'beta_relative_4', 'theta_relative_1','theta_relative_2', 'theta_relative_3', 'theta_relative_4', 'gamma_relative_1','gamma_relative_2', 'gamma_relative_3', 'gamma_relative_4']	
	for col in columns:
		fiveData.append([])
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
			fiveData=[]
			for col in columns:
				fiveData.append([])
			y.append(stat.mode(fiveY)) 
			fiveY = []
		# temp = list(tripleL.appen)		
		fiveY.append(int(row['distLevel']))

		counter2 = 0 

		for col in columns:
			fiveData[counter2].append(row[col])
			counter2 = counter2 + 1 
		n = n + 1
accuracy = []

# make x and y
#add'
#for count in range(0, 100):
for x in range(0,6):
	# tempy = np.asarray(y_train)
	print(str(x) + ": " + str(y.count(x)))

# Normalize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

#undersample
sampling_strategy = {0:0, 2:303, 1:303, 3:303, 4:303, 5:303}
rus = RandomUnderSampler(sampling_strategy = sampling_strategy)
X, y = rus.fit_resample(X, y)
for x in range(0,6):
	# tempy = np.asarray(y_train)
	print(str(x) + ": " + str(y.count(x)))

# split data 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


#create and fit the mod
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
print(regressor.coef_)
coeff_df = pd.DataFrame(regressor.coef_, columns, columns=['Coefficient'])  
print(coeff_df)
#predict
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
allErrors = y_pred - y_test

df1 = df.head(25)
print(df1)
df1.plot(kind='bar',figsize=(10,8))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print("median: ", stat.median(allErrors))
allErrors  = allErrors * 100
allErrors = [round(x) for x in allErrors]
print("mode: ", stat.mode(allErrors))