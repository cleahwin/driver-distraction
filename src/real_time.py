
# immport files
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
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import accuracy_score

# read the files
f1 = pd.read_csv("A2G2.csv")
f2 = pd.read_csv("A3B2.csv")
f3 = pd.read_csv("A3D2.csv")
f4 = pd.read_csv("B1A1.csv")
f5 = pd.read_csv("B1A2.csv")
f6 = pd.read_csv("B1A3.csv")
f7 = pd.read_csv("B1C1.csv")
f8 = pd.read_csv("B1E1.csv")
f9 = pd.read_csv("B1G2.csv") 
f10 = pd.read_csv("B1G3.csv")
f11 = pd.read_csv("B1A2.csv")
f12 = pd.read_csv("B1A3.csv")
f13 = pd.read_csv("B2C1.csv")
f14 = pd.read_csv("B2C2.csv")
f15 = pd.read_csv("B2C3.csv")
f16 = pd.read_csv("B2E2.csv")
f17 = pd.read_csv("B2G1.csv")
f18 = pd.read_csv("B2G2.csv")
f19 = pd.read_csv("C1A2.csv")
f20 = pd.read_csv("C1E2.csv")
f21 = pd.read_csv("C1F2.csv")
f22 = pd.read_csv("C2A1.csv")
f23 = pd.read_csv("C2E1.csv")
f24 = pd.read_csv("C2G1.csv")
f25 = pd.read_csv("C2G2.csv")
#, f19, f20, f21, f22, f23, f24, f25
# filesAdd = ['E2A2', 'E1A2', 'E3G4', 'E2G4', 'E3A2', 'E1G1', 'E3A3', 'E1A1', 'E2B2', 'E2G3', 'E3A4', 'E3G3', 'E1A4', 'E3F4', 'E2B5', 'E2A1', 'E1F5', 'F3A3', 'F2B2', 'F2B1', 'F3E4', 'F1A3', 'F3A2', 'F1E3', 'F1B5', 'F3F5', 'F1F3', 'F1E4', 'F1F4', 'F2F5', 'F2E4', 'F1F2', 'F2A4', 'F3B5', 'F2G2']
files = [f1, f2, f3, f4, f5, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18]
# for file in filesAdd:
# 	df = pd.read_csv(file+".csv")
# 	df.isnull().any()
# 	df = df.fillna(method='ffill')
# 	files.append(df)
X = []
y = []
columns =  ['delta_relative_1','delta_relative_2', 'delta_relative_3', 'delta_relative_4', 'alpha_relative_1','alpha_relative_2', 'alpha_relative_3', 'alpha_relative_4', 'beta_relative_1','beta_relative_2', 'beta_relative_3', 'beta_relative_4', 'theta_relative_1','theta_relative_2', 'theta_relative_3', 'theta_relative_4', 'gamma_relative_1','gamma_relative_2', 'gamma_relative_3', 'gamma_relative_4']	

# make x and y
for f in files:	
	n = 0
	#list of lists
	tripleL = []
	#0 or 1 distracted
	binary = []
	fiveData = []
	#columns =  ['delta_relative_2', 'beta_relative_3']
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
		fiveY.append(int(row['distracted?']))
		
		counter2 = 0 

		for col in columns:
			fiveData[counter2].append(row[col])
			counter2 = counter2 + 1 
		n = n + 1

accuracy = []
#add'
# for count in range(0, 100):
# Make both classes same size
sampling_strategy = 1
rus = RandomUnderSampler(sampling_strategy = sampling_strategy)
X, y = rus.fit_resample(X, y)
# training our model 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)



# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

# Test the model
y_pred=logreg.predict(X_test)





##### REAL TIME STREAMING, PROCESSING, EVALUATING

### pythonosc
from time import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

dictadd = {}
initAdd = "/coders/elements/"
columns2 =  ['delta_relative', 'alpha_relative', 'beta_relative', 'theta_relative', 'gamma_relative']	
for col in columns2:
	dictadd[initAdd+col] = col
started = False
start = time()
xLIVE = []
currList = []
canWe = False
def handler(address, *args):
	# TODO this is where the following things should happen
	# 1. check address to see if it is something you want to work with (like "/theCobraCoders/elements")
	# 2. extract values from args (like the value for alpha power)
	# 3. append value to a list of data for last 5s that is stored in some global variable
	# 4. look through the list of data and remove stuff from more than 5s ago
	# 5. input the list of data into your trained machine learning model (like a LogisticRegression that has been .fit'ed already)
	# 6. get output from model and print the output distraction value to the console (with a print() statement)
	# if you think of a better way than the above, go for it! :-)
    
    if (address in dictadd):
    	print(f"DEFAULT {address}: {args}")
    	if (started == False):
    		start= time()
    		started = True
    	currTime = time()
    	if (currTime - start >= 5 and canWe):
    		start = time()
    		realX = [[]]
    		for countie in range(0, 20):
    			tba=[]
    			for listie in xLIVE:
    				tba.append(listie[countie])
    			realX[0].append(stat.mean(tba))
    		print(logreg.predict(realX))
    		## TO DO EVAL AND RESET
    	else: # add to xLIVE
    		for val in args:
    			currList.append(val)
    		if (dictadd[address] == 'gamma_relative'):
    			xLIVE.append(currList)
    			currList = []
    			canWe=True
    		else:
    			canWe=False


dispatcher = Dispatcher()
dispatcher.set_default_handler(handler)

# this should be set to whatever your computer's IP address is (https://support.microsoft.com/en-us/help/15291/windows-find-pc-ip-address)
# your computer's IP address should also be in the corresponding field in the Muse Direct app
# it needs to be the IP address of YOUR computer because that's the only way Muse Direct will know which computer to stream the data to
ip = "10.0.0.27"
# this should be whatever the field is set to in the Muse Direct app
# as of 2/4/2020 it is 7000
port = 6000

server = BlockingOSCUDPServer((ip, port), dispatcher)
server.serve_forever()  # Blocks forever