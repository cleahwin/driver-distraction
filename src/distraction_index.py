import pandas as pd 

filesAdd = ['C2G1', 'C1A2', 'C2G2', 'C2E1', 'C1F2', 'C1E2','B2G2', 'B1A2', 'B2A2', 'B2A1','B1G3','B1E1','B2C1','B1A1','E2A2', 'E1A2', 'E3G4', 'E2G4', 'E3A2', 'E1G1', 'E3A3', 'E1A1', 'E2B2', 'E2G3', 'E3A4', 'E3G3', 'E1A4', 'E3F4', 'E2B5', 'E2A1', 'E1F5', 'F3A3', 'F2B2', 'F2B1', 'F3E4', 'F1A3', 'F3A2', 'F1E3', 'F1B5', 'F3F5', 'F1F3', 'F1E4', 'F1F4', 'F2F5', 'F2E4', 'F1F2', 'F2A4', 'F3B5', 'F2G2']
distractedIndecis = [4,0,2,3,2,1,4,1,1,0,3,1,4,0,1, 2, 4, 4, 4, 5, 2, 4, 4, 5, 3, 4, 4, 3, 3, 1, 3, 1, 4, 2, 2, 1, 1, 2, 3, 3, 2, 2, 3, 3, 2, 3, 1, 4, 5]
count = 0
for file in filesAdd:
	df = pd.read_csv("Binary Data/" + file + ".csv")
	df.isnull().any()
	df['distLevel'] = distractedIndecis[count]
	df = df.fillna(method = 'ffill')
	count = count+1
	df.to_csv("Binary Data/" + file+".csv")