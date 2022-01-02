import pandas as pd 
fileNames = ['A3B2', 'A3D2', 'A2G2', 'B2G1', 'B2A2', 'B1C1', 'B1A1', 'B1G2', 'B2E2', 'B2C1', 'B1A2', 'B1E1', 'B1G3', 'B2C2', 'B2A1', 'B2C1', 'B2G2', 'B1A2', 'B2A2', 'C2G1', 'C1A2', 'C2G2', 'C2E1', 'C1F2','C1E2', 'C2A1', 'E2A2', 'E1A2', 'E3G4', 'E2G4', 'E3A2', 'E1G1', 'E3A3', 'E1A1', 'E2B2', 'E2G3', 'E3A4', 'E3G3', 'E1A4', 'E3F4', 'E2B5', 'E2A1', 'E1F5', 'F3A3', 'F2B2', 'F2B1', 'F3E4', 'F1A3', 'F3A2', 'F1E3', 'F1B5', 'F3F5', 'F1F3', 'F1E4', 'F1F4', 'F2F5', 'F2E4', 'F1F2', 'F2A4', 'F3B5', 'F2G2']
distractions_dict = {"A" : "No distraction",
					 "B" : "Ask math questions",
					 "C" : "Do I spy questions",
					 "D" : "Drink Water and Eat Snacks",
					 "E" : "Listen to music",
					 "F" : "Answer conversational questions",
					 "G" : "Send text messages according to dictation",
					 "H" : "Tongue Twister"}
for file in fileNames:
	df = pd.read_csv('Binary Data/' + file + ".csv")
	df.isnull().any()
	df = df.fillna(method='ffill')
	distraction = distractions_dict[file[2]]
	print (distraction)
	df["distraction"] = distraction
	df.to_csv('Binary Data/' + file + ".csv")

