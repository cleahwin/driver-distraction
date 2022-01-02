import pandas as pd
import time
import sys

# will be as UTC so add 2880
# df = pd.read_csv("startEnd.csv")
df = pd.DataFrame()
go = True 
# KEY
# e = stop
# sd = start distraction
# ed = end distraction
# c = crash
crashTups = []
tempPair = []
swerves = 0
crashes = 0
swerveTups = []
while (go):
	event = input("next! ")
	if (event == 'e'):
	 	go = False
	elif (event == 's'):
		swerveTups.append([time.time() - 1.0, time.time() + 1.0])
		swerves+=1
	elif (event == 'c'):
		crashTups.append([time.time() - 1.0, time.time() + 1.0])
		crashes+=1
	
	df2 = pd.DataFrame({"event": [event], 
						"at" : [time.time()],
						"start time": [time.time()-0.5], 
						"end time": [time.time() + 0.5]})
	df = df.append(df2)

print("crash times: ", crashTups)
print(crashes, " crashes, and ", swerves, " swerves")
print("swerve times: ",swerveTups)
#df.to_csv("event_times_data/events_" + str(sys.argv[1]) + ".csv")
