import random

route = [1, 2, 3]
distractions = ["B", "C", "D", "E", "F", "G"]
trials = []

route_dict = {"1" : "Drive to Waterloo",
			  "2" : "Garden Lane",
			  "3" : "Bourbon Street"}
distractions_dict = {"A" : "No distraction",
					 "B" : "Ask math questions",
					 # "C" : "Do I spy questions",
					 "E" : "Listen to music",
					 "F" : "Answer conversational questions",
					 "G" : "Send text messages according to dictation",
					 "H" : "Tongue Twister"}

# for each participant
participant = input("Please type the letter name of your participant.")

chosen_routes = random.sample(route, 3)
for chosen_route in chosen_routes:
	# 3 distractions
	# distractions = random.sample(distractions, 3)
	for j in range(1, 3):
		trials.append(participant + str(chosen_route) + "A" + str(j))
	for distraction in distractions:
		for i in range(1, 6):
			trials.append(participant + str(chosen_route) + distraction + str(i))
shuffled_trials = random.sample(trials, 18)
i = 1
for t in shuffled_trials :
	print(str(i) + ". " + t + ": Participant \"" + participant + "\" on route \"" + str(route_dict.get(t[1])) + "\" with \"" + str(distractions_dict.get(t[2])) + "\"")
	i = i + 1
print(shuffled_trials)