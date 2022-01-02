### pythonosc
from time import time
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

dictadd = {}
initAdd = "/coders/elements/"
columns =  ['delta_relative', 'alpha_relative', 'beta_relative', 'theta_relative', 'gamma_relative']	
for col in columns:
	dictadd[initAdd+col] = col
started = False
start = time()
def handler(address, *args):
	# TODO this is where the following things should happen
	# 1. check address to see if it is something you want to work with (like "/theCobraCoders/elements")
	# 2. extract values from args (like the value for alpha power)
	# 3. append value to a list of data for last 5s that is stored in some global variable
	# 4. look through the list of data and remove stuff from more than 5s ago
	# 5. input the list of data into your trained machine learning model (like a LogisticRegression that has been .fit'ed already)
	# 6. get output from model and print the output distraction value to the console (with a print() statement)
	# if you think of a better way than the above, go for it! :-)
	global start
	global started

	if (address in dictadd):
		if (started == False):
			start= time()
			started = True
		print(f"DEFAULT {address}: {args}")
		currTime = time()
		if (currTime - start >= 5):
			start = time()


dispatcher = Dispatcher()
dispatcher.set_default_handler(handler)

# this should be set to whatever your computer's IP address is (https://support.microsoft.com/en-us/help/15291/windows-find-pc-ip-address)
# your computer's IP address should also be in the corresponding field in the Muse Direct app
# it needs to be the IP address of YOUR computer because that's the only way Muse Direct will know which computer to stream the data to
ip = "10.0.0.27"
# this should be whatever the field is set to in the Muse Direct app
# as of 2/4/2020 it is 7000
port = 8000

server = BlockingOSCUDPServer((ip, port), dispatcher)
server.serve_forever()  # Blocks forever