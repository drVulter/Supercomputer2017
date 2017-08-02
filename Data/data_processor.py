# Script to process data from computer clusters
# Quinn Stratton

import matplotlib.pyplot as plt
import numpy as np

# list of files to process
file_names = ["/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataSingle.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataFour.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataEight.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataSixteen.txt"]
run_times = [] # list of runtimes
#line_list = []
for file_name in file_names:
	runtime_total = 0.0 # total of all runtimes added together
	pi_total = 0.0 # total of all values of pi added together
	count = 0 # number of lines processed
	print(file_name)
	with open(file_name, 'r') as f:
		line_list = f.read().splitlines()

	for line in line_list:
		if (count < 40):
			values = line.split(" ")
			pi_total += float(values[0])
			runtime_total += float(values[1])
		count += 1

	pi_avg = pi_total / 40.0 # obtain average pi value
	time_avg = runtime_total / 40.0 # obtain average runtime
	run_times.append(time_avg)
	print ("pi average is ", pi_avg)
	print("average runtime is ", time_avg)
	result_file = open("/Users/quinngardnerstratton/Desktop/Supercomputer2017/data/MPIPIResults.txt", 'a')
	result_file.write("%s\n" %(file_name))
	result_file.write("%f %f\n\n" % (pi_avg, time_avg))
	result_file.close()

speedup_list = []
for time in run_times:
	speedup = run_times[0] / time
	speedup_list.append(speedup)

print("Contents of speedup_list:")
for i in speedup_list:
	print("%f" % i)

# Now plot the data
x = [1, 4, 8, 16] # number of processes
y = reversed(x)
z = reversed(run_times)


#y = np.power(x, -1)
plt.figure("runtimes")
plt.plot(x, run_times)
#plt.plot(y, z)
plt.xlabel("Number of nodes")
plt.ylabel("Average Runtime (Seconds)")
plt.title("Runtimes for pi_approx running on Raspberry PI cluster")
plt.grid(True)
plt.show()

plt.figure("speedup")
plt.plot(x, x) # ideal
plt.plot(x, speedup_list)
plt.xlabel("Number of nodes")
plt.ylabel("Speedup")
plt.title("Ideal v.s. Actual Speedup of pi_approx.c on Pi cluster")
plt.grid(True)
plt.show()

new_x = [1, 4, 16]
new_times = [317.09, 97.94, 38.42]
new_su_list = []
for time in new_times:
	new_su = new_times[0] / time
	new_su_list.append(new_su)


plt.figure("otherSpeedup")
plt.plot(new_x, new_x)
plt.plot(new_x, new_su_list)
plt.xlabel("Number of nodes")
plt.ylabel("Speedup")
plt.title("Ideal v.s. Actual Speedup of HPL on Pi Cluster")
plt.grid(True)
plt.show()
