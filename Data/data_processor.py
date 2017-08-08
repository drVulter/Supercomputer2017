# Script to process data from computer clusters
# Quinn Stratton

import matplotlib.pyplot as plt
import numpy as np

# lists of files to process
pi_file_names = ["/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataSingle.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataFour.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataEight.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/MPIPIRawDataSixteen.txt"]
gpu_file_names = ["/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/CUDAData/MPICUDAPIResultsOne.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/CUDAData/MPICUDAPIResultsTwo.txt",
	"/Users/quinngardnerstratton/Desktop/Supercomputer2017/Data/CUDAData/MPICUDAPIResultsFour.txt"]
run_times = [] # list of runtimes
which_list = input("GPU or PI cluster?: ")
if (which_list == "GPU"):
	file_names = gpu_file_names
else:
	file_names = pi_file_names

# Crunch the numbers
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
	# Crunch
	pi_trials = 40.0
	gpu_trials = 10.0
	if (which_list == "GPU"):
		num_trials = gpu_trials
	else:
		num_trials = pi_trials
	pi_avg = pi_total / num_trials # obtain average pi value
	time_avg = runtime_total / num_trials # obtain average runtime
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

# Assign process number list for x axis
if (which_list == "GPU"):
	x = [1, 2, 4]
else:
	x = [1, 4, 8, 16]


# Now plot the data
y = reversed(x)
z = reversed(run_times)


if (which_list == "GPU"):
	plt.figure("gpu_runtimes")
	plt.plot(x, run_times, marker='o')
	plt.xlabel("Number of nodes")
	plt.ylabel("Average Runtime (Seconds)")
	plt.title("Runtimes for pi_approx Running on Jetson Cluster")
	plt.grid(True)
	plt.show()

	plt.figure("gpu_speedup")
	plt.plot(x, x, marker='o') # ideal
	plt.plot(x, speedup_list, marker='o')
	plt.xlabel("Number of nodes")
	plt.ylabel("Speedup")
	plt.title("Ideal v.s. Actual Speedup of pi_approx on Jetson Cluster")
	plt.grid(True)
	plt.show()
else:
	plt.figure("raspi_runtimes")
	plt.plot(x, run_times, marker='o')
	plt.xlabel("Number of nodes")
	plt.ylabel("Average Runtime (Seconds)")
	plt.title("Runtimes for pi_approx running on Raspberry PI cluster")
	plt.grid(True)
	plt.show()

	plt.figure("raspi_speedup")
	plt.plot(x, x, marker='o') # ideal
	plt.plot(x, speedup_list, marker='o')
	plt.xlabel("Number of nodes")
	plt.ylabel("Speedup")
	plt.title("Ideal v.s. Actual Speedup of pi_approx on Jetson Cluster")
	plt.grid(True)
	plt.show()


# LINPACK data
if (which_list != "GPU"):
	new_x = [1, 4, 16]
	new_times = [317.09, 97.94, 38.42]
	new_su_list = []
	for time in new_times:
		new_su = new_times[0] / time
		new_su_list.append(new_su)
	plt.figure("HPLSpeedup")
	plt.plot(new_x, new_x, marker='o')
	plt.plot(new_x, new_su_list, marker='o')
	plt.xlabel("Number of nodes")
	plt.ylabel("Speedup")
	plt.title("Ideal v.s. Actual Speedup of HPL on Pi Cluster")
	plt.grid(True)
	plt.show()
