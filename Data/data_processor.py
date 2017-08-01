# Script to process data from computer clusters
# Quinn Stratton

file_name = input("Which file should I process?: ") # careful typing...

runtime_total = 0.0 # total of all runtimes added together
pi_total = 0.0 # total of all values of pi added together
count = 0 # number of lines processed
with open(file_name, 'r') as f:
	for line in f:
		lines = f.read().splitlines()

for line in lines:
	values = line.split(" ")
	pi_total += float(values[0])
	runtime_total += float(values[1])
	count += 1



print(count)


pi_avg = pi_total / float(count) # obtain average pi value

time_avg = runtime_total / float(count) # obtain average runtime


print("pi average is ", pi_avg)

print("average runtime is ", time_avg)
