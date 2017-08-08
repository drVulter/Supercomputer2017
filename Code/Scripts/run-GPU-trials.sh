echo "Running trials on one process!"

for run in {1..10}
do
	mpiexec -np 1 --map-by node --hostfile ~/.mpi_hostfile ./pi
done

echo "Running trials on two processes!"
for run in {1..10}
do
	mpiexec -np 2 --map-by node --hostfile ~/.mpi_hostfile ./pi
done

echo "Running trials on Four processes!"
for run in {1..10}
do
	mpiexec -np 4 --map-by node --hostfile ~/.mpi_hostfile ./pi
done
