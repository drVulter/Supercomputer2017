#!/bin/bash
mpicc -c pi_main.c -o pi_main.o
nvcc -c pi_kernel.cu -o pi_kernel.o
mpicc pi_main.o pi_kernel.o -lcudart -L/usr/local/cuda/lib/ -o pi
echo "Successfully compiled"
scp ./pi ubuntu@node1:~/MPICode/PI
scp ./pi ubuntu@node2:~/MPICode/PI
scp ./pi ubuntu@node3:~/MPICode/PI
echo "Successfully copied!"
