#!/bin/bash

module load cuda
module load gcc
nvcc main.cu -O3 -std=c++11 -gencode arch=compute_70,code=sm_70
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_0.csv ./a.out 0
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_1.csv ./a.out 1
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_2.csv ./a.out 2
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_3.csv ./a.out 3
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_4.csv ./a.out 4
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_5.csv ./a.out 5
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_6.csv ./a.out 6
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_7.csv ./a.out 7
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_8.csv ./a.out 8

