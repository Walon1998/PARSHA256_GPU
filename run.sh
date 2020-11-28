#!/bin/bash

module load cuda
module load gcc
nvcc main.cu -O3 -std=c++11 -gencode arch=compute_70,code=sm_70
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_0.txt ./a.out 0
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_1.txt ./a.out 1
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_2.txt ./a.out 2
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_3.txt ./a.out 3
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_4.txt ./a.out 4
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_5.txt ./a.out 5
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_6.txt ./a.out 6
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_7.txt ./a.out 7
nvprof --concurrent-kernels off --csv --profile-from-start off --print-gpu-trace --log-file Result/GPU_8.txt ./a.out 8

