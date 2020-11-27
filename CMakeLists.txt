cmake_minimum_required(VERSION 3.17)
project(parsha256_bench_gpu CUDA)

set(CMAKE_CUDA_STANDARD 14)

add_executable(parsha256_bench_gpu main.cu)

set_target_properties(
        parsha256_bench_gpu
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)