#include <iostream>
#include "parsha256_on_gpu.h"

int main(int argc, char *argv[]) {


    if (argc == 2) {
        parsha256_on_gpu_bench(atoi(argv[1]));
    } else {
        parsha256_on_gpu_test();

    }

    return 0;
}
