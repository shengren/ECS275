#include <optix.h>  // must include these two headers to compile
#include <optixu/optixu_math_namespace.h>

#include "device_functions.h"

#include <cstdio>

void TestDeviceFunction(void) {
  printf("before calling the test kernel.\n");
  
  int num_elements = 256;
  int num_bytes = num_elements * sizeof(int);

  int *host_array = (int*)malloc(num_bytes);
  int *device_array = 0;
  cudaMalloc((void**)&device_array, num_bytes);

  int block_size = 128;
  int num_blocks = num_elements / block_size;

  device_test<<<num_blocks, block_size>>>(device_array);

  cudaMemcpy(host_array, device_array, num_bytes, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num_elements; ++i)
    printf("%d ", host_array[i]);
  printf("\n");

  free(host_array);
  cudaFree(device_array);

  printf("after calling the test kernel.\n");
}

__global__ void device_test(int *array) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  array[index] = 255 - index;
}
