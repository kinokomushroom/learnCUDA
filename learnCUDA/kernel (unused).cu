#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <iostream>
#include <cmath>


__global__ void vectorAdd(int* a, int* b, int* c, int n)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < n)
	{
		c[index] = a[index] + b[index];
	}
}


void host_device_memory(int n)
{
	int* host_a;
	int* host_b;
	int* host_c;

	int* device_a;
	int* device_b;
	int* device_c;

	size_t bytes = n * sizeof(int);

	host_a = (int*)malloc(bytes);
	host_b = (int*)malloc(bytes);
	host_c = (int*)malloc(bytes);

	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);

	for (int i = 0; i < n; i++)
	{
		host_a[i] = rand() % 100;
		host_b[i] = rand() % 100;
	}

	cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

	int BLOCK_SIZE = 256;
	int GRID_SIZE = (int)ceil((float)n / BLOCK_SIZE);

	vectorAdd <<<BLOCK_SIZE, GRID_SIZE>>> (device_a, device_b, device_c, n);

	cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	for (int i = 0; i < n; i++)
	{
		std::cout << host_a[i] << " + " << host_b[i] << " = " << host_c[i] << std::endl;
	}

	free(host_a);
	free(host_b);
	free(host_c);
}


void unified_memory(int n)
{
	int id = cudaGetDevice(&id);

	size_t bytes = n * sizeof(int);

	int* a;
	int* b;
	int* c;

	cudaMallocManaged(&a, bytes);
	cudaMallocManaged(&b, bytes);
	cudaMallocManaged(&c, bytes);

	for (int i = 0; i < n; i++)
	{
		a[i] = rand() % 100;
		b[i] = rand() % 100;
	}

	int BLOCK_SIZE = 256;
	int GRID_SIZE = (int)ceil((float)n / BLOCK_SIZE);

	cudaMemPrefetchAsync(a, bytes, id);
	cudaMemPrefetchAsync(b, bytes, id);
	vectorAdd <<<BLOCK_SIZE, GRID_SIZE>>> (a, b, c, n);

	cudaDeviceSynchronize();

	cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

	for (int i = 0; i < n; i++)
	{
		std::cout << a[i] << " + " << b[i] << " = " << c[i] << std::endl;
	}

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
}


__global__ void matrixMul(int* a, int* b, int* c, int n)
{
	int row = (blockIdx.x * blockDim.x) + threadIdx.x;
	int column = (blockIdx.y * blockDim.y) + threadIdx.y;

	if ((row < n) && (column < n))
	{
		int sum = 0;
		for (int index = 0; index < n; index++)
		{
			sum += a[row * n + index] * b[index * n + column];
		}
		c[row * n + column] = sum;
	}
}


void multiply_matrix(int n)
{
	size_t bytes = n * n * sizeof(int);

	int* host_a;
	int* host_b;
	int* host_c;

	host_a = (int*)malloc(bytes);
	host_b = (int*)malloc(bytes);
	host_c = (int*)malloc(bytes);

	int* device_a;
	int* device_b;
	int* device_c;

	cudaMalloc(&device_a, bytes);
	cudaMalloc(&device_b, bytes);
	cudaMalloc(&device_c, bytes);

	for (int row = 0; row < n; row++)
	{
		for (int column = 0; column < n; column++)
		{
			int index = row * n + column;
			host_a[index] = rand() % 100;
			host_b[index] = rand() % 100;
		}
	}

	cudaMemcpy(device_a, host_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(device_b, host_b, bytes, cudaMemcpyHostToDevice);

	int BLOCK_SIZE = 16;
	int GRID_SIZE = (int)ceil((float)n / BLOCK_SIZE);

	dim3 grid(GRID_SIZE, GRID_SIZE);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

	std::cout << "STARTED COMPUTING" << std::endl;

	matrixMul <<<grid, threads>>> (device_a, device_b, device_c, n);

	cudaMemcpy(host_c, device_c, bytes, cudaMemcpyDeviceToHost);

	std::cout << "STARTED ERROR-CHECKING" << std::endl;

	bool successful = true;
	for (int row = 0; row < n; row++)
	{
		for (int column = 0; column < n; column++)
		{
			int sum = 0;
			for (int index = 0; index < n; index++)
			{
				sum += host_a[row * n + index] * host_b[index * n + column];
			}
			if (sum != host_c[row * n + column])
			{
				successful = false;
				break;
			}
		}
	}

	if (successful)
	{
		std::cout << "SUCCESSFUL" << std::endl;
	}
	else
	{
		std::cout << "UNSUCCESSFUL" << std::endl;
	}

	cudaFree(device_a);
	cudaFree(device_b);
	cudaFree(device_c);

	free(host_a);
	free(host_b);
	free(host_c);
}


//int main()
//{
//	//host_device_memory(20);
//	//unified_memory(20);
//	multiply_matrix(1024);
//
//	return 0;
//}