#include <cmath>
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};


static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define CHECK(x) check(x, #x)

__global__ void mykernel(float* nums_sum, int* location, float* optimal, int nx, int ny, float total_sum) {

    int height = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int width = threadIdx.x + blockIdx.x * blockDim.x + 1;

    int x0t = 0;
    int y0t = 0;

    if (height > ny || width > nx) {return;}

    float sz_in = (float)width * height;
    float sz_out = (float)nx * ny - width * height;
    float scale_in = 1.0 / sz_in;
    float scale_out = 1.0 / sz_out;



    float max_value = -1729;

    for (int y0 = 0; y0 < ny - height + 1; y0++) {
        for (int x0 = 0; x0 < nx - width + 1; x0++) {

            // inclusion-exclusion principle for sum of pixels inside rectangle
            float inner_sum = nums_sum[x0 + width + (nx + 1) * (y0 + height)] - nums_sum[x0 + width + (nx + 1) * y0] - nums_sum[x0 + (nx + 1) * (y0 + height)] + nums_sum[x0 + (nx + 1) * y0];
            float outer_sum = total_sum - inner_sum;

           // simplifying the expression and observing that the sum of the squares doesn't have to be calculated explicitly, as it will be an invariant in the total error,
           // we can simply maximize these term that reduces the value of the total error

            float inner_value = scale_in * inner_sum * inner_sum;
            float outer_value = scale_out * outer_sum * outer_sum;
            float total = inner_value + outer_value;
            float value = total;


            if (value > max_value) {
                max_value = value; // find best result for given size
                x0t = x0;
                y0t = y0;
              }
        }
    }

    optimal[nx * (height - 1) + (width - 1)] = max_value;
    location[4 * (nx * (height - 1) + (width - 1))] = x0t;
    location[4 * (nx * (height - 1) + (width - 1)) + 1] = y0t;
    location[4 * (nx * (height - 1) + (width - 1)) + 2] = width;
    location[4 * (nx * (height - 1) + (width - 1)) + 3] = height;
}



/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {

    std::vector<float> nums_sum((ny + 1) * (nx + 1));

    for (int j = 0; j < ny + 1; j++) {
            nums_sum[(nx + 1) * j] = 0.0;
    }

    for (int i = 0; i < nx + 1; i++) {
            nums_sum[i] = 0.0;
    }

    for (int j = 1; j < ny + 1; j++) {
        for (int i = 1; i < nx + 1; i++) {
            nums_sum[i + (nx + 1) * j] = data[3 * (i - 1) + 3 * nx * (j - 1)] + nums_sum[i + (nx + 1) * (j - 1)] + nums_sum[i - 1 + (nx + 1) * j] - nums_sum[i - 1 + (nx + 1) * (j - 1)];
        }
    }
    float total_sum = nums_sum[(nx + 1) * (ny + 1) - 1];

    std::vector<float> optimal_value(nx * ny, 0);
    std::vector<int> location(4 * nx * ny, 0);


    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, (nx + 1) * (ny + 1) * sizeof(float)));

    int* dGPU_location = NULL;
    CHECK(cudaMalloc((void**)&dGPU_location, 4 * nx * ny * sizeof(int)));

    float* dGPU_value = NULL;
    CHECK(cudaMalloc((void**)&dGPU_value, nx * ny * sizeof(float)));

    CHECK(cudaMemcpy(dGPU, nums_sum.data(), (nx + 1) * (ny + 1) * sizeof(float), cudaMemcpyHostToDevice));

    {
        dim3 dimBlock(16, 16);
        dim3 dimGrid(divup(nx + 1, dimBlock.x), divup(ny + 1, dimBlock.y)); // thread finds best position for rectangle
        mykernel<<<dimGrid, dimBlock>>>(dGPU, dGPU_location, dGPU_value, nx, ny, total_sum);
        CHECK(cudaGetLastError());
    }


    CHECK(cudaMemcpy(optimal_value.data(), dGPU_value, nx * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(location.data(), dGPU_location, 4 * nx * ny * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU_value));
    CHECK(cudaFree(dGPU_location));
    CHECK(cudaFree(dGPU));


    float optimal_final = -1.0;
    int k = 0;
    // find rectangle with best result
    for (int i = 0; i < nx * ny; i++) {
      if (optimal_value[i] > optimal_final) {
        optimal_final = optimal_value[i];
        k = i;
      }
    }

    int best_width = location[4 * k + 2];
    int best_height = location[4 * k + 3];
    // best values

    int x0b = location[4 * k];
    int x1b = x0b + best_width;
    int y0b = location[4 * k + 1];
    int y1b = y0b + best_height;

    int bsz = best_width * best_height;
    int osz = nx * ny - bsz;

    float inner_sum = nums_sum[x0b + best_width + (nx + 1) * (y0b + best_height)] - nums_sum[x0b + best_width + (nx + 1) * y0b] - nums_sum[x0b + (nx + 1) * (y0b + best_height)] + nums_sum[x0b + (nx + 1) * y0b];
    float outer_sum = total_sum - inner_sum;

    float inner_best = inner_sum / bsz;
    float outer_best = outer_sum / osz;

    Result result{y0b, x0b, y1b, x1b, {float(outer_best), float(outer_best), float(outer_best)}, {float(inner_best), float(inner_best), float(inner_best)}};

    return result;

}
