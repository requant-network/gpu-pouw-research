// Independent diagnostic primitive. The research protocol runs on CPU first.
#include <cuda_runtime.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda_fp8.h>

using Word = uint32_t;

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

__global__ void primitive(const Word* a, const Word* b, Word* out, Word n, Word first, Word second) {
    const Word i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const __nv_fp8_e4m3 value(__uint_as_float(a[i]));
    out[i] = value.__x;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("Usage: gpu-check results/gpu-fixture.bin");
        std::ifstream file(argv[1], std::ios::binary);
        Word n = 0, first = 0, second = 0;
        file.read(reinterpret_cast<char*>(&n), sizeof(n));
        file.read(reinterpret_cast<char*>(&first), sizeof(first));
        file.read(reinterpret_cast<char*>(&second), sizeof(second));
        if (!file || !n || n > (1u << 24)) throw std::runtime_error("Invalid fixture header");
        std::vector<Word> a(n), b(n), expected(n), actual(n);
        for (auto* values : {&a, &b, &expected}) {
            file.read(reinterpret_cast<char*>(values->data()), n * sizeof(Word));
        }
        if (!file) throw std::runtime_error("Truncated fixture");
        Word *device_a = nullptr, *device_b = nullptr, *device_out = nullptr;
        check(cudaMalloc(&device_a, n * sizeof(Word)));
        check(cudaMalloc(&device_b, n * sizeof(Word)));
        check(cudaMalloc(&device_out, n * sizeof(Word)));
        check(cudaMemcpy(device_a, a.data(), n * sizeof(Word), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_b, b.data(), n * sizeof(Word), cudaMemcpyHostToDevice));
        for (int repeat = 0; repeat < 10; ++repeat) {
            primitive<<<(n + 255) / 256, 256>>>(device_a, device_b, device_out, n, first, second);
        }
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        cudaEvent_t start, end;
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&end));
        check(cudaEventRecord(start));
        for (int repeat = 0; repeat < 100; ++repeat) {
            primitive<<<(n + 255) / 256, 256>>>(device_a, device_b, device_out, n, first, second);
        }
        check(cudaGetLastError());
        check(cudaEventRecord(end));
        check(cudaEventSynchronize(end));
        float milliseconds = 0;
        check(cudaEventElapsedTime(&milliseconds, start, end));
        check(cudaMemcpy(actual.data(), device_out, n * sizeof(Word), cudaMemcpyDeviceToHost));
        if (actual != expected) throw std::runtime_error("GPU result differs from CPU fixture");
        cudaDeviceProp properties;
        check(cudaGetDeviceProperties(&properties, 0));
        std::cout << "{\"verified\":true,\"elements\":" << n
                  << ",\"kernel_mean_ms\":" << milliseconds / 100
                  << ",\"device\":\"" << properties.name
                  << "\",\"scope\":\"diagnostic primitive only; kernel event timing\"}\n";
        check(cudaEventDestroy(start));
        check(cudaEventDestroy(end));
        check(cudaFree(device_a));
        check(cudaFree(device_b));
        check(cudaFree(device_out));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
