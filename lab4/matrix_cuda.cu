#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <string>

using Matrix = std::vector<std::vector<double>>;

__global__ void matrixMulKernel(const double* A, const double* B, double* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrixMulSharedKernel(const double* A, const double* B, double* C, int n) {
    __shared__ double As[16][16];
    __shared__ double Bs[16][16];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    double sum = 0.0;
    int tiles = (n + 15) / 16;

    for (int t = 0; t < tiles; ++t) {
        if (row < n && t * 16 + tx < n)
            As[ty][tx] = A[row * n + t * 16 + tx];
        else
            As[ty][tx] = 0.0;

        if (t * 16 + ty < n && col < n)
            Bs[ty][tx] = B[(t * 16 + ty) * n + col];
        else
            Bs[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < 16; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

Matrix read_matrix(const std::string& filename, int& n) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file >> n;
    Matrix matrix(n, std::vector<double>(n));

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            file >> matrix[i][j];
        }
    }

    return matrix;
}

void write_matrix(const std::string& filename, const Matrix& matrix) {
    std::ofstream file(filename);
    int n = matrix.size();
    file << n << "\n";

    for (const auto& row : matrix) {
        for (double value : row) {
            file << value << " ";
        }
        file << "\n";
    }
}

void checkCudaError(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << message << " - " 
                  << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void printDeviceInfo() {
    int deviceCount;
    checkCudaError(cudaGetDeviceCount(&deviceCount), "Get device count");
    
    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found!" << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaDeviceProp deviceProp;
    checkCudaError(cudaGetDeviceProperties(&deviceProp, 0), "Get device properties");
    
    std::cout << "GPU Device: " << deviceProp.name << std::endl;
    std::cout << "Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Total Global Memory: " << deviceProp.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "Multiprocessors: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "Max Threads Per Block: " << deviceProp.maxThreadsPerBlock << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " A.txt B.txt result.txt [block_size] [use_shared]\n";
        std::cerr << "  block_size: block dimension (default: 16)\n";
        std::cerr << "  use_shared: use shared memory optimization (0/1, default: 0)\n";
        return 1;
    }

    printDeviceInfo();

    int block_size = 16;
    int use_shared = 0;
    
    if (argc >= 5) {
        block_size = std::stoi(argv[4]);
    }
    if (argc >= 6) {
        use_shared = std::stoi(argv[5]);
    }

    int n = 0;
    Matrix A = read_matrix(argv[1], n);
    Matrix B = read_matrix(argv[2], n);
    Matrix C(n, std::vector<double>(n, 0.0));

    std::cout << "Matrix size: " << n << "x" << n << std::endl;
    std::cout << "Block size: " << block_size << "x" << block_size << std::endl;
    std::cout << "Using shared memory: " << (use_shared ? "yes" : "no") << std::endl;
    std::cout << "Operations: " << n * n * n << std::endl;

    std::vector<double> h_A(n * n);
    std::vector<double> h_B(n * n);
    std::vector<double> h_C(n * n, 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            h_A[i * n + j] = A[i][j];
            h_B[i * n + j] = B[i][j];
        }
    }

    double *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(double);
    
    checkCudaError(cudaMalloc(&d_A, size), "Malloc d_A");
    checkCudaError(cudaMalloc(&d_B, size), "Malloc d_B");
    checkCudaError(cudaMalloc(&d_C, size), "Malloc d_C");
    checkCudaError(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice), "Memcpy H2D A");
    checkCudaError(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice), "Memcpy H2D B");

    dim3 blockDim(block_size, block_size);
    dim3 gridDim((n + block_size - 1) / block_size, 
                 (n + block_size - 1) / block_size);

    std::cout << "Grid configuration: " << gridDim.x << "x" << gridDim.y 
              << " blocks" << std::endl;

    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "Create start event");
    checkCudaError(cudaEventCreate(&stop), "Create stop event");

    if (use_shared) {
        matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    } else {
        matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    }
    cudaDeviceSynchronize();

    checkCudaError(cudaEventRecord(start, 0), "Record start");
    
    if (use_shared) {
        matrixMulSharedKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    } else {
        matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, n);
    }
    
    checkCudaError(cudaEventRecord(stop, 0), "Record stop");
    checkCudaError(cudaEventSynchronize(stop), "Sync stop");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "Event elapsed time");

    checkCudaError(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost), "Memcpy D2H C");

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = h_C[i * n + j];
        }
    }

    write_matrix(argv[3], C);

    double seconds = milliseconds / 1000.0;
    double gflops = (2.0 * n * n * n) / seconds / 1e9;

    std::cout << "Execution time: " << seconds << " sec" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

    checkCudaError(cudaEventDestroy(start), "Destroy start event");
    checkCudaError(cudaEventDestroy(stop), "Destroy stop event");
    checkCudaError(cudaFree(d_A), "Free d_A");
    checkCudaError(cudaFree(d_B), "Free d_B");
    checkCudaError(cudaFree(d_C), "Free d_C");

    return 0;
}