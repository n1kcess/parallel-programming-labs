#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>

using Matrix = std::vector<std::vector<double>>;

Matrix read_matrix(const std::string& filename, std::size_t& n) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    file >> n;

    Matrix matrix(n, std::vector<double>(n));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            file >> matrix[i][j];
        }
    }

    return matrix;
}

void write_matrix(const std::string& filename, const Matrix& matrix) {
    std::ofstream file(filename);
    std::size_t n = matrix.size();
    file << n << "\n";

    for (const auto& row : matrix) {
        for (double value : row) {
            file << value << " ";
        }
        file << "\n";
    }
}

Matrix multiply_parallel(const Matrix& A, const Matrix& B, int num_threads) {
    std::size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));

    omp_set_num_threads(num_threads);
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            double a_ik = A[i][k];
            for (std::size_t j = 0; j < n; ++j) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }

    return C;
}

Matrix multiply_sequential(const Matrix& A, const Matrix& B) {
    std::size_t n = A.size();
    Matrix C(n, std::vector<double>(n, 0.0));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return C;
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " A.txt B.txt result.txt [num_threads] [sequential]\n";
        std::cerr << "  num_threads: number of OpenMP threads (default: 4)\n";
        std::cerr << "  sequential: use 'seq' for sequential version\n";
        return 1;
    }

    int num_threads = 4;
    bool sequential = false;
    
    if (argc >= 5) {
        if (std::string(argv[4]) == "seq") {
            sequential = true;
        } else {
            num_threads = std::stoi(argv[4]);
        }
    }
    
    if (argc >= 6 && std::string(argv[5]) == "seq") {
        sequential = true;
    }

    std::size_t nA, nB;
    Matrix A = read_matrix(argv[1], nA);
    Matrix B = read_matrix(argv[2], nB);

    if (nA != nB) {
        throw std::runtime_error("Matrix sizes do not match");
    }

    std::cout << "Matrix size: " << nA << "x" << nA << "\n";
    std::cout << "Operations: " << nA * nA * nA << "\n";
    
    if (sequential) {
        std::cout << "Running sequential version...\n";
    } else {
        std::cout << "Running parallel version with " << num_threads << " threads\n";
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    Matrix C;
    if (sequential) {
        C = multiply_sequential(A, B);
    } else {
        C = multiply_parallel(A, B, num_threads);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    
    write_matrix(argv[3], C);

    std::cout << "Execution time: " << duration.count() << " sec\n";
    std::cout << "Performance: " << (nA * nA * nA * 2.0 / duration.count() / 1e9) << " GFLOPS\n";

    return 0;
}