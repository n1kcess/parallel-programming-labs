#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

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

Matrix multiply(const Matrix& A, const Matrix& B) {
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
    if (argc != 4) {
        std::cerr << "Usage: lab1 A.txt B.txt result.txt\n";
        return 1;
    }

    std::size_t nA, nB;
    Matrix A = read_matrix(argv[1], nA);
    Matrix B = read_matrix(argv[2], nB);

    if (nA != nB) {
        throw std::runtime_error("Matrix sizes do not match");
    }

    auto start = std::chrono::high_resolution_clock::now();
    Matrix C = multiply(A, B);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    write_matrix(argv[3], C);

    std::cout << "Matrix size: " << nA << "x" << nA << "\n";
    std::cout << "Operations: " << nA * nA * nA << "\n";
    std::cout << "Execution time: " << duration.count() << " sec\n";

    return 0;
}