#include <mpi.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <algorithm>
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

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " A.txt B.txt result.txt\n";
        return 1;
    }

    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::size_t n = 0;
    Matrix A, B, C;

    if (rank == 0) {
        try {
            A = read_matrix(argv[1], n);
            B = read_matrix(argv[2], n);
            C.resize(n, std::vector<double>(n, 0.0));
        } catch (const std::exception& e) {
            std::cerr << "Error reading matrices: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&n, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (n == 0) {
        MPI_Finalize();
        return 0;
    }

    std::size_t rows_per_proc = (n + size - 1) / size;
    std::size_t row_start = rank * rows_per_proc;
    std::size_t row_end = std::min(row_start + rows_per_proc, n);
    std::size_t local_rows = row_end - row_start;

    std::vector<int> sendcounts(size);
    std::vector<int> displs(size);
    for (int i = 0; i < size; ++i) {
        std::size_t start = i * rows_per_proc;
        if (start >= n) {
            sendcounts[i] = 0;
            displs[i] = 0;
        } else {
            std::size_t end = std::min(start + rows_per_proc, n);
            sendcounts[i] = static_cast<int>((end - start) * n);
            displs[i] = static_cast<int>(start * n);
        }
    }

    std::vector<double> send_A; 
    std::vector<double> recv_A(sendcounts[rank]);
    std::vector<double> local_B(n * n);
    std::vector<double> local_C(sendcounts[rank], 0.0);

    if (rank == 0) {
        send_A.resize(n * n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                send_A[i * n + j] = A[i][j];
                local_B[i * n + j] = B[i][j];
            }
        }
        std::cout << "Matrix size: " << n << "x" << n << "\n";
        std::cout << "Processes: " << size << "\n";
        std::cout << "Operations: " << n * n * n << "\n";
    }

    MPI_Bcast(local_B.data(), static_cast<int>(n * n), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(send_A.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                 recv_A.data(), sendcounts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    for (std::size_t i = 0; i < local_rows; ++i) {
        for (std::size_t k = 0; k < n; ++k) {
            double a_ik = recv_A[i * n + k];
            for (std::size_t j = 0; j < n; ++j) {
                local_C[i * n + j] += a_ik * local_B[k * n + j];
            }
        }
    }

    double end_time = MPI_Wtime();

    std::vector<double> global_C(n * n);
    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_DOUBLE,
                global_C.data(), sendcounts.data(), displs.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i][j] = global_C[i * n + j];
            }
        }
        write_matrix(argv[3], C);
        double exec_time = end_time - start_time;
        std::cout << "Execution time: " << exec_time << " sec\n";
        std::cout << "Performance: " << (n * n * n * 2.0 / exec_time / 1e9) << " GFLOPS\n";
    }

    MPI_Finalize();
    return 0;
}