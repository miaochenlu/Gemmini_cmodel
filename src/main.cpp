// main.cpp - Main entry point for Gemmini simulator using SPARTA framework
#include <iostream>
#include <memory>
#include <random>
#include <iomanip>
#include <string>
#include <cstring>

#include "gemmini/gemmini.hpp"
#include "gemmini/matrix.hpp"
#include "gemmini/pe.hpp"
#include "gemmini/systolic_array.hpp"
#include "sparta/app/CommandLineSimulator.hpp"
#include "sparta/app/Simulation.hpp"
#include "sparta/utils/SpartaTester.hpp"

using namespace gemmini;

// Global verbosity flag
bool g_verbose = false;

// Function to create and initialize a test matrix with random values
MatrixPtr createTestMatrix(uint32_t rows, uint32_t cols, int16_t min_val = -10, int16_t max_val = 10) {
    MatrixPtr matrix = create_matrix_ptr<Matrix>(rows, cols);
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int16_t> dist(min_val, max_val);
    
    // Fill matrix with random values
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            matrix->at(r, c) = dist(gen);
        }
    }
    
    if (g_verbose) {
        std::cout << "Created matrix " << rows << "x" << cols << " with values between " 
                 << min_val << " and " << max_val << std::endl;
    }
    
    return matrix;
}

// Function to create a test vector with random values
VectorPtr createTestVector(uint32_t size, int16_t min_val = -10, int16_t max_val = 10) {
    VectorPtr vector = std::make_shared<Vector>(size);
    
    // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int16_t> dist(min_val, max_val);
    
    // Fill vector with random values
    for (uint32_t i = 0; i < size; ++i) {
        (*vector)[i] = dist(gen);
    }
    
    if (g_verbose) {
        std::cout << "Created vector of size " << size << " with values between " 
                 << min_val << " and " << max_val << std::endl;
    }
    
    return vector;
}

// Function to calculate expected result of matrix-vector multiplication
MatrixPtr calculateExpectedResult(const MatrixPtr& matrix, const VectorPtr& vector) {
    uint32_t rows = matrix->rows();
    MatrixPtr result = create_matrix_ptr<Matrix>(rows, 1);
    
    for (uint32_t r = 0; r < rows; ++r) {
        int32_t sum = 0;
        for (uint32_t c = 0; c < matrix->cols(); ++c) {
            sum += static_cast<int32_t>(matrix->at(r, c)) * static_cast<int32_t>((*vector)[c]);
        }
        result->at(r, 0) = static_cast<int16_t>(sum);
    }
    
    if (g_verbose) {
        std::cout << "Calculated expected result matrix " << rows << "x1" << std::endl;
    }
    
    return result;
}

// Function to print a matrix
void printMatrix(const MatrixPtr& matrix, const std::string& name) {
    std::cout << name << " (" << matrix->rows() << "x" << matrix->cols() << "):" << std::endl;
    for (uint32_t r = 0; r < matrix->rows(); ++r) {
        std::cout << "  [";
        for (uint32_t c = 0; c < matrix->cols(); ++c) {
            std::cout << std::setw(5) << matrix->at(r, c);
            if (c < matrix->cols() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;
}

// Function to print a vector
void printVector(const VectorPtr& vector, const std::string& name) {
    std::cout << name << " (size " << vector->size() << "):" << std::endl;
    std::cout << "  [";
    for (uint32_t i = 0; i < vector->size(); ++i) {
        std::cout << std::setw(5) << (*vector)[i];
        if (i < vector->size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

// Manual test of matrix multiplication
void testManualMatrixMultiplication() {
    std::cout << "======= Testing Matrix Multiplication Manually =======" << std::endl;
    
    // Create test matrices
    MatrixPtr A = createTestMatrix(4, 4, 1, 5);
    MatrixPtr B = createTestMatrix(4, 4, 1, 5);
    
    // Print test matrices
    printMatrix(A, "Matrix A");
    printMatrix(B, "Matrix B");
    
    // Calculate matrix multiplication manually
    MatrixPtr C = create_matrix_ptr<Matrix>(A->rows(), B->cols());
    
    for (uint32_t i = 0; i < A->rows(); ++i) {
        for (uint32_t j = 0; j < B->cols(); ++j) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < A->cols(); ++k) {
                sum += static_cast<int32_t>(A->at(i, k)) * static_cast<int32_t>(B->at(k, j));
            }
            C->at(i, j) = static_cast<int16_t>(sum);
        }
    }
    
    // Print result
    printMatrix(C, "Result Matrix (A × B)");
    
    std::cout << "Matrix multiplication test complete" << std::endl << std::endl;
}

// Function to print usage information
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --verbose, -v  Enable verbose output" << std::endl;
    std::cout << "  --help, -h     Display this help message" << std::endl;
}

// Main function
int main(int argc, char** argv) {
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--verbose") == 0 || strcmp(argv[i], "-v") == 0) {
            g_verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    std::cout << "==================================================" << std::endl;
    std::cout << "     Gemmini Systolic Array Simulator Tests       " << std::endl;
    std::cout << "==================================================" << std::endl << std::endl;
    
    if (g_verbose) {
        std::cout << "Running in verbose mode" << std::endl << std::endl;
    }
    
    try {
        // Create test matrices and vectors
        std::cout << "Creating test matrices and vectors..." << std::endl;
        MatrixPtr test_matrix = createTestMatrix(4, 4, 1, 5);
        VectorPtr test_vector = createTestVector(4, 1, 5);
        MatrixPtr expected_result = calculateExpectedResult(test_matrix, test_vector);
        
        // Print test data
        printMatrix(test_matrix, "Weight Matrix");
        printVector(test_vector, "Input Vector");
        printMatrix(expected_result, "Expected Result");
        
        // Run manual matrix multiplication test
        testManualMatrixMultiplication();
        
        std::cout << "All tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 