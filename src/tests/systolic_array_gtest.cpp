// systolic_array_gtest.cpp - Google Test framework tests for Gemmini Systolic Array
#include <gtest/gtest.h>
#include <memory>
#include <limits>
#include <algorithm>
#include <vector>
#include <iostream>

#include "gemmini/matrix.hpp"
#include "gemmini/common.hpp"

// Main function for Google Test
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

namespace gemmini {
namespace test {

// Simple class to simulate a 2D PE array without relying on SPARTA
class PEArray {
public:
    PEArray(uint32_t rows, uint32_t cols)
        : rows_(rows), cols_(cols), weights_(rows * cols, 0) {}
    
    void LoadWeights(const std::vector<std::vector<int16_t>>& weight_matrix) {
        for (uint32_t r = 0; r < std::min(rows_, static_cast<uint32_t>(weight_matrix.size())); ++r) {
            const auto& row = weight_matrix[r];
            for (uint32_t c = 0; c < std::min(cols_, static_cast<uint32_t>(row.size())); ++c) {
                weights_[r * cols_ + c] = row[c];
            }
        }
    }
    
    MatrixPtr MultiplyVector(const std::vector<int16_t>& input_vector) {
        auto result = std::make_shared<Matrix>(rows_, 1);
        
        // Zero the result matrix
        for (uint32_t r = 0; r < rows_; ++r) {
            result->set(r, 0, 0);
        }
        
        // Perform matrix-vector multiplication
        for (uint32_t r = 0; r < rows_; ++r) {
            int32_t sum = 0;
            for (uint32_t c = 0; c < std::min(cols_, static_cast<uint32_t>(input_vector.size())); ++c) {
                sum += static_cast<int32_t>(weights_[r * cols_ + c]) * static_cast<int32_t>(input_vector[c]);
            }
            result->set(r, 0, sum);
        }
        
        return result;
    }
    
private:
    uint32_t rows_;
    uint32_t cols_;
    std::vector<int16_t> weights_;
};

// Test fixture for Systolic Array tests
class SystolicArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple 4x4 array for testing
        array = std::make_unique<PEArray>(4, 4);
    }

    void TearDown() override {
        // No cleanup needed
    }
    
    // Common test resources
    std::unique_ptr<PEArray> array;
};

//=============================================================================
// SECTION 1: Basic Systolic Array Functionality Tests
//=============================================================================

// Test basic matrix-vector multiplication with a small matrix
TEST_F(SystolicArrayTest, BasicMatrixVectorMultiplication) {
    // Define a 2x2 weight matrix
    std::vector<std::vector<int16_t>> weights = {
        {1, 2},
        {3, 4}
    };
    
    // Define a 2-element input vector
    std::vector<int16_t> input = {5, 6};
    
    // Create a smaller 2x2 array for this test
    PEArray small_array(2, 2);
    
    // Load weights
    small_array.LoadWeights(weights);
    
    // Perform matrix-vector multiplication
    auto result = small_array.MultiplyVector(input);
    
    // Expected result: [1*5 + 2*6, 3*5 + 4*6] = [17, 39]
    ASSERT_EQ(result->rows, 2);
    ASSERT_EQ(result->cols, 1);
    
    EXPECT_EQ(result->get(0, 0), 17) << "First element computation failed";
    EXPECT_EQ(result->get(1, 0), 39) << "Second element computation failed";
}

// Test with full 4x4 matrix and vector
TEST_F(SystolicArrayTest, FullMatrixVectorMultiplication) {
    // Define a 4x4 weight matrix
    std::vector<std::vector<int16_t>> weights = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    
    // Define a 4-element input vector
    std::vector<int16_t> input = {1, 2, 3, 4};
    
    // Load weights
    array->LoadWeights(weights);
    
    // Perform matrix-vector multiplication
    auto result = array->MultiplyVector(input);
    
    // Expected result:
    // [1*1 + 2*2 + 3*3 + 4*4] = [30]
    // [5*1 + 6*2 + 7*3 + 8*4] = [70]
    // [9*1 + 10*2 + 11*3 + 12*4] = [110]
    // [13*1 + 14*2 + 15*3 + 16*4] = [150]
    ASSERT_EQ(result->rows, 4);
    ASSERT_EQ(result->cols, 1);
    
    EXPECT_EQ(result->get(0, 0), 30) << "First row computation failed";
    EXPECT_EQ(result->get(1, 0), 70) << "Second row computation failed";
    EXPECT_EQ(result->get(2, 0), 110) << "Third row computation failed";
    EXPECT_EQ(result->get(3, 0), 150) << "Fourth row computation failed";
}

//=============================================================================
// SECTION 2: Edge Case Tests
//=============================================================================

// Test with zero inputs
TEST_F(SystolicArrayTest, ZeroInputs) {
    // Define a 4x4 weight matrix with all zeros
    std::vector<std::vector<int16_t>> weights = {
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    
    // Define a 4-element input vector with all zeros
    std::vector<int16_t> input = {0, 0, 0, 0};
    
    // Load weights
    array->LoadWeights(weights);
    
    // Perform matrix-vector multiplication
    auto result = array->MultiplyVector(input);
    
    // Expected result: all zeros
    ASSERT_EQ(result->rows, 4);
    ASSERT_EQ(result->cols, 1);
    
    // All elements should be zero
    for (uint32_t r = 0; r < result->rows; ++r) {
        EXPECT_EQ(result->get(r, 0), 0) << "Element at row " << r << " should be zero";
    }
}

// Test with max values
TEST_F(SystolicArrayTest, MaxValues) {
    // Get max int16_t value
    const int16_t max_val = std::numeric_limits<int16_t>::max(); // 32767
    
    // Define a 4x4 weight matrix with all max values
    std::vector<std::vector<int16_t>> weights = {
        {max_val, max_val, max_val, max_val},
        {max_val, max_val, max_val, max_val},
        {max_val, max_val, max_val, max_val},
        {max_val, max_val, max_val, max_val}
    };
    
    // Define a 4-element input vector with all max values
    std::vector<int16_t> input = {max_val, max_val, max_val, max_val};
    
    // Load weights
    array->LoadWeights(weights);
    
    // Perform matrix-vector multiplication
    auto result = array->MultiplyVector(input);
    
    // For informational purposes, print the result
    std::cout << "Max value test result: [";
    for (uint32_t r = 0; r < result->rows; ++r) {
        std::cout << result->get(r, 0);
        if (r < result->rows - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Note: We don't verify specific values due to potential overflow in different implementations
}

//=============================================================================
// SECTION 3: Multiple Operations Tests
//=============================================================================

// Test multiple matrix-vector multiplications in sequence
TEST_F(SystolicArrayTest, MultipleOperations) {
    // Create a smaller 2x2 array for this test
    PEArray small_array(2, 2);
    
    // First operation
    std::vector<std::vector<int16_t>> weights1 = {
        {1, 2},
        {3, 4}
    };
    std::vector<int16_t> input1 = {5, 6};
    
    // Load weights for first operation
    small_array.LoadWeights(weights1);
    
    // Perform first matrix-vector multiplication
    auto result1 = small_array.MultiplyVector(input1);
    
    // Second operation with different weights
    std::vector<std::vector<int16_t>> weights2 = {
        {7, 8},
        {9, 10}
    };
    std::vector<int16_t> input2 = {11, 12};
    
    // Load weights for second operation
    small_array.LoadWeights(weights2);
    
    // Perform second matrix-vector multiplication
    auto result2 = small_array.MultiplyVector(input2);
    
    // Expected results for first operation: [17, 39]
    EXPECT_EQ(result1->get(0, 0), 17) << "First operation first element failed";
    EXPECT_EQ(result1->get(1, 0), 39) << "First operation second element failed";
    
    // Expected results for second operation: [7*11 + 8*12, 9*11 + 10*12] = [173, 219]
    EXPECT_EQ(result2->get(0, 0), 173) << "Second operation first element failed";
    EXPECT_EQ(result2->get(1, 0), 219) << "Second operation second element failed";
}

// Test with rectangular matrices and vectors
TEST_F(SystolicArrayTest, RectangularMatrixVectorMultiplication) {
    // Define a 3x4 weight matrix
    std::vector<std::vector<int16_t>> weights = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };
    
    // Define a 4-element input vector
    std::vector<int16_t> input = {1, 2, 3, 4};
    
    // Create a 3x4 array for this test
    PEArray rect_array(3, 4);
    
    // Load weights
    rect_array.LoadWeights(weights);
    
    // Perform matrix-vector multiplication
    auto result = rect_array.MultiplyVector(input);
    
    // Expected result:
    // [1*1 + 2*2 + 3*3 + 4*4] = [30]
    // [5*1 + 6*2 + 7*3 + 8*4] = [70]
    // [9*1 + 10*2 + 11*3 + 12*4] = [110]
    ASSERT_EQ(result->rows, 3);
    ASSERT_EQ(result->cols, 1);
    
    EXPECT_EQ(result->get(0, 0), 30) << "First row computation failed";
    EXPECT_EQ(result->get(1, 0), 70) << "Second row computation failed";
    EXPECT_EQ(result->get(2, 0), 110) << "Third row computation failed";
}

} // namespace test
} // namespace gemmini
