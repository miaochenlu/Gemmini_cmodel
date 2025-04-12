// matrix.hpp - Matrix and Vector data structures for Gemmini simulator
#pragma once

#include <cstdint>
#include <vector>
#include <memory>
#include <iostream>

namespace gemmini
{
    // Forward declarations
    class Matrix;
    class Vector;
    
    // Type definitions
    using MatrixPtr = std::shared_ptr<Matrix>;
    using VectorPtr = std::shared_ptr<Vector>;
    
    // Simple Vector class for data storage and manipulation
    class Vector
    {
    public:
        Vector(uint32_t size) : data_(size, 0) {}
        
        int16_t& operator[](size_t idx) { return data_[idx]; }
        const int16_t& operator[](size_t idx) const { return data_[idx]; }
        
        uint32_t size() const { return data_.size(); }
        
        friend std::ostream& operator<<(std::ostream& os, const Vector& vec) {
            os << "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                os << vec[i];
                if (i < vec.size() - 1) os << ", ";
            }
            os << "]";
            return os;
        }
        
    private:
        std::vector<int16_t> data_;
    };
    
    // Simple Matrix class for data storage and manipulation
    class Matrix
    {
    public:
        Matrix(uint32_t rows, uint32_t cols) : 
            rows_(rows), 
            cols_(cols), 
            data_(rows * cols, 0) {}
        
        int16_t& at(uint32_t row, uint32_t col) { return data_[row * cols_ + col]; }
        const int16_t& at(uint32_t row, uint32_t col) const { return data_[row * cols_ + col]; }
        
        uint32_t rows() const { return rows_; }
        uint32_t cols() const { return cols_; }
        
        friend std::ostream& operator<<(std::ostream& os, const Matrix& mat) {
            for (uint32_t r = 0; r < mat.rows(); ++r) {
                os << "[";
                for (uint32_t c = 0; c < mat.cols(); ++c) {
                    os << mat.at(r, c);
                    if (c < mat.cols() - 1) os << ", ";
                }
                os << "]";
                if (r < mat.rows() - 1) os << std::endl;
            }
            return os;
        }
        
    private:
        uint32_t rows_;
        uint32_t cols_;
        std::vector<int16_t> data_;
    };
    
    // Create matrix shared pointer using standard library
    template<typename T, typename... Args>
    std::shared_ptr<T> create_matrix_ptr(Args&&... args) {
        return std::make_shared<T>(std::forward<Args>(args)...);
    }
} // namespace gemmini 