// matrix.hpp - Matrix and Vector data structures for Gemmini simulator
#pragma once

#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "gemmini/common.hpp"

BEGIN_NS(gemmini)

// Forward declarations
class Matrix;
class Vector;

// Type definitions
using MatrixPtr = std::shared_ptr<Matrix>;
using VectorPtr = std::shared_ptr<Vector>;

// Simple Vector class for data storage and manipulation
class Vector {
public:
    Vector(uint32_t size) : mData(size, 0) {}

    int16_t & operator[](size_t idx) { return mData[idx]; }

    const int16_t & operator[](size_t idx) const { return mData[idx]; }

    uint32_t Size() const { return mData.size(); }

    friend std::ostream & operator<<(std::ostream & os, const Vector & vec) {
        os << "[";
        for (size_t i = 0; i < vec.Size(); ++i) {
            os << vec[i];
            if (i < vec.Size() - 1)
                os << ", ";
        }
        os << "]";
        return os;
    }

private:
    std::vector<int16_t> mData;
};

// Simple Matrix class for data storage and manipulation
class Matrix {
public:
    Matrix(uint32_t rows, uint32_t cols) : mRows(rows), mCols(cols), mData(rows * cols, 0) {}

    int16_t & At(uint32_t row, uint32_t col) { return mData[row * mCols + col]; }

    const int16_t & At(uint32_t row, uint32_t col) const { return mData[row * mCols + col]; }

    uint32_t Rows() const { return mRows; }

    uint32_t Cols() const { return mCols; }

    friend std::ostream & operator<<(std::ostream & os, const Matrix & mat) {
        for (uint32_t r = 0; r < mat.Rows(); ++r) {
            os << "[";
            for (uint32_t c = 0; c < mat.Cols(); ++c) {
                os << mat.At(r, c);
                if (c < mat.Cols() - 1)
                    os << ", ";
            }
            os << "]";
            if (r < mat.Rows() - 1)
                os << std::endl;
        }
        return os;
    }

private:
    uint32_t mRows;
    uint32_t mCols;
    std::vector<int16_t> mData;
};

// Create matrix shared pointer using standard library
template <typename T, typename... Args> std::shared_ptr<T> CreateMatrixPtr(Args &&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

END_NS(gemmini)