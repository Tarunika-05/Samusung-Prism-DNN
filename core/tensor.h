#pragma once

#include <vector>
#include <cassert>
#include <iostream>

/*
 * Simple 2D Tensor (Matrix) structure
 * Used as the numerical backbone for all models
 */
struct Tensor {
    int rows;
    int cols;
    std::vector<float> data;

    Tensor() : rows(0), cols(0) {}

    Tensor(int r, int c)
        : rows(r), cols(c), data(r * c, 0.0f) {}

    // Vector-like constructor (for 1D tensors)
    explicit Tensor(int size)
        : rows(1), cols(size), data(size, 0.0f) {}

    float& operator()(int r, int c) {
        return data[r * cols + c];
    }

    float operator()(int r, int c) const {
        return data[r * cols + c];
    }

    // Vector-like access (for 1D tensors)
    float& operator[](int i) {
        assert(rows == 1 && i < cols);
        return data[i];
    }

    float operator[](int i) const {
        assert(rows == 1 && i < cols);
        return data[i];
    }

    // Size for vector-like access
    int size() const {
        if (rows == 1) return cols;
        return rows * cols;
    }
};

/*
 * Matrix multiplication: C = A * B
 * A: (m x n)
 * B: (n x p)
 * C: (m x p)
 */
inline Tensor matmul(const Tensor& A, const Tensor& B) {
    assert(A.cols == B.rows);

    Tensor C(A.rows, B.cols);

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

/*
 * Add bias vector to each row
 * A: (batch x features)
 * b: (features)
 */
inline void add_bias(Tensor& A, const std::vector<float>& b) {
    assert(A.cols == static_cast<int>(b.size()));

    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            A(i, j) += b[j];
        }
    }
}

/*
 * Transpose of a matrix
 */
inline Tensor transpose(const Tensor& A) {
    Tensor T(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            T(j, i) = A(i, j);
        }
    }
    return T;
}

/*
 * Print tensor (for debugging / verification)
 */
inline void print_tensor(const Tensor& A, const std::string& name = "") {
    if (!name.empty()) {
        std::cout << name << ":\n";
    }
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            std::cout << A(i, j) << " ";
        }
        std::cout << "\n";
    }
}

/*
 * Find index of maximum value in tensor (for classification)
 */
inline int argmax(const Tensor& A) {
    int max_idx = 0;
    float max_val = A(0, 0);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            if (A(i, j) > max_val) {
                max_val = A(i, j);
                max_idx = i * A.cols + j;
            }
        }
    }
    // For 1D tensors (rows=1), return column index
    if (A.rows == 1) {
        return max_idx;
    }
    // For 2D tensors, return column index of first row (typical for batch_size=1)
    return max_idx % A.cols;
}
