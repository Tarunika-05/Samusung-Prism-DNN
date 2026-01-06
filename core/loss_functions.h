#pragma once

#include "tensor.h"
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>

/*
 * Supported loss functions
 */
enum class LossType {
    MEAN_SQUARED_ERROR,
    BINARY_CROSS_ENTROPY,
    CATEGORICAL_CROSS_ENTROPY,
    SPARSE_CATEGORICAL_CROSS_ENTROPY
};

/*
 * Unified Loss interface (TensorFlow-like)
 */
class Loss {
public:
    LossType type;
    int num_classes;
    float eps;

    Loss(LossType t, int classes = 0, float epsilon = 1e-7f)
        : type(t), num_classes(classes), eps(epsilon) {}

    /*
     * Forward loss computation
     */
    float forward(const Tensor& y_pred,
                  const std::vector<int>& y_true_sparse,
                  const Tensor* y_true_dense = nullptr) const {
        switch (type) {
            case LossType::MEAN_SQUARED_ERROR:
                return mse(y_pred, *y_true_dense);

            case LossType::BINARY_CROSS_ENTROPY:
                return binary_cross_entropy(y_pred, *y_true_dense);

            case LossType::CATEGORICAL_CROSS_ENTROPY:
                return categorical_cross_entropy(y_pred, *y_true_dense);

            case LossType::SPARSE_CATEGORICAL_CROSS_ENTROPY:
                return sparse_categorical_cross_entropy(y_pred, y_true_sparse);
        }
        return 0.0f;
    }

    /*
     * Backward gradient computation
     */
    Tensor backward(const Tensor& y_pred,
                    const std::vector<int>& y_true_sparse,
                    const Tensor* y_true_dense = nullptr) const {
        switch (type) {
            case LossType::MEAN_SQUARED_ERROR:
                return mse_backward(y_pred, *y_true_dense);

            case LossType::BINARY_CROSS_ENTROPY:
                return binary_cross_entropy_backward(y_pred, *y_true_dense);

            case LossType::CATEGORICAL_CROSS_ENTROPY:
                return categorical_cross_entropy_backward(y_pred, *y_true_dense);

            case LossType::SPARSE_CATEGORICAL_CROSS_ENTROPY:
                return sparse_categorical_cross_entropy_backward(y_pred, y_true_sparse);
        }
        return Tensor();
    }

private:
    // ---------- MSE ----------
    float mse(const Tensor& y_pred, const Tensor& y_true) const {
        float loss = 0.0f;
        for (int i = 0; i < y_pred.rows; ++i)
            for (int j = 0; j < y_pred.cols; ++j)
                loss += std::pow(y_pred(i, j) - y_true(i, j), 2);
        return loss / (y_pred.rows * y_pred.cols);
    }

    Tensor mse_backward(const Tensor& y_pred, const Tensor& y_true) const {
        Tensor grad(y_pred.rows, y_pred.cols);
        float scale = 2.0f / (y_pred.rows * y_pred.cols);
        for (int i = 0; i < grad.rows; ++i)
            for (int j = 0; j < grad.cols; ++j)
                grad(i, j) = scale * (y_pred(i, j) - y_true(i, j));
        return grad;
    }

    // ---------- Binary Cross-Entropy ----------
    float binary_cross_entropy(const Tensor& y_pred,
                               const Tensor& y_true) const {
        float loss = 0.0f;
        for (int i = 0; i < y_pred.rows; ++i) {
            float p = std::clamp(y_pred(i, 0), eps, 1.0f - eps);
            float y = y_true(i, 0);
            loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
        }
        return loss / y_pred.rows;
    }

    Tensor binary_cross_entropy_backward(const Tensor& y_pred,
                                         const Tensor& y_true) const {
        Tensor grad(y_pred.rows, 1);
        for (int i = 0; i < y_pred.rows; ++i) {
            float p = std::clamp(y_pred(i, 0), eps, 1.0f - eps);
            float y = y_true(i, 0);
            grad(i, 0) = (p - y) / (p * (1.0f - p));
        }
        return grad;
    }

    // ---------- Categorical Cross-Entropy ----------
    float categorical_cross_entropy(const Tensor& y_pred,
                                    const Tensor& y_true) const {
        float loss = 0.0f;
        for (int i = 0; i < y_pred.rows; ++i)
            for (int j = 0; j < y_pred.cols; ++j)
                if (y_true(i, j) > 0.0f)
                    loss += -std::log(std::max(y_pred(i, j), eps));
        return loss / y_pred.rows;
    }

    Tensor categorical_cross_entropy_backward(const Tensor& y_pred,
                                              const Tensor& y_true) const {
        Tensor grad(y_pred.rows, y_pred.cols);
        for (int i = 0; i < grad.rows; ++i)
            for (int j = 0; j < grad.cols; ++j)
                grad(i, j) = (y_pred(i, j) - y_true(i, j)) / y_pred.rows;
        return grad;
    }

    // ---------- Sparse Categorical Cross-Entropy ----------
    float sparse_categorical_cross_entropy(const Tensor& y_pred,
                                           const std::vector<int>& y_true) const {
        float loss = 0.0f;
        for (int i = 0; i < y_pred.rows; ++i) {
            float p = std::max(y_pred(i, y_true[i]), eps);
            loss += -std::log(p);
        }
        return loss / y_pred.rows;
    }

    Tensor sparse_categorical_cross_entropy_backward(
        const Tensor& y_pred,
        const std::vector<int>& y_true) const {

        Tensor grad(y_pred.rows, y_pred.cols);
        for (int i = 0; i < y_pred.rows; ++i) {
            for (int j = 0; j < y_pred.cols; ++j)
                grad(i, j) = y_pred(i, j);
            grad(i, y_true[i]) -= 1.0f;
        }

        float inv_batch = 1.0f / y_pred.rows;
        for (float& v : grad.data)
            v *= inv_batch;

        return grad;
    }
};
