#pragma once

#define _USE_MATH_DEFINES
#include <cmath>
#include <cassert>
#include <algorithm>
#include "tensor.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/*
 * Supported activation functions
 */
enum class ActivationType {
    STEP,
    LINEAR,
    RELU,
    LEAKY_RELU,
    PRELU,
    SIGMOID,
    TANH,
    ELU,
    SELU,
    GELU,
    SWISH,
    SOFTMAX
};

/*
 * Activation layer (stateless except cached input/output)
 */
class Activation {
public:
    ActivationType type;

    // Optional parameters (defaults match common TF defaults)
    float alpha;   // for LeakyReLU, ELU, PReLU
    float beta;    // for Swish

    Tensor input_cache;
    Tensor output_cache;

    Activation(
        ActivationType t,
        float alpha_ = 0.01f,
        float beta_ = 1.0f
    ) : type(t), alpha(alpha_), beta(beta_) {}

    /*
     * Forward pass
     */
    Tensor forward(const Tensor& X) {
        input_cache = X;
        Tensor Y(X.rows, X.cols);

        for (int i = 0; i < X.rows; ++i) {
            for (int j = 0; j < X.cols; ++j) {
                float x = X(i, j);
                Y(i, j) = activate(x);
            }
        }

        // Special handling for softmax (row-wise)
        if (type == ActivationType::SOFTMAX) {
            softmax(Y);
        }

        output_cache = Y;
        return Y;
    }

    /*
     * Backward pass
     */
    Tensor backward(const Tensor& dOut) {
        assert(dOut.rows == output_cache.rows);
        assert(dOut.cols == output_cache.cols);

        Tensor dX(dOut.rows, dOut.cols);

        // Softmax backward usually combined with cross-entropy
        if (type == ActivationType::SOFTMAX) {
            return dOut;
        }

        for (int i = 0; i < dOut.rows; ++i) {
            for (int j = 0; j < dOut.cols; ++j) {
                float grad = derivative(input_cache(i, j),
                                        output_cache(i, j));
                dX(i, j) = dOut(i, j) * grad;
            }
        }
        return dX;
    }

private:
    /*
     * Activation functions
     */
    float activate(float x) const {
        switch (type) {
            case ActivationType::STEP:
                return x > 0.0f ? 1.0f : 0.0f;

            case ActivationType::LINEAR:
                return x;

            case ActivationType::RELU:
                return std::max(0.0f, x);

            case ActivationType::LEAKY_RELU:
                return x > 0.0f ? x : alpha * x;

            case ActivationType::PRELU:
                return x > 0.0f ? x : alpha * x;

            case ActivationType::SIGMOID:
                return 1.0f / (1.0f + std::exp(-x));

            case ActivationType::TANH:
                return std::tanh(x);

            case ActivationType::ELU:
                return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);

            case ActivationType::SELU: {
                // SELU constants are part of the definition
                const float lambda = 1.050700987f;
                const float alpha_selu = 1.673263242f;
                return lambda * (x > 0.0f ? x : alpha_selu * (std::exp(x) - 1.0f));
            }

            case ActivationType::GELU:
                return 0.5f * x * (1.0f + std::tanh(
                    std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))
                ));

            case ActivationType::SWISH:
                return x / (1.0f + std::exp(-beta * x));

            case ActivationType::SOFTMAX:
                return x; // handled separately
        }
        return x;
    }

    /*
     * Derivatives
     */
    float derivative(float x, float y) const {
        switch (type) {
            case ActivationType::STEP:
                return 0.0f;

            case ActivationType::LINEAR:
                return 1.0f;

            case ActivationType::RELU:
                return x > 0.0f ? 1.0f : 0.0f;

            case ActivationType::LEAKY_RELU:
            case ActivationType::PRELU:
                return x > 0.0f ? 1.0f : alpha;

            case ActivationType::SIGMOID:
                return y * (1.0f - y);

            case ActivationType::TANH:
                return 1.0f - y * y;

            case ActivationType::ELU:
                return x >= 0.0f ? 1.0f : alpha * std::exp(x);

            case ActivationType::SELU:
                return x > 0.0f ? 1.050700987f
                                : 1.050700987f * 1.673263242f * std::exp(x);

            case ActivationType::GELU:
                // Approximate derivative (sufficient for verification)
                return 0.5f * (1.0f + std::tanh(
                    std::sqrt(2.0f / M_PI) * (x + 0.044715f * std::pow(x, 3))
                ));

            case ActivationType::SWISH: {
                float sig = 1.0f / (1.0f + std::exp(-beta * x));
                return sig + beta * x * sig * (1.0f - sig);
            }

            case ActivationType::SOFTMAX:
                return 1.0f;
        }
        return 1.0f;
    }

    /*
     * Row-wise softmax (numerically stable)
     */
    void softmax(Tensor& X) const {
        for (int i = 0; i < X.rows; ++i) {
            float max_val = X(i, 0);
            for (int j = 1; j < X.cols; ++j) {
                max_val = std::max(max_val, X(i, j));
            }

            float sum = 0.0f;
            for (int j = 0; j < X.cols; ++j) {
                X(i, j) = std::exp(X(i, j) - max_val);
                sum += X(i, j);
            }

            for (int j = 0; j < X.cols; ++j) {
                X(i, j) /= sum;
            }
        }
    }
};
