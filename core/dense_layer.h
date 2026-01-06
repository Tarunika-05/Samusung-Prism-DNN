#pragma once

#include "tensor.h"
#include "activations.h"
#include "optimizers.h"
#include <vector>
#include <cassert>

/*
 * Fully Connected (Dense) Layer
 * Implements:
 *   Forward:  Y = activation(X * W + b)
 *   Backward: dW, db, dX
 */
class DenseLayer {
public:
    // Parameters
    Tensor W;                  // (input_dim x output_dim)
    std::vector<float> b;      // (output_dim)

    // Gradients
    Tensor grad_W;
    std::vector<float> grad_b;

    // Parameters for optimizer
    Parameter W_param;
    Parameter b_param;

    // Cache (for backprop)
    Tensor input_cache;

    // Activation function
    Activation activation;

    DenseLayer(int input_dim, int output_dim, ActivationType act_type = ActivationType::LINEAR)
        : W(input_dim, output_dim),
          b(output_dim, 0.0f),
          grad_W(input_dim, output_dim),
          grad_b(output_dim, 0.0f),
          activation(act_type) {
        // Initialize parameters to reference the same data
        W_param.data = W.data;
        W_param.grad = grad_W.data;
        b_param.data = b;
        b_param.grad = grad_b;
    }

    /*
     * Forward pass
     * X: (batch_size x input_dim)
     */
    Tensor forward(const Tensor& X) {
        assert(X.cols == W.rows);

        input_cache = X;  // cache input for backward

        Tensor out = matmul(X, W);
        add_bias(out, b);
        return activation.forward(out);
    }

    /*
     * Backward pass
     * dOut: gradient from next layer (batch_size x output_dim)
     *
     * Returns:
     * dX: gradient w.r.t input (batch_size x input_dim)
     */
    Tensor backward(const Tensor& dOut) {
        assert(dOut.cols == W.cols);
        assert(dOut.rows == input_cache.rows);

        // Apply activation backward
        Tensor dOut_activated = activation.backward(dOut);

        // dW = X^T * dOut_activated
        Tensor X_T = transpose(input_cache);
        grad_W = matmul(X_T, dOut_activated);

        // db = sum over batch
        std::fill(grad_b.begin(), grad_b.end(), 0.0f);
        for (int i = 0; i < dOut_activated.rows; ++i) {
            for (int j = 0; j < dOut_activated.cols; ++j) {
                grad_b[j] += dOut_activated(i, j);
            }
        }

        // Sync gradients to parameters (for optimizer)
        sync_gradients();

        // dX = dOut_activated * W^T
        Tensor W_T = transpose(W);
        Tensor dX = matmul(dOut_activated, W_T);

        return dX;
    }

    // Sync gradients from grad_W/grad_b to W_param/b_param
    void sync_gradients() {
        W_param.grad = grad_W.data;
        b_param.grad = grad_b;
    }

    // Sync weights from W_param/b_param back to W/b
    void sync_weights() {
        W.data = W_param.data;
        b = b_param.data;
    }
};
