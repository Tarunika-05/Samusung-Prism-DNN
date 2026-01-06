#pragma once
#include <vector>
#include <iostream>
#include <cassert>

#include "tensor.h"
#include "dense_layer.h"
#include "loss_functions.h"
#include "optimizers.h"



class Model {
private:
    std::vector<DenseLayer*> layers;
    Loss* loss_fn = nullptr;
    Optimizer* optimizer = nullptr;

    /* -------- INTERNAL ENGINE (HIDDEN FROM USER) -------- */

    Tensor forward_internal(const Tensor& input) {
        Tensor x = input;
        for (auto* layer : layers) {
            x = layer->forward(x);
        }
        return x;
    }

    void backward_internal(const Tensor& grad_output) {
        Tensor grad = grad_output;
        for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i) {
            grad = layers[i]->backward(grad);
        }
    }

    void optimizer_step() {
        // Note: This is a simplified version. For full optimizer support,
        // DenseLayer would need to expose Parameter structs.
        // For now, this is a placeholder that does nothing.
        // The actual weight updates should be handled in the backward pass.
    }

public:
    /* -------- MODEL CONSTRUCTION -------- */

    void add(DenseLayer& layer) {
        layers.push_back(&layer);
    }

    void compile(Loss& loss, Optimizer& opt) {
        loss_fn = &loss;
        optimizer = &opt;
    }

    /* -------- TRAINING (TensorFlow: model.fit) -------- */

    void fit(const std::vector<Tensor>& X,
             const std::vector<int>& y,
             int epochs,
             int batch_size = 1) {

        assert(loss_fn && optimizer && "Model must be compiled before training");

        for (int epoch = 0; epoch < epochs; ++epoch) {
            float epoch_loss = 0.0f;
            int correct = 0;

            for (size_t i = 0; i < X.size(); ++i) {
                // Forward
                Tensor output = forward_internal(X[i]);

                // Loss
                std::vector<int> y_vec = {y[i]};
                float loss = loss_fn->forward(output, y_vec);
                epoch_loss += loss;

                // Accuracy
                if (argmax(output) == y[i]) correct++;

                // Backward
                Tensor grad = loss_fn->backward(output, y_vec);
                backward_internal(grad);

                // Update
                optimizer_step();
            }

            std::cout << "Epoch " << epoch + 1
                      << " | Loss: " << epoch_loss / X.size()
                      << " | Accuracy: "
                      << static_cast<float>(correct) / X.size()
                      << std::endl;
        }
    }

    /* -------- EVALUATION (TensorFlow: model.evaluate) -------- */

    float evaluate(const std::vector<Tensor>& X,
                   const std::vector<int>& y) {

        assert(loss_fn && "Loss function not set");

        float total_loss = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < X.size(); ++i) {
            Tensor output = forward_internal(X[i]);
            std::vector<int> y_vec = {y[i]};
            total_loss += loss_fn->forward(output, y_vec);

            if (argmax(output) == y[i]) correct++;
        }

        float accuracy = static_cast<float>(correct) / X.size();

        std::cout << "Evaluation Loss: " << total_loss / X.size() << std::endl;
        std::cout << "Evaluation Accuracy: " << accuracy << std::endl;

        return accuracy;
    }

    /* -------- INFERENCE (TensorFlow: model.predict) -------- */

    Tensor predict(const Tensor& input) {
        return forward_internal(input);
    }


};
