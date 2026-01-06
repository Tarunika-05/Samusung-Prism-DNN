#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>

#include "core/tensor.h"
#include "core/dense_layer.h"
#include "core/activations.h"
#include "core/model.h"
#include "core/loss_functions.h"
#include "core/optimizers.h"

/* -------------------------------------------------
   Binary weight loader
------------------------------------------------- */
void load_bin(const std::string& path, std::vector<float>& buffer) {
    std::ifstream fin(path, std::ios::binary);
    if (!fin) {
        std::cerr << "ERROR: Cannot open " << path << std::endl;
        std::exit(1);
    }
    fin.read(reinterpret_cast<char*>(buffer.data()),
             buffer.size() * sizeof(float));
}

/* -------------------------------------------------
   Binary weight saver
------------------------------------------------- */
void save_bin(const std::string& path, const std::vector<float>& buffer) {
    std::ofstream fout(path, std::ios::binary);
    fout.write(reinterpret_cast<const char*>(buffer.data()),
               buffer.size() * sizeof(float));
}

/* -------------------------------------------------
   Load single test input
------------------------------------------------- */
Tensor load_input(const std::string& path) {
    Tensor x(80);
    std::ifstream fin(path);
    for (int i = 0; i < 80; ++i)
        fin >> x[i];
    return x;
}

/* -------------------------------------------------
   Load single label
------------------------------------------------- */
int load_label(const std::string& path) {
    int y;
    std::ifstream fin(path);
    fin >> y;
    return y;
}

int main() {

    std::cout << "\n===== DNN Forward + Backward (C++) =====\n";

    /* -------------------------------------------------
       1. Load input + label
    ------------------------------------------------- */
    Tensor x = load_input("data/test_input.txt");
    int y_true = load_label("data/test_label.txt");
    std::vector<int> y = { y_true };

    /* -------------------------------------------------
       2. Build model (same as TensorFlow)
    ------------------------------------------------- */
    DenseLayer d1(80, 256, ActivationType::RELU);
    DenseLayer d2(256, 128, ActivationType::RELU);
    DenseLayer d3(128, 64,  ActivationType::RELU);
    DenseLayer d4(64,  10,  ActivationType::SOFTMAX);

    Model model;
    model.add(d1);
    model.add(d2);
    model.add(d3);
    model.add(d4);

    /* -------------------------------------------------
       3. Load baseline weights (TF-trained)
    ------------------------------------------------- */
    load_bin("weights/dense1_W.bin", d1.W.data);
    load_bin("weights/dense1_b.bin", d1.b);

    load_bin("weights/dense2_W.bin", d2.W.data);
    load_bin("weights/dense2_b.bin", d2.b);

    load_bin("weights/dense3_W.bin", d3.W.data);
    load_bin("weights/dense3_b.bin", d3.b);

    load_bin("weights/dense4_W.bin", d4.W.data);
    load_bin("weights/dense4_b.bin", d4.b);

    // Sync loaded weights to parameters
    d1.W_param.data = d1.W.data;  d1.b_param.data = d1.b;
    d2.W_param.data = d2.W.data;  d2.b_param.data = d2.b;
    d3.W_param.data = d3.W.data;  d3.b_param.data = d3.b;
    d4.W_param.data = d4.W.data;  d4.b_param.data = d4.b;

    /* -------------------------------------------------
       4. Forward warm-up
    ------------------------------------------------- */
    model.predict(x);

    /* -------------------------------------------------
       5. Forward latency
    ------------------------------------------------- */
    const int N = 100;
    Tensor output;

    auto f_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i)
        output = model.predict(x);
    auto f_end = std::chrono::high_resolution_clock::now();

    double forward_latency_ms =
        std::chrono::duration<double, std::milli>(f_end - f_start).count() / N;

    /* -------------------------------------------------
       6. Print forward output
    ------------------------------------------------- */
    std::cout << "\nOutput probabilities:\n";
    for (int i = 0; i < output.size(); ++i) {
        std::cout << "Class " << i << ": "
                  << std::fixed << std::setprecision(3)
                  << output[i] << "\n";
    }

    std::cout << "\nForward latency: "
              << forward_latency_ms << " ms\n";

    /* -------------------------------------------------
       7. BACKWARD PASS (ONE STEP – correctness)
    ------------------------------------------------- */
    Loss loss_fn(LossType::SPARSE_CATEGORICAL_CROSS_ENTROPY, 10);
    SGDOptimizer opt(0.01f, 0.9f);

    Tensor grad = loss_fn.backward(output, y);
    grad = d4.backward(grad);
    grad = d3.backward(grad);
    grad = d2.backward(grad);
    grad = d1.backward(grad);

    opt.step(d1.W_param); opt.step(d1.b_param);
    opt.step(d2.W_param); opt.step(d2.b_param);
    opt.step(d3.W_param); opt.step(d3.b_param);
    opt.step(d4.W_param); opt.step(d4.b_param);

    d1.sync_weights();
    d2.sync_weights();
    d3.sync_weights();
    d4.sync_weights();

    /* -------------------------------------------------
       8. Save UPDATED weights (TF comparison)
    ------------------------------------------------- */
    save_bin("updated_weights/dense1_W_updated.bin", d1.W.data);
    save_bin("updated_weights/dense1_b_updated.bin", d1.b);

    save_bin("updated_weights/dense2_W_updated.bin", d2.W.data);
    save_bin("updated_weights/dense2_b_updated.bin", d2.b);

    save_bin("updated_weights/dense3_W_updated.bin", d3.W.data);
    save_bin("updated_weights/dense3_b_updated.bin", d3.b);

    save_bin("updated_weights/dense4_W_updated.bin", d4.W.data);
    save_bin("updated_weights/dense4_b_updated.bin", d4.b);

    std::cout << "\nUpdated C++ weights saved (correctness)\n";

    /* -------------------------------------------------
       9. BACKWARD + UPDATE LATENCY (separate)
    ------------------------------------------------- */
    // Reload baseline weights
    load_bin("weights/dense1_W.bin", d1.W.data);
    load_bin("weights/dense1_b.bin", d1.b);
    load_bin("weights/dense2_W.bin", d2.W.data);
    load_bin("weights/dense2_b.bin", d2.b);
    load_bin("weights/dense3_W.bin", d3.W.data);
    load_bin("weights/dense3_b.bin", d3.b);
    load_bin("weights/dense4_W.bin", d4.W.data);
    load_bin("weights/dense4_b.bin", d4.b);

    d1.W_param.data = d1.W.data;  d1.b_param.data = d1.b;
    d2.W_param.data = d2.W.data;  d2.b_param.data = d2.b;
    d3.W_param.data = d3.W.data;  d3.b_param.data = d3.b;
    d4.W_param.data = d4.W.data;  d4.b_param.data = d4.b;

    // Warm-up backward once
    output = model.predict(x);
    grad = loss_fn.backward(output, y);
    grad = d4.backward(grad);
    grad = d3.backward(grad);
    grad = d2.backward(grad);
    grad = d1.backward(grad);

    opt.step(d1.W_param); opt.step(d1.b_param);
    opt.step(d2.W_param); opt.step(d2.b_param);
    opt.step(d3.W_param); opt.step(d3.b_param);
    opt.step(d4.W_param); opt.step(d4.b_param);

    d1.sync_weights();
    d2.sync_weights();
    d3.sync_weights();
    d4.sync_weights();

    const int B = 100;

    auto b_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < B; ++i) {

        output = model.predict(x);
        grad = loss_fn.backward(output, y);

        grad = d4.backward(grad);
        grad = d3.backward(grad);
        grad = d2.backward(grad);
        grad = d1.backward(grad);

        opt.step(d1.W_param); opt.step(d1.b_param);
        opt.step(d2.W_param); opt.step(d2.b_param);
        opt.step(d3.W_param); opt.step(d3.b_param);
        opt.step(d4.W_param); opt.step(d4.b_param);

        d1.sync_weights();
        d2.sync_weights();
        d3.sync_weights();
        d4.sync_weights();
    }
    auto b_end = std::chrono::high_resolution_clock::now();

    double backward_latency_ms =
        std::chrono::duration<double, std::milli>(b_end - b_start).count() / B;

    std::cout << "\nBackward + update latency: "
              << backward_latency_ms << " ms\n";

    std::cout << "\n===== C++ Forward & Backward Complete =====\n";
    return 0;
}
