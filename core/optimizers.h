#pragma once
#include <vector>
#include <cmath>
#include <unordered_map>

/*
 Each parameter has:
 - data  : actual weights / biases
 - grad  : gradient computed in backward pass
*/
struct Parameter {
    std::vector<float> data;
    std::vector<float> grad;
};

/*
 Base optimizer interface
 */
class Optimizer {
public:
    virtual void step(Parameter& param) = 0;
    virtual ~Optimizer() = default;
};

class SGDOptimizer : public Optimizer {
    private:
        float lr;
        float momentum;
        std::unordered_map<Parameter*, std::vector<float>> velocity;
    
    public:
        // Pure SGD
        explicit SGDOptimizer(float learning_rate)
            : lr(learning_rate), momentum(0.0f) {}
    
        // SGD + Momentum
        SGDOptimizer(float learning_rate, float momentum_factor)
            : lr(learning_rate), momentum(momentum_factor) {}
    
        void step(Parameter& param) override {
            if (momentum > 0.0f) {
                auto& v = velocity[&param];
                if (v.empty())
                    v.resize(param.data.size(), 0.0f);
    
                for (size_t i = 0; i < param.data.size(); ++i) {
                    v[i] = momentum * v[i] - lr * param.grad[i];
                    param.data[i] += v[i];
                }
            } else {
                for (size_t i = 0; i < param.data.size(); ++i) {
                    param.data[i] -= lr * param.grad[i];
                }
            }
        }
};
class RMSPropOptimizer : public Optimizer {
    private:
        float lr;
        float beta;
        float eps;
        std::unordered_map<Parameter*, std::vector<float>> cache;
    
    public:
        explicit RMSPropOptimizer(float learning_rate,
                                  float beta = 0.9f,
                                  float epsilon = 1e-8f)
            : lr(learning_rate), beta(beta), eps(epsilon) {}
    
        void step(Parameter& param) override {
            auto& v = cache[&param];
            if (v.empty())
                v.resize(param.data.size(), 0.0f);
    
            for (size_t i = 0; i < param.data.size(); ++i) {
                v[i] = beta * v[i] + (1.0f - beta) * param.grad[i] * param.grad[i];
                param.data[i] -= lr * param.grad[i] / (std::sqrt(v[i]) + eps);
            }
        }
};
    
class AdamOptimizer : public Optimizer {
    private:
        float lr;
        float beta1;
        float beta2;
        float eps;
        int timestep;
    
        std::unordered_map<Parameter*, std::vector<float>> m;
        std::unordered_map<Parameter*, std::vector<float>> v;
    
    public:
        explicit AdamOptimizer(float learning_rate,
                               float beta1 = 0.9f,
                               float beta2 = 0.999f,
                               float epsilon = 1e-8f)
            : lr(learning_rate),
              beta1(beta1),
              beta2(beta2),
              eps(epsilon),
              timestep(0) {}
    
        void step(Parameter& param) override {
            timestep++;
    
            auto& m_vec = m[&param];
            auto& v_vec = v[&param];
    
            if (m_vec.empty()) {
                m_vec.resize(param.data.size(), 0.0f);
                v_vec.resize(param.data.size(), 0.0f);
            }
    
            for (size_t i = 0; i < param.data.size(); ++i) {
                m_vec[i] = beta1 * m_vec[i] + (1.0f - beta1) * param.grad[i];
                v_vec[i] = beta2 * v_vec[i] + (1.0f - beta2) * param.grad[i] * param.grad[i];
    
                float m_hat = m_vec[i] / (1.0f - std::pow(beta1, timestep));
                float v_hat = v_vec[i] / (1.0f - std::pow(beta2, timestep));
    
                param.data[i] -= lr * m_hat / (std::sqrt(v_hat) + eps);
            }
        }
};
    