/*
Demonstrate the Eigen automatic differentiation module by training a multilayer perceptron (fully-connected feedforward neural-network).
Python3.8 and matplotlib are used for plotting the results.
Compile with:
    g++ eigen3_autodiff.cpp -o eigen3_autodiff -std=c++11 -O3 -ffast-math -I /usr/include/eigen3/ -I /usr/local/include/matplotlib-cpp.h -I /usr/include/python3.8 -l python3.8
    ./eigen3_autodiff 0.01 5  # step hidden_layer1_size hidden_layer2_size ...
*/
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>
#include "matplotlibcpp.h"
namespace pyplot = matplotlibcpp;

////////////////////////////////////////////////// ALIASES

using Jet = Eigen::AutoDiffScalar<Eigen::VectorXd>;                // real number with many dual parts
using Jetvec = Eigen::Matrix<Jet, Eigen::Dynamic, 1>;              // vector of jets
using Jetmat = Eigen::Matrix<Jet, Eigen::Dynamic, Eigen::Dynamic>; // matrix of jets

////////////////////////////////////////////////// HELPERS

// Returns a random real uniform on [-1, 1]
inline double randf();

// Returns a vector of just the real parts of the given jet vector
inline Eigen::VectorXd reals(Jetvec const& jv);

// Pulls scalar data embedded in Eigen objects out to plain doubles
inline std::vector<double> extract_values(std::vector<Eigen::Matrix<double, 1, 1>> const& vec_eig);

////////////////////////////////////////////////// MAIN CLASS

class MLP {
    std::vector<int> const dims; // network dimensionalities (input_size, hidden_layer1_size, ..., output_size)
    std::vector<Jetmat> weights; // container of weight matrices
    std::vector<Jetvec> biases; // container of bias vectors
    std::vector<Jet*> params; // pointers to each individual parameter

public:
    MLP(std::vector<int> const& dims) :
        dims(dims),
        weights(dims.size()-1),
        biases(dims.size()-1) {
        // Parse the dims architecture to count up the parameters
        int n_params = 0;
        for(int layer=0; layer<weights.size(); ++layer) {
            weights[layer] = Jetmat(dims[layer+1], dims[layer]);
            biases[layer] = Jetvec(dims[layer+1]);
            n_params += weights[layer].size() + biases[layer].size();
        }
        // Initialize parameters with random reals and orthogonal basis on the duals
        int param_id = 0;
        for(Jetmat & W : weights) {
            for(int col=0; col<W.cols(); ++col) {
                for(int row=0; row<W.rows(); ++row) {
                    W(row, col) = Jet(randf(), n_params, param_id); // (value, number_of_dual_parts, which_dual_part)
                    params.push_back(&W(row, col));
                    ++param_id;
                }
            }
        }
        for(Jetvec & b : biases) {
            for(int row=0; row<b.rows(); ++row) {
                b(row) = Jet(randf(), n_params, param_id);
                params.push_back(&b(row));
                ++param_id;
            }
        }
        assert(params.size() == n_params);
    }

    ////////////////////////////////////////////////// GETTERS

    inline std::vector<int> get_dims() const {
        return dims;
    }

    inline std::vector<Jetmat> get_weights() const {
        return weights;
    }

    inline std::vector<Jetvec> get_biases() const {
        return biases;
    }

    ////////////////////////////////////////////////// WORKERS

    // Returns the sigmoid function of a given scalar jet
    inline Jet neuron(Jet const& x) const {
        return pow(Jet(1)+exp(-x), -1);
    }

    // Returns the jet vector MLP output corresponding to the given real input vector
    template <class D>
    inline Jetvec feedforward(Eigen::MatrixBase<D> const& input) const {
        Jetvec act = input.template cast<Jet>();
        for(int layer=0; layer<weights.size()-1; ++layer) {
            act = weights[layer]*act;
            act += biases[layer]; // bit of Eigen weirdness, this has to be a separate statement
            for(int i=0; i<act.size(); ++i) act(i) = neuron(act(i));
        }
        return weights[weights.size()-1]*act + biases[biases.size()-1]; // don't saturate last layer
    }

    // Calls feedforward on each input in an input set to produce the MLP's corresponding output set
    template <class V>
    inline std::vector<V> feedforward(std::vector<V> const& input_set) const {
        std::vector<V> output_set;
        for(V const& input : input_set) {
            output_set.push_back(reals(feedforward(input)));
        }
        return output_set;
    }

    // Common quadratic cost function for a single input-output pair
    template <class D>
    Jet inline cost_function(Eigen::MatrixBase<D> const& input, Eigen::MatrixBase<D> const& output) const {
        Jetvec err = output.template cast<Jet>() - feedforward(input);
        return err.dot(err);
    }

    // Cost function summed over an entire input-output pair batch
    template <class V>
    Jet inline cost_function(std::vector<V> const& input_set, std::vector<V> const& output_set) const {
        Jet cost(0);
        for(int i=0; i<input_set.size(); ++i) {
            cost += cost_function(input_set[i], output_set[i]);
        }
        return cost;
    }

    // Stochastic gradient descent
    template <class V>
    void train(std::vector<V> const& input_set, std::vector<V> const& output_set,
               double step, double tol=1e-5, int max_epoch=10000) {
        assert(input_set.size() == output_set.size());
        std::cout << "Training with step-size " << step << "..." << std::endl;
        std::vector<int> ordering;
        for(int i=0; i<input_set.size(); ++i) ordering.push_back(i);
        double last_epoch_cost = std::numeric_limits<double>::infinity();
        int epoch = 0;
        while(epoch < max_epoch) {
            double epoch_cost = 0;
            for(int i=0; i<input_set.size(); ++i) {
                Jet cost = cost_function(input_set[ordering[i]], output_set[ordering[i]]);
                epoch_cost += cost.value();
                auto dp = (-step*cost.derivatives()).eval();
                for(int i=0; i<params.size(); ++i) *params[i] += dp(i);
            }
            if(!(epoch % 100)) std::cout << "    epoch " << epoch << ": " << epoch_cost << std::endl;
            if(fabs(epoch_cost-last_epoch_cost) < tol) break;
            std::random_shuffle(ordering.begin(), ordering.end());
            last_epoch_cost = epoch_cost;
            ++epoch;
        }
        std::cout << "Finished training on epoch " << epoch << "/" << max_epoch << std::endl;
    }
};

////////////////////////////////////////////////// MAIN EXECUTABLE

int main(int argc, char** argv) {
    if(argc < 3) {
        std::cout << "Please pass in a step-size and the size of each hidden layer!" << std::endl;
        return 1;
    }

    // We will make a fixed set of data within this code for now for simplicity
    using Edouble = Eigen::Matrix<double, 1, 1>;
    std::vector<Edouble> input_set;
    std::vector<Edouble> output_set;
    for(double val=-10; val<=10; val+=0.3) input_set.push_back(Edouble(val));
    for(Edouble const& input : input_set) output_set.push_back(Edouble(5*exp(-input(0)*input(0))) +
                                                               Edouble(2.5*exp(-pow(input(0)-3, 2))) +
                                                               Edouble(0.05*(randf())));

    // Construct and train our MLP from user choices
    std::vector<int> dims;
    dims.push_back(input_set[0].size());
    for(int i=2; i<argc; ++i) {
        int dim = atoi(argv[i]);
        assert(dim > 0);
        dims.push_back(dim);
    }
    dims.push_back(output_set[0].size());
    std::cout << "MLP Layout: ";
    for(int dim : dims) std::cout << dim << ' ';
    std::cout << std::endl;
    MLP mlp(dims);
    mlp.train(input_set, output_set, atof(argv[1]));

    // Evaluate fit
    std::vector<Edouble> fine_input_set;
    for(double val=-10; val<=10; val+=0.005) fine_input_set.push_back(Edouble(val));
    auto mlp_output_set = mlp.feedforward(fine_input_set);
    std::cout << "Plotting results..." << std::endl;
    pyplot::named_plot("data", extract_values(input_set), extract_values(output_set), "g--");
    pyplot::named_plot("fit", extract_values(fine_input_set), extract_values(mlp_output_set), "k");
    pyplot::legend();
    std::cout << "Close plots to finish..." << std::endl;
    pyplot::show();

    return 0;
}

////////////////////////////////////////////////// OTHER IMPLEMENTATION DETAILS

inline double randf() {
    return 2*Eigen::MatrixXd::Random(1, 1)(0) - 1;
}

inline Eigen::VectorXd reals(Jetvec const& jv) {
    Eigen::VectorXd v(jv.size());
    for(int i=0; i<v.size(); ++i) v(i) = jv(i).value();
    return v;
}

inline std::vector<double> extract_values(std::vector<Eigen::Matrix<double, 1, 1>> const& vec_eig) {
    std::vector<double> vec(vec_eig.size());
    for(int i=0; i<vec.size(); ++i) vec[i] = vec_eig[i](0);
    return vec;
}
