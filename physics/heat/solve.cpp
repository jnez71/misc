/**************************************************
Simulation of the 1D heat equation: the diffusion of temperature u(x,t) through a solid rod.

The finite-difference method is used for spacial discretization. Stepsize dx is uniform.
Euler's method ("RK1") is used for temporal discretization. Stepsize dt is constant.
Boundary conditions are applied at the endpoints (x=0 and x=1), and are constant.
The initial condition u(x,t=0) throughout the interior is all zeros.
The material's thermal diffusivity parameter a(x,t) is uniform and constant.

Symbolically,
   d/dt(u(x,t)) = a(x,t) * d^2/dx^2(u(x,t))
| u(x,0)=0 | u(0,t)=b | u(1,t)=0 | a(x,t)=a |
with dx, dt, a, and b specified by the user, as well as the total simulation duration tfin.

If compiling with GCC on Linux, in the terminal do:
```
     g++       solve.cpp         -o            solve           -std=c++11             -O3  -ffast-math
# (program)   (input-file)  (output-flag)  (output-name)  (language-standard-flag)  (optimization-flags)
```

This results in an executable called "solve" which can be run from this directory as follows (for example):
```
   ./solve         5e-3                2e-4               5e-2                2                       5
# (program)  (spatial-stepsize)  (temporal-stepsize)  (diffusivity)  (left-boundary-value)  (simulation-duration)
```

The simulation writes to a file called "result.txt" that contains u(x,t) where x varies by
dx over the columns and t varies by dt down the rows, all comma delimited. The leftmost
column, however, is just the time at that iteration. The file looks as follows:
=========================================
  0,   u(0,0),   u(dx,0),   u(2dx,0), ...
 dt,  u(0,dt),  u(dx,dt),  u(2dx,dt), ...
2dt, u(0,2dt), u(dx,2dt), u(2dx,2dt), ...
 . ,     .   ,     .    ,      .    , ...
 . ,     .   ,     .    ,      .    , ...
 . ,     .   ,     .    ,      .    , ...
=========================================

This is a simula*tion* not a simula*tor* because it is very limited in user flexibility
and is written as a single script instead of an object-oriented library.

Also, while I generally recommend using the Eigen library for vector math, for demo
purposes I only make use of C++ standard libraries here. This is decently less efficient.
You will notice that I iterate over the temperature distribution vector in a number of
places. If Eigen had been used, the math could still be functional and concise but Eigen's
"expression template" system would under-the-hood merge all the math into one efficient loop.

Lastly, note that FDM with uniform and constant discretization is a very poor way to solve this.
Increasing dt just a little bit too much can result in numerical instability. Further, there is
obviously a lot of waste. The system evolves the fastest at the very beginning when the temperature
gradient is large, so dt should be small then, and gradually increase as the system approaches
steady-state. The spatial discretization should also be finer in the regions of more intense gradient,
like near the left boundary. Adaptive integration and meshing is interesting and useful, but mildly
challenging, and thus not used here for demo purposes. This massively hurts performance, though!

The analytical steady-state solution to this problem is well known. The temperature varies linearly
between the two boundary conditions (at x=0 and x=l). Thus we expect our simulation to converge
to that as time goes to infinity. For a=5e-2, I found that t=infinity is roughly t=5.

**************************************************/
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

//////////////////////////////////////////////////

// Let's alias some types for brevity and clearer math intent
using Real = double;
using Vector = std::vector<Real>;

/*
Unlike mathematical vectors, however, std::vector does
not have an addition operation or scaling operation.
For this reason, when those basic operations are needed,
we will iterate over the vector and do things element-wise.
This is where a library like Eigen would be really helpful.
Under-the-hood, it is still doing math element-wise (I mean,
there isn't really another option), but through clever use of
templates, it is able to consolidate the loops without compromising
code readability. This improves performance by reducing the allocation
of temporary variables. Yes I really like Eigen. But here we will suffer!
*/

//////////////////////////////////////////////////

// Define a function that writes the time 't' and vector
// 'vec' values to a given file 'stream', comma delimited
inline void record(std::ofstream & stream, Real t, Vector const& vec) {
    // Write the t value to the file
    stream << t;
    // Iterate over each value in the vector
    for(auto val : vec) {
        // Write a comma and the value to the file
        stream << "," << val;
    }
    // Finish with a new line
    stream << std::endl;
}

//////////////////////////////////////////////////

// Define a function that computes the discrete Laplacian of a finite 1D vector
// 'vec' with domain-distance between elements 'step' and scale factor 'scale'
inline Vector scaled_laplacian(Vector const& vec, Real step, Real scale) {
    // Initialize the output to a vector of the same size as the input
    int n = vec.size();
    Vector dvec(n);
    // Memoize the step squared
    Real step2 = step*step;
    // Compute second-order finite-differences over the interior, scaled
    for(int i=1; i<n-1; ++i) {
        dvec[i] = scale * (vec[i+1] - 2*vec[i] + vec[i-1]) / step2;
    }
    // The boundaries of dvec are not computed because the solver ignores them
    return dvec;
}

//////////////////////////////////////////////////

// Define a function that modifies the given 'state' vector by
// euler-integrating it with 'state_dot' times 'step'
inline void evolve(Vector & state, Vector const& state_dot, Real step) {
    for(int i=0; i<state.size(); ++i) {
        state[i] += step * state_dot[i];
    }
}

//////////////////////////////////////////////////

// This is what runs when the compiled executable is called
int main(int argc, char** argv) {

    // Verify the number of arguments supplied to the program
    // (the first argument is always implicitly the file path of the program)
    if(argc != 6) {
        std::cerr << "Please supply arguments dx, dt, a, b, and tfin." << std::endl;
        return 1;
    }

    // Parse arguments, convert to real numbers
    Real space_step = std::atof(argv[1]);
    Real time_step = std::atof(argv[2]);
    Real diffusivity = std::atof(argv[3]);
    Real boundary = std::atof(argv[4]);
    Real duration = std::atof(argv[5]);

    /*
    A nicer way to do the above would be to let the user
    write a file called "config.txt" and pass its location
    as the only argument to solve. Then solve reads from that
    file and extracts the above from a pretty table of values.
    */

    // Open (and clear) the results file
    std::ofstream result_stream;
    result_stream.open("result.txt");

    // Number of discrete space and time points
    int n_space = 1 / space_step;
    int n_time = duration / time_step;

    // Initial condition for solution
    Vector temperature(n_space);
    temperature[0] = boundary;
    for(int i=1; i<n_space; ++i) {
        temperature[i] = 0;
    }
    record(result_stream, 0, temperature);

    // Simulation loop (note that we already did t=0)
    std::cout << "Simulating..." << std::endl;
    for(int i=1; i<n_time; ++i) {
        // Compute state derivative
        Vector temperature_dot = scaled_laplacian(temperature, space_step, diffusivity);
        // Integrate
        evolve(temperature, temperature_dot, time_step);
        // Enforce boundary conditions
        temperature[0] = boundary;
        temperature[n_space-1] = 0;
        // Record current result
        record(result_stream, i*time_step, temperature);
    }
    std::cout << "Done!" << std::endl;

    // Close files and exit nominally
    result_stream.close();
    return 0;
}
