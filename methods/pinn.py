#!/usr/bin/env python3
"""
Neural-network ansatz for the differential equation,
    D^2[u](x) = f(x)
    f(x) = -(2*pi)^2*sin(2*pi*x) - 0.1*(50*pi)^2*sin(50*pi*x)
    u(0) = u(1) = 0

The analytical solution is,
    u(x) = sin(2*pi*x) + 0.1*sin(50*pi*x)
which is notably a superposition of two very different frequencies.

Even numerically, there are much better ways to solve this (e.g. Fourier methods)
but this is a useful example for gaining insight into general PINN behavior on
problems with disparate scales.

"""
import jax
from numpy import random
from scipy import optimize
from matplotlib import pyplot

np = jax.numpy  # enables autodiff
random.seed(1)  # lock RNG for repeatability
jax.config.update('jax_enable_x64', True)  # this task is numerically sensitive

##################################################

class NeuralNetwork:
    """
    Multilayer-perceptron (dense, affine synapses with tanh neurons).

        dimensions: list of integers specifying the width of each layer

    """
    def __init__(self, dimensions):
        self.size = 0    # total number of parameters
        self.index = []  # correspondence between the parameter-vector's indices and weight / bias matrices
        # Initialize size and index based on pairs of dimensions
        for (i,j) in zip(dimensions[:-1], dimensions[1:]):
            w = slice(self.size, self.size+i*j)
            b = slice(w.stop, w.stop+j)
            self.size += b.stop - w.start
            self.index.append((w, b, (i,j)))

    def __call__(self, inputs, parameters):
        """
        Evaluates the neural-network (forward-pass).

                inputs: samples-by-features array of input data
            parameters: vector of dimension self.size containing weights & biases (according to self.index)

        """
        # Iterate through layers
        for (w, b, s) in self.index:
            # Unpack parameters
            weights = parameters[w].reshape(s)
            biases = parameters[b]
            # Layer operations
            outputs = np.dot(inputs, weights) + biases  # left-multiplication simplifies batch processing
            inputs = np.tanh(outputs)
        return outputs  # no neurons on last layer

##################################################

class Ansatz:
    """
    Optimizes an approximation of the solution to the differential equation,
    D^2[u](x) = f(x)  with  u(0) = u(1) = 0

        form: callable u(x;parameters)
        parameters: vector of dimension form.size (defaults to randomly sampled from standard normal)

    """
    def __init__(self, form, parameters='random'):
        self.form = form
        self.parameters = parameters if parameters is not 'random' else random.normal(size=self.form.size)
        self.laplacian = Ansatz._meshwise(jax.jacfwd(jax.jacrev(self.form)))
        self.boundaries = np.array([[0.0], [1.0]])

    def __call__(self, x):
        """
        Evaluates the ansatz u(x) using the current parameters.

            x: scalar or n-by-1 array of domain points to evaluate at

        """
        return self.form(x, self.parameters)

    def collocate(self, mesh, forces, maxiter=10000):
        """
        Optimizes the parameters so that the ansatz satisfies the differential equation (in the least-squares sense).

            mesh: n-by-1 array of domain points to collocate at
            forces: n-by-1 array of the values of f(x) at each mesh point
            maxiter: maximum number of optimization iterations to run

        """
        # Functions for mean-square-error evaluation on the domain and boundary
        residuals  = lambda parameters: np.sum(np.square(forces - self.laplacian(mesh, parameters))) / len(mesh)**2
        conditions = lambda parameters: np.sum(np.square(self.form(self.boundaries, parameters))) / len(self.boundaries)**2
        # Run optimization
        print("====\nAnsatz: beginning collocation")
        self.parameters = optimize.minimize(
            jax.jit(jax.value_and_grad(residuals)),
            self.parameters,
            jac=True,
            constraints={'type': 'eq', 'fun': jax.jit(conditions), 'jac': jax.jit(jax.grad(conditions))},  # = 0
            method='SLSQP',
            options={'maxiter': maxiter, 'disp': True},
        ).x
        print("Ansatz: collocation finished\n====")

    def _meshwise(field):
        # Tell Jax that the many outputs of the function are just an array of element-wise evaluations
        field = jax.vmap(field, in_axes=(0, None))
        return lambda *args, **kwargs: field(*args, **kwargs).reshape(-1,1)

##################################################

# Main script if run directly
if __name__ == "__main__":

    # Disparate frequencies to learn
    freq_lo =  2.0*np.pi
    freq_hi = 50.0*np.pi

    # Forcing function and analytical solution
    force = lambda x: -(freq_lo**2)*np.sin(freq_lo*x) - (freq_hi**2)*np.sin(freq_hi*x)/10.0
    exact = lambda x:               np.sin(freq_lo*x) +              np.sin(freq_hi*x)/10.0

    # Domain meshes to train and test on
    train = np.arange(0.0, 1.0, 0.001).reshape(-1,1)
    test  = np.arange(0.0, 1.0, 0.001).reshape(-1,1)

    # "Physics-informed" neural-network construction and training
    pinn = Ansatz(form=NeuralNetwork([1, 64, 1]))
    pinn.collocate(mesh=train, forces=force(train))

    # Display results on test mesh
    pyplot.plot(test, exact(test), color='k', label='exact')
    pyplot.plot(test,  pinn(test), color='m', label='pinn')
    pyplot.legend()
    pyplot.show()

##################################################
