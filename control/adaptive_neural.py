#!/usr/bin/env python3
"""
Lets see just how far we can push adaptive neural-network (ANN) control
by using it without any traditional feedback term.

Consider a time-invariant additive-control dynamical system:
    dx/dt = f(x) + u
with the objective of state x(t) tracking reference r(t).

Assume that the "natural" dynamic f(x) is unknown.
Thus, control designs cannot have an explicit -f(x) term to cancel nature.

Instead, design feedback u := n(x;p) as a neural-network parameterized by p.
Using a neural-network specifically as the function-approximator is because
nothing is known about f. Otherwise, architect n to reflect known parts of f.

Furthermore, since the system is Markovian, letting the input to n just be x
is sufficient. Ideally, a deep architecture will transform x into more useful
features, but the best possible features are guaranteed to only depend on x
because the theoretical optimal value function depends only on x.

Define the tracking-error e := r - x.
Consider the error-"energy" J := (1/2)*e'*G*e for some gains G > 0.

The error-energy is always nonnegative, and has a dynamic,
    dJ/dt = e'*G*(de/dt)
          = e'*G*(dr/dt) - e'*G*(f(x) + u)
          = e'*G*(dr/dt) - e'*G*f(x) - e'*G*n(x;p)

If the error-energy's rate-of-change were always negative, the error itself
must go to zero. (This is a non-rigorous Lyapunov argument). Thus, we want to
choose control parameters then that minimize dJ/dt. Looking at the variation,
    d/dp(dJ/dt) = -e'*G*(dn/dp(x;p))

We can then design a gradient-descent adaptation mechanism to move the
parameters however most decreases the instantaneous error-energy.
    dp/dt := -d/dp(dJ/dt)'
           = (dn/dp(x;p))'*G*e
This mechanism cannot guarantee dJ/dt will be or stay negative (i.e. stability)
but it is at least trying in the right direction.

From the perspective of typical ML training, the final design is that our
neural-network predicts actions based on states (a policy), and its parameters
are updated to minimize a reinforcement cost c online as the state is compared
to the reference at every timestep.
    c(p) := -e'*G*n(x,p)
    p(t+dt) = p(t) - dt*(dc/dp(p))'

This is clearly a myopic strategy, because all of the parameters are moved based
on just the instantaneous tracking-error. I.e. as this controller operates in one
region of the error space, it will overfit to those samples and "forget" anything
that it learned elsewhere. It is "reactive" rather than "predictive."

One can also see this controller as a special case of a policy-gradient RL method
with a horizon of 1 step. The clear way to improve its ability to "plan" rather than
just "react" is to let J sum a sequence of error energies ("rollouts") rather than
just the current one.
    C(p) := sum_i [-e_i'*G*n(x_i,p)]
Then the parameters will converge to something that works reasonably well over all
these samples (assuming sufficient capacity) and thus generalize better to plans
in the regions of the error space the samples are from.

But still, suppose we don't need planning. Suppose this is for a low-level control
problem, like a regulator, but the natural dynamic is just very difficult. For example,
controlling the attitude of a fixed-wing aircraft: the external torque due to aerodynamic
effects is a complicated nonlinear function of the airspeed and center-of-pressure, the
latter of which is itself a very complicated function of attitude. The aircraft wants to
use its actuators to torque the body into the correct orientation, needing just feedback,
no planning, but it is very hard to compensate away the effects of the air's external
torque on the body. For micro-UAVs with small stability margins, this is a big problem,
and wind-tunnel tests for building f look-up tables are costly. How well will just a
myopic neural-controller work?

Final design:
    SYSTEM: dx/dt = f(x) + u           # f unknown, x known, u chosen
    POLICY:     u = n(x;p)             # some learning architecture
     ADAPT: dp/dt = (dn/dp(x;p))'*G*e  # for vectors, e = r - x

Conveniently, this directly applies to Lie group state spaces if e is computed appropriately.

We'll use automatic differentiation for computing dn/dp so that we can quickly swap in any
(neural or not) architecture for our policy function. We'll allow the specification of any
nature f, and compare the result of the neural controller to that of an ordinary state
feedback law with integrator ("PID"), K*e + k*S[e*dt].

"""
from autograd import numpy as np, grad  # pip3 install --user autograd
from matplotlib import pyplot  # pip3 install --user matplotlib

# Alert user
print("Setting up...")

##################################################

class ANN:
    """
    Adaptive neural-network feedback controller.
    Transforms the system state into an action decision.
    Learns to minimize the instantaneous error-energy rate-of-change.

    """
    def __init__(self, gains, dimensions, stdev0):
        """
        gains: adaptation-rate gains, one for each dimension of the state space
        dimensions: the neural architecture as a list of layer widths (first element
                    is state dimensionality and last element is the action dimensionality)
        stdev0: standard deviation of the initial parameters guess (set this to the
                rough order-of-magnitude of what you'd consider a very small action)

        """
        # Store the given gains
        self.gains = np.array(gains, dtype=float)
        # Store dimensions and initialize the parameters randomly
        self.dimensions = tuple(dimensions)
        self.parameters = [(np.random.normal(0.0, stdev0, size=(outdim, indim)),  # weights
                            np.random.normal(0.0, stdev0, size=outdim))  # biases
                           for indim, outdim in zip(dimensions[:-1], dimensions[1:])]
        # Verify that all dimensions are consistent
        assert len(self.dimensions) >= 2
        assert self.gains.shape == (self.dimensions[0],)
        # Prepare autodiff for the gradient of our objective function
        self._dvdp = grad(self._v)

    def control(self, state):
        """
        Returns the action u = n(x;p) for the given state x and current parameters p.

        """
        return self._n(self.parameters, state)

    def adapt(self, state, error, timestep):
        """
        Updates the parameters p with dp/dt = (dn/dp(x;p))'*G*e for the given
        state x, error e, timestep dt, and stored gains G.

        """
        gradients = self._dvdp(self.parameters, state, error)
        for (W, b), (dWdt, dbdt) in zip(self.parameters, gradients):
            W += timestep*dWdt
            b += timestep*dbdt

    def _n(self, parameters, state):
        """
        Runs the neural-network forward.

        """
        for weight, bias in parameters:
            action = np.dot(weight, state) + bias
            state = np.tanh(action)
        return action

    def _v(self, parameters, state, error):
        """
        Computes the part of the error-energy rate-of-change that actually
        depends on the parameters.

        """
        return np.dot(self.gains*error, self._n(parameters, state))

##################################################

def nature(state):
    """
    Arbitrary unknown natural dynamic f(x) mapping state to state rate-of-change
    in the absence of control effort.

    """
    # return [1.0, 2.0, 3.0] - state  # simple linear
    return np.array([         10.0*(state[1]-state[0]),
                     state[0]*(28.0-state[2])-state[1],
                        state[0]*state[1]-2.6*state[2]]) - 5.0  # wacky lorenz

##################################################

# Time domain
timestep = 0.005
times = np.arange(0.0, 5.0, timestep)

# Initialize error, reference, and state records
s = 3  # state dimension
errors = np.zeros((len(times), s), dtype=float)
references = np.zeros((len(times), s), dtype=float)
states = np.zeros((len(times), s), dtype=float)
states[0] = [5.0]*s  # initial condition

# Initialize action record
a = s  # action dimension
actions = np.zeros((len(times), a), dtype=float)

##################################################

# Build controller
gains = [5.0]*s
dimensions = [s, a]
ann = ANN(gains, dimensions, stdev0=1e-6)

def reference(time):
    """
    Computes the desired state at the given time.

    """
    return np.zeros(s, dtype=float)  # simple regulation objective
    # return 5.0*np.sin(2.0*time)*np.ones(s, dtype=float)  # track a nice sinusoid

def error(state, desire):
    """
    Computes the instantaneous reference tracking-error vector.

    """
    return desire - state  # vectors can just be subtracted

##################################################

def PID(error):
    """
    Computes an ordinary linear full-state feedback action with error integral.

    """
    feedback = gains * error
    PID.integral += feedback*timestep
    return feedback + PID.integral
setattr(PID, "integral", 0.0)

##################################################

# Specify a controller to try
controller = "ann"  # "ann", "pid", or "none"

# Alert user
print("Running simulation with controller '"+controller+"'...")

# Run the simulation using the specified controller
for i, time in enumerate(times[1:]):
    # Assess error
    references[i] = reference(time)
    errors[i] = error(states[i], references[i])
    # Select action
    if controller == "ann":
        actions[i] = ann.control(states[i])
        ann.adapt(states[i], errors[i], timestep)
    elif controller == "pid":
        actions[i] = PID(errors[i])
    else:
        actions[i] = 0.0
    # Integrate dynamics
    rate = nature(states[i]) + actions[i]
    states[i+1] = states[i] + rate*timestep

##################################################

# Alert user
print("Plotting results...")

# Plot results
figure = pyplot.figure()
axis = None
for i in range(s):
    axis = figure.add_subplot(s, 1, i+1, sharex=axis)
    axis.plot(times, states[:, i], color='b', linewidth=2, label="state")
    axis.plot(times, references[:, i], color='g', linewidth=3, linestyle=':', label="desire")
    axis.plot(times[:-1], actions[:-1, i], color='r', linewidth=0.5, label="action", scaley=False)
    axis.set_xlim([times[0], times[-1]])
    axis.set_ylabel("state "+str(i), fontsize=20)
    axis.grid(True)
axis.set_xlabel("time", fontsize=20)
axis.legend()
pyplot.show()
