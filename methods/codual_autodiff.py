#!/usr/bin/env python3
"""
Calculus, on the call-stack!
This is a functional programming (specifically, continuation-passing style (CPS)) implementation of reverse-mode automatic differentiation (RAD).
See: https://papers.nips.cc/paper/8221-backpropagation-with-callbacks-foundations-for-efficient-and-expressive-differentiable-programming.pdf
Apparently, it's very efficient? That probably depends on the programming language... But one thing is certain - the implementation is neat!

In forward-mode automatic differentiation (FAD), real variables are superimposed with a "dual" part that encodes their derivatives (sensitivities) to
the program's independent variables (those just declared by the nullary "creation operator"). The underlying theory is simply Taylor Series truncation.
See: https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers
This is easily implemented via operator overloading (as shown in the Wiki article) because every "parent" creating a "child" through some operator
can assign their child's sensitivity as the product of their own sensitivity and the partial derivative of the operator. For example,

def fad_sin(DualVar parent):
    DualVar child
    child.value = sin(parent.value)
    child.sensitivity = cos(parent.value) * parent.sensitivity
    #   dc/dx         =       dc/dp       *       dp/dx
    return child

Elegant! However, propagating DualVars through your program means that you will compute the sensitivity of every variable with respect to the
independents (x). We often don't care about every variable's sensitivities, but instead only care about one specific variable's sensitivities
(usually the final program output). In that case, it would make more sense for every variable to not store its sensitivities, but rather its
"influence" on that special final variable. I.e., instead of using dual variables defined as:

var.value = c
var.sensitivity = dc/dx

we should instead use "codual" variables defined as:

var.value = c
var.influence = df/dc

The word "codual" is made-up but more clear in my opinion: for FAD, sensitivity is a vector (a "column" of a certain Jacobian) while for RAD,
influence is a COvector (linear functional, a "row" of that same Jacobian) hence the term "codual" part. The accepted terminology is that the
influence part is called the "adjoint state" of c. The program output is f and every "codual" variable stores its value (state) and its influence
on f (adjoint state). The underlying theory is the categorical-dual (https://en.wikipedia.org/wiki/Dual_(category_theory)) to FAD. (Same math as
is used to construct the "dual problem" in constrained optimization - "dual" is a hella abstract term).

Now usually when explaining RAD, people jump straight to how a program can be represented by a directed acyclic graph on which we can do all
possible analyses - NAY! While a cool way to think of things, it is not the simplest for RAD. Instead of building some auxiliary operation graph
data structure, lets just stick with codual variables and see what we can do with operator overloading.

Upon creation, any codual variable can know for sure that its influence on f is 0. That's because it was just created! There is no way for f
to be affected by it. Until... it has a child. When a parent codual variable has a child, there is a possibility that that child may someday
affect f, so the parent must update its own influence on f by adding the product of its child's dependence on it and f's dependence on its child.

def rad_sin(CodualVar parent):
    CodualVar child
    child.value = sin(parent.value)
    child.influence = ?
    parent.influence += child.influence * cos(parent.value)
    #   df/dp        +=     df/dc       *      dc/dp

Every time a parent is involved in an operation, it will add a term to its influence that is due to the potential influence the resulting
child may have on f. But during the creation of the child, how the heck can it possibly know what influence its child will have on f?
Here's where functional programming comes in. We will defer that computation with a "continuation callback." Simply put, the answer is that
the future knows what that child will do, so we'll give flow back to the program to proceed into the future, and once it knows the child's
influence, it will return that to the parent so the parent can finish updating its own influence.

Every RAD operator thus requires a continuation callback to be passed in - a function that takes in a child and returns to you its influence.
Since it "peers into the future" lets call it an "oracle."

def rad_sin(CodualVar parent, Function oracle):
    CodualVar child
    child.value = sin(parent.value)
    child.influence = oracle(child)
    parent.influence += child.influence * cos(parent.value)
    #   df/dp        +=     df/dc       *      dc/dp

Here's the trick: every RAD operator can also act as an oracle if it simply returns its own influence when finished. Why? Because with the logic
above, at the end of rad_sin, parent has determined its own influence! Even parent is the child of some grandparent, so to that grandparent,
the oracle for parent was simply rad_sin. I.e., rad_sin is the function that takes in a CodualVar (e.g. parent) and returns (to grandparent)
its correct influence. To make this work, we simply put `return parent.influence` as the last line of rad_sin.

Eventually, we'll have to hit some operator that doesn't need an oracle. That operator is the "final" operator that produces
the f we want the influences for. We have by definition that f.influence = df/df = 1, so final doesn't need an oracle. Then the whole
call-stack of returns will occur (this is backpropagation) and everyone's influences will be evaluated. Unlike PyTorch and Tensorflow
which backpropagate through a "tape" recording of operations, here the call-stack itself acts as our tape.

To make things neater, we'll make functions like rad_sin into methods (like CodualVar.sin) and make the following abbreviations,
Codual := CodualVar
   val := value
   adj := adjoint == influence
   con := continuation == oracle
   fin := final == f
  self := parent
 other := other_parent (for binary operators)
   dep := dependent == child

Here it is!

"""
import numpy as np

##################################################

class Codual:
    """
    Defines a "codual" variable, which is a real variable (state) superimposed
    with an adjoint state that is its influence on some "final" variable.

    """
    def __init__(self, val):
        self.val = np.float64(val)  # const
        self.adj = np.float64(0)

    def __str__(self):
        """
        Handler for Python print().

        """
        return "Value: {:.3f} | Influence: {:.3f}".format(self.val, self.adj)

    def fin(self, res):
        """
        Special "final" operator which simply says "I am the final variable."
        Self gets copied into the provided "result" Codual for external use.

        """
        self.adj = 1.0
        res.val = self.val
        res.adj = self.adj
        return self

    def tanh(self, con):
        dep = con(Codual(np.tanh(self.val)))     # make a child (dep) with child.val=tanh(parent.val), then call con to fill in its adj
        self.adj += dep.adj * (1.0 - dep.val**2) # now that my child's influence on f is known, update my own influence on f
        return self                              # I am someone else's child and tanh was their con, so return my fully determined self to them

    def plus(self, other, con):                  # same logic as tanh but with two parents
        dep = con(Codual(self.val + other.val))
        self.adj += dep.adj
        other.adj += dep.adj
        return self

    def times(self, other, con):
        dep = con(Codual(self.val * other.val))
        self.adj += dep.adj * other.val
        other.adj += dep.adj * self.val
        return self

##################################################

# Test
if __name__ == "__main__":

    # Input states
    x = Codual(0.5)
    y = Codual(0.4)

    # Place to store the result
    r = Codual(None)

    # Define the program dynamic
    program = lambda v1, v2: v1.plus(v2,   # v3 = v1 + v2
              lambda     v3: v3.times(v2,  # v4 = v3 * v2
              lambda     v4: v4.tanh(      # v5 = tanh(v4)
              lambda     v5: v5.fin(r))))  #  f = v5, store in r

    # Anticipated Answers
    # -------------------
    # f(x,y) = tanh(y*(x + y))             = 0.345 == r.val
    #  df/dx = y*sech^2(y*(x + y))         = 0.352 == x.adj
    #  df/dy = (x + 2*y)*sech^2(y*(x + y)) = 1.145 == y.adj

    # Run the program
    program(x, y)

    # Recover the result value, and the linearized influences that the inputs have on it
    print(" Result |", r)
    print("Input x |", x)
    print("Input y |", y)

    # The big question I am left with is how to make the definition of program
    # less weird to write for a typical user. The comments on those lines
    # depict how the lines would look in the FAD case, where the fact any
    # gradient business going on is completely hidden. For Codual RAD however,
    # despite the elegant FAD-looking Codual class, the need to pass continuation
    # callbacks makes defining the program very, lambda-tious. Functional, and yet
    # I still desire object oriented. Surely there is some abstraction or syntax
    # to make the program definition look cleaner? Regardless, RAD without an
    # auxiliary graph data structure is pretty rad, and the true relation between
    # FAD and RAD is in my opinion much more clear with the codual implementation.

##################################################
