"""
Fun with the Schrodinger equation!

A solution to the 2D Schrodinger equation is evolved in time given some
initial condition and a potential function. The hamiltonian used is that of
the quantum particle with dissipation, H = alph*del^2 + V(r,t) + j*hbar*B(r,t).

The wave function solution F is computed on an Lx by Ly grid and time is
discretized to a step of size dt. F is stored as a (2, Lx, Ly)-dimensional array,
where F[0] contains the real part of the solution and F[1] the imaginary part.
The visualization maps phase angle to hue and relative magnitude to value.

A note about time integration: the simple Euler-step scheme has been commented out
in favor of a semi-analytical scheme that takes advantage of how much slower the
magnitude of F varies compared to its phase. Thanks Forrest.

Plan for the future: allow user to interactively draw the potential field in realtime.

"""
# Dependencies
from __future__ import division
import numpy as np; npl = np.linalg
import time
import cv2

# Computational tools
normalize = lambda F: F/npl.norm(F)
claplacian = lambda F: np.array((cv2.Laplacian(F[0], cv2.CV_64F), cv2.Laplacian(F[1], cv2.CV_64F)))
cexp = lambda M: (lambda eM: np.array((eM.real, eM.imag)))(np.exp(M))
cmult = lambda a, b: np.array((a[0]*b[0] - a[1]*b[1], a[0]*b[1]+a[1]*b[0]))
over_j = lambda F: np.array((F[1], -F[0]))
amp = lambda F: np.sum(F**2, axis=0)

# Physical constants
hbar = 1
mass = 0.1
alph = -(hbar**2)/(2*mass)

# Discrete solution space
dt = 0.001
Lx, Ly = 110, 110
imshow_scale = 3
fps = 30

# Initial wave function, Gaussian packet with momentum
F = np.zeros((2, Lx, Ly), dtype=np.float64)
for x in xrange(Lx):
    for y in xrange(Ly):
        gauss = np.exp(-0.5*np.sqrt(5*((x-Lx/2)**2+(y-Ly/6)**2) - 100j*y))
        F[:, x, y] = np.real(gauss), np.imag(gauss)
F = normalize(F)

# Potential function, double slit anyone?
V = np.zeros((Lx, Ly), dtype=np.float64)
vl_a, vl_b, vl_c = int(Ly/3), int(Ly/3)+2, int(Lx/2)
V[:vl_c-7, vl_a:vl_b], V[vl_c-3:vl_c+4, vl_a:vl_b], V[vl_c+8:, vl_a:vl_b] = [50]*3
V[:, -5:] = 100

# Dissipation function on boundaries to effectively "open the box"
B = np.zeros((Lx, Ly), dtype=np.float64)
bthick = 10
for i in xrange(bthick):
    s = 50*(i/bthick)**3
    B[bthick-1-i, :] = s
    B[-(bthick-1-i)-1, :] = s
    B[:, bthick-1-i] = s
    B[:, -(bthick-1-i)-1] = s

# Visualization tools
imshow_size = tuple(np.multiply(imshow_scale, (Ly, Lx)))
def cdisp(name, F, V, auto=True):
    mag = npl.norm(F, axis=0)
    ang = np.degrees(np.arctan2(F[1], F[0]) + np.pi)
    if auto: val = 255*mag/np.max(mag)
    else: val = 10000*mag
    img = cv2.cvtColor(np.uint8(np.dstack((ang/2, 255*np.ones_like(mag), val))), cv2.COLOR_HSV2BGR)
    img = img + cv2.cvtColor(np.uint8((255/np.max(V))*V), cv2.COLOR_GRAY2BGR)
    img = cv2.resize(np.clip(img, 0, 255), imshow_size)
    cv2.imshow(name, img)
    return img

# Recording tools
record = False
if record: recorder = cv2.VideoWriter('quantum.avi', cv2.cv.CV_FOURCC(*'XVID'), fps, imshow_size)

# Simulation
last_draw = 0
speed_flag = False
print("Press esc to quit.\n")
while True:
    start = time.time()

    # Draw frame
    if time.time() - last_draw > 1/fps:
        img = cdisp("Solution", F, V)
        key = cv2.waitKey(1)
        if key == 27: break
        if record: recorder.write(img)
        last_draw = time.time()

    # Increment wave function
    # dFdt = over_j((alph*claplacian(F) + V*F)/hbar) - B*F
    # F = F + dFdt*dt
    F[:] = cmult(cexp((-1j*dt/hbar)*V-dt*B), F - (hbar*dt/(2*mass))*over_j(claplacian(F)))
    F[:] = normalize(F)

    # Regulate sim speed
    elapsed = time.time() - start
    if elapsed < dt:
        if not speed_flag:
            print("Nice computer!")
            print("You can run {} times faster than realtime.".format(np.round(dt/elapsed, 4)))
            print("Try using that excess performance by simulating over a larger grid.\n")
            speed_flag = True
        time.sleep(dt - elapsed)
