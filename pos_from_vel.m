% Demonstration of how, for a system with kindly coupled dynamics,
% you can estimate absolute position from just velocity measurements.
% Observable is observable!

clear all;
close all;
clc

% x1dot = x2  (position velocity relationship)
% x2dot = -k*x1 -b*x2 + u  (simple linear system)
A_true = [0, 1; -2.3, -3.2];
B_true = [0; 1];
sys_true = ss(A_true, B_true, eye(2), 0)

% We'll push the system around in an interesting way
% to get some response data, the state x for T seconds
T = 200;
dt = 0.01;
t = [0 : dt : T];
u = 2 * sin(0.05*t) .* cos(t);
x = lsim(sys_true, u, t);
% plot(t, x(:,1), 'g', t, x(:,2), 'b', t, u, 'r')
% legend('position', 'velocity', 'input')

% Suppose we only have velocity readings
% and our model is imperfect
A = round(A_true);
B = B_true;
C = [0, 1];
D = 0;
sys = ss(A, B, C, D)

% Theory says it's fine
observability = rank(obsv(sys))

% Throw some noise on the velocity values from the previous sim
noise_mag = 0.03;
y = x(:,2) + noise_mag*randn(size(t))';

% Observer system
% change in xhat = predicted xhat + correction feedback
% xhatdot = A*xhat + L*(y - C*xhat)
%         = (A - LC)*xhat + L*y
observer_poles = [0.02; 220] .* eig(A);  % hand tuned
L = place(A', C', observer_poles)'
A_obs = A - L*C;
B_obs = L;
sys_obs = ss(A_obs, B_obs, eye(2), 0);

% Have it track state, with incorrect initial condition
xhat0 = [0.5, -0.5];
xhat = lsim(sys_obs, y, t, xhat0);

% Velocity filtering effect
figure;
subplot(3,1,2);
plot(t, xhat(:,2), 'k', t, x(:,2), 'g');
legend('estimated', 'true');
title('veloicty');

% Position reconstruction effect
subplot(3,1,1);
plot(t, xhat(:,1), 'k', t, x(:,1), 'g');
legend('estimated', 'true');
title('position');

% Estimation errors
subplot(3,1,3);
plot(t, x(:,1)-xhat(:,1), 'k', t, x(:,2)-xhat(:,2), 'b');
legend('position', 'velocity');
title('estimation erros');

rms_position_error = rms(x(:,1)-xhat(:,1))
rms_velocity_error = rms(x(:,2)-xhat(:,2))




