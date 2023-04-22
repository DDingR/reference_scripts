function [A,B,C,D] = model2(state, pre_input)
% MODEL2 very simple model with steering angle
%     [A, B, C, D] = model2(state, pre_input)
%     
%     DESCRIPTION
%         use linearization at operation point (current state and input)
%         full observation (C is unit matrix and D is 0 vector)
% 
%     STATE
%         y
%         dot y
%         psi
%         dot psi
%     
%     INPUT
%         steering angle (delta)

Cf  = 80000;
Cr = 80000;
m = 1573;
Iz = 2873;
lf = 1.1;
lr = 1.58;
Vx = 30;

a22 = -1 * (2 * Cf + 2 * Cr) / (m * Vx);
a24 = -Vx + (-2 * Cf * lf + 2 * Cr * lr) / (m * Vx);
a42 = -1 * (2 * Cf * lf - 2 * Cr * lr) / (Iz * Vx);
a44 = -1 * (2 * Cf * lf^2 + 2 * Cr * lr^2) / (Iz * Vx);

b12 = 2 * Cf / m;
b14 = 2 * Cf * lf / Iz;
%%

% f = f_op + dfdx_op * (x-x0) + dfdu_op * (u-u0)
% \delta f = dfdx_op * \delta x + dfdu_op * \delta u
f_op = [state(1);
        a22 * state(2) + a24 * state(4) + b12 * sin(pre_input);
        state(3);
        a42 * state(2) + a44 * state(4) + b14 * sin(pre_input)];

dfdx_op = [0   1   0   0;
        0   a22 0   a24;
        0   0   0   1;
        0   a42 0   a44];
dfdu_op = [0;
           b12 * cos(pre_input);
           0;
           b14 * cos(pre_input)
           ];


A = dfdx_op;
B = dfdu_op;
C = eye(4);
D = zeros(4,1);

end
