function [A, B, C, D] = model5(state, pre_input)
    % MODEL5 4 Wheel nonlinear model 
    %     [A, B, C, D] = model5(state, pre_input)
    %
    %     DESCRIPTION
    %         4 wheel vehicle
    %         use linearization at operation point (current state and input)
    %         full observation (C is unit matrix and D is 0 vector)
    %
    %     STATE
    %         dot x
    %         dot y
    %         dot psi
    %
    %     INPUT
    %         steering angle (delta)
    %         front tire force (Ffx)

    Cf = 80000;
    Cr = 80000;
    m = 1573;
    Iz = 2873;
    lf = 1.1;
    lr = 1.58;
    w = 1.4;

    %% operation point
    Vx = state(1);
    Vy = state(2);
    dot_psi = state(3);
    delta = pre_input(1);
    Ffxr = pre_input(2);
    Ffxl = pre_input(3);
    
    Ffx = Ffxr + Ffxl;
    delFfx = Ffxr - Ffxl;
    Ffy = 2 * Cr * (-(Vy-lr*dot_psi)/Vx);
% 
% Ffx = pre_input(2);
% delFfx = pre_input(3);

    %% element
    dFfydx_dot = 2 * Cf * (-(Vy + lf * dot_psi) / (-Vx^2));
    dFfydy_dot = 2 * Cf * (- 1/Vx);
    dFfydpsi_dot = 2 * Cf * (- lf/Vx);

    dFrydx_dot = 2 * Cr * (-(Vy - lr * dot_psi) / (-Vx^2));
    dFrydy_dot = 2 * Cr * (- 1/Vx);
    dFrydpsi_dot = 2 * Cr * (lr/Vx);

    a11 = (-dFfydx_dot * sin(delta)) / m;
    a21 = (dFfydx_dot * cos(delta)) / m - dot_psi;
    a31 = (dFfydx_dot * cos(delta) * lf - dFrydx_dot * lr) / Iz;
    
    a12 = (-dFfydy_dot * sin(delta)) / m + dot_psi;
    a22 = (dFfydy_dot * cos(delta)) / m;
    a32 = (dFfydy_dot * cos(delta) * lf - dFrydy_dot * lr) / Iz;
    
    a13 = (-dFfydpsi_dot * sin(delta)) / m + Vy;
    a23 = (dFfydpsi_dot * cos(delta)) / m - Vx;
    a33 = (dFfydpsi_dot * cos(delta) * lf - dFrydpsi_dot * lr) / Iz;

    b11 = (Ffx * -sin(delta) - Ffy * cos(delta)) / m;
    b21 = (Ffx * cos(delta) + Ffy * -sin(delta)) / m;
    b31 = ((Ffx * cos(delta) + Ffy * - sin(delta)) * lf + delFfx * -sin(delta) * w) / Iz;
    
    b12 = 1 * cos(delta) / m;
    b22 = 1 * sin(delta) / m;
    b32 = (1 * sin(delta) * lf + 1 * cos(delta) * w) / Iz;

    b13 = 1 * cos(delta) / m;
    b23 = 1 * sin(delta) / m;
    b33 = (1 * sin(delta) * lf + -1 * cos(delta) * w) / Iz;

%     b12 = cos(delta) / m;
%     b22 = sin(delta) / m;
%     b32 = (sin(delta) * lf) / Iz;
% 
%     b13 = 0;
%     b23 = 0;
%     b33 = (cos(delta) * w) / Iz;
    
    %%

    dfdx_op = [a11 a12 a13;
               a21 a22 a23;
               a31 a32 a33];

    dfdu_op = [b11 b12 b13;
               b21 b22 b23;
               b31 b32 b33];

    A = dfdx_op;
    B = dfdu_op;
    C = eye(size(A));
    D = zeros(size(B));

end
