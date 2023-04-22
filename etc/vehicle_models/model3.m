function [A, B, C, D] = model3(state, pre_input)
    % MODEL3 simple nonlinear model
    %     [A, B, C, D] = model3(state, pre_input)
    %
    %     DESCRIPTION
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

    %% operation point
    Vx = state(1);
    Vy = state(2);
    dot_psi = state(3);
    delta = pre_input(1);
    Ffx = pre_input(2);

    %% element
    a11 =- Cf * sin(delta) / m * (Vy + lf * dot_psi) / Vx ^ 2;
    a21 = Cf * cos(delta) / m * (Vy + lf * dot_psi) / Vx ^ 2 + Cr / m * (Vy - lr * dot_psi) / Vx ^ 2 - dot_psi;
    a31 = 1 / Iz * (Cf * (Vy + lf * dot_psi) / Vx ^ 2 * cos(delta) * lf + Cr * (Vy - lr * dot_psi) / Vx ^ 2 * lr);

    a12 =- Cf * sin(delta) / m / (-Vx) + dot_psi;
    a22 = Cf * cos(delta) / m / (-Vx) + Cr / m / (-Vx);
    a32 = 1 / Iz * (Cf / (-Vx) * cos(delta) * lf + Cr / (-Vx) * lr);

    a13 =- Cf * sin(delta) / m * (- lf / Vx) + Vy;
    a23 = Cf * cos(delta) / m * (- lf / Vx) + Cr / m * (lr / Vx) - Vx;
    a33 = 1 / Iz * (Cf / (- lf / Vx) * cos(delta) * lf + Cr * (lr / Vx) * lr);

    b11 = Ffx / m * (-sin(delta)) - Cf * sin(delta) / m - Cf * (delta - (Vy + lf * dot_psi) / Vx) * cos(delta) / m;
    b21 = Ffx / m * cos(delta) + Cf * cos(delta) / m + Cf * (delta - (Vy + lf * dot_psi) / Vx) * (-sin(delta) / m);
    b31 = 1 / Iz * (Ffx * cos(delta) + Cf * cos(delta) + Cf * (delta - (Vy + lf * dot_psi) / Vx) * (-sin(delta))) * lf;

    b12 = cos(delta) / m;
    b22 = sin(delta) / m;
    b32 = sin(delta) * lf / Iz;
    %%

    dfdx_op = [a11 a12 a13;
               a21 a22 a23;
               a31 a32 a33];

    dfdu_op = [b11 b12;
               b21 b22;
               b31 b32];

    A = dfdx_op;
    B = dfdu_op;
    C = eye(size(A));
    D = zeros(size(B));

end
