function [A, B, C, D] = model4(state, pre_input)
    % MODEL4 simple nonlinear model with out Fy
    %     [A, B, C, D] = model4(state, pre_input)
    %
    %     DESCRIPTION
    %         set the Fy 0
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
    a11 = 0;
    a21 = -dot_psi;
    a31 = 0;
    a12 = dot_psi;
    a22 = 0;
    a32 = 0;
    a13 = Vy;
    a23 = -Vx;
    a33 = 0;

    b11 = Ffx/m*(-sin(delta));
    b21 = Ffx/m*(cos(delta));
    b31 = Ffx/Iz*cos(delta)*lf;
    b12 = cos(delta)/m;
    b22 = sin(delta)/m;
    b32 = sin(delta)/Iz*lf;
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
