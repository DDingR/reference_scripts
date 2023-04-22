clear

%% Simulations Parameters
Ts = 0.01;
Vx_ref = 50 / 3.6;

    Cf = 8e4;
    Cr = 8e4;
    m = 1644.80;
    Iz = 2488.892;
    lf = 1.240;
    lr = 1.510;
    w = 0.816;
%% Initial Conditions
initial_state = [50/3.6 0.0000 0.000000]';
initial_input = [0.000001 0.0000001 0.0000001]';

%% Dynamics Parameters
Iw = 0.589;
reff = 0.4;
[A, B, C, D] = CMmodel3(initial_state, initial_input)

%% mpc parameters
predictionHorizon = 20;
controlHorizon = 10;
%% mpc weights
stateWeight = [1e0 1e4 1e5];
controlWeight = [1e-5  1e-15 1e-15];

%% mpc constraints
% stateAbsMax = []
stateAbsMax = [
    70/3.6 10 10
];
controlAbsMax = [
   pi/10 0.7 0.7
];

stateConstraints = [
    -stateAbsMax(1)*0.1 +stateAbsMax(1)*0.1             30/3.6  stateAbsMax(1)
    -stateAbsMax(2)*0.1 +stateAbsMax(2)*0.1 -stateAbsMax(2) stateAbsMax(2)
    -stateAbsMax(3)*0.1 +stateAbsMax(3)*0.1 -stateAbsMax(3) stateAbsMax(3)    
];
controlConstraints = [
    -controlAbsMax(1)*0.1 +controlAbsMax(1)*0.1  -controlAbsMax(1) +controlAbsMax(1)
    -controlAbsMax(2)*0.1 +controlAbsMax(2)*0.1  -controlAbsMax(2) +controlAbsMax(2)
    -controlAbsMax(3)*0.1 +controlAbsMax(3)*0.1  -controlAbsMax(3) +controlAbsMax(3)
    
    ];
%% PID gain
Kp = 10;
Ki = 14;
Kd = 0;

