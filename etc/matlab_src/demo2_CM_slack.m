clear

%% Simulations Parameters
Ts = 0.01;
Vx_ref = 50 / 3.6;

    Cf = 3e2;
    Cr =3e2;
    m = 1644.80;
    Iz = 2488.892;
    lf = 1.240;
    lr = 1.510;
    w = 0.816;
%% Initial Conditions
initial_state = [50/3.6 0 0]';
initial_input = [0 0 0]';

%% Dynamics Parameters
Iw = 0.589;
reff = 0.33;
[A, B, C, D] = CMmodel2_2(initial_state, initial_input)

%% mpc parameters
predictionHorizon = 20;
controlHorizon = 10;
%% mpc weights
stateWeight = [1e3 1e-2 5e3];
% stateWeight = [1 1 1];
controlWeight = [1e-15 1e-20 1e-20];
% controlWeight = [0 0 0];

%% mpc constraints
% stateAbsMax = []
stateAbsMax = [
    75/3.6 5e5 5e5
];
stateRateMax = stateAbsMax .* [1 1 1];

controlAbsMax = [
    pi/3 5e2 5e2
];
controlRateMax = controlAbsMax .* [1 1 1];

stateConstraints = [
    stateAbsMax(1) stateAbsMax(1)         30/3.6  stateAbsMax(1) 1e1 1e1 1e5 1e5
    stateAbsMax(2) stateAbsMax(2) -stateAbsMax(2) stateAbsMax(2) 1e1 1e1 1e5 1e5
    stateAbsMax(3) stateAbsMax(3) -stateAbsMax(3) stateAbsMax(3) 1e1 1e1 1e5 1e5    
];
controlConstraints = [
    controlRateMax(1) +controlRateMax(1) -controlAbsMax(1) +controlAbsMax(1) inf inf inf inf 
    controlRateMax(2) +controlRateMax(2) -controlAbsMax(2) +controlAbsMax(2) inf inf inf inf
    controlRateMax(3) +controlRateMax(3) -controlAbsMax(3) +controlAbsMax(3) inf inf inf inf

];

%% PID gain
Kp = 0.35;
Ki = 0.5;
Kd = 0.000;

