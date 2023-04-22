clear

%% Simulations Parameters
Ts = 0.01;
Vx_ref = 60 / 3.6;

    Cf = 435.418/0.296296;
    Cr = 756.349/(0.6*pi/180);
    m = 1644.80;
    Iz = 2488.892;
    lf = 1.240;
    lr = 1.510;
    w = 0.8;
%% Initial Conditions
initial_state = [50/3.6 0 0]';
initial_input = [0 0 0]';

%% Dynamics Parameters
Iw = 0.589;
reff = 0.321;
[A, B, C, D] = CMmodel2_2(initial_state, initial_input)

%% mpc parameters
predictionHorizon = 20;
controlHorizon = 1;
%% mpc weights
% stateWeight = [1e10 1e5 1.5e10];
stateWeight = [1 1 5];
% controlWeight = [1e-15 1e-10 1e-10];
% controlWeight = [5e3 1e-3 1e-3];
controlWeight = [1e-7 1e-10 1e-10];

%% mpc constraints
% stateAbsMax = []
stateAbsMax = [
    75/3.6 5e5 5e5
];
stateRateMax = stateAbsMax .* [1 1 1];

controlAbsMax = [
    30*pi/180 15e2 15e2
];
controlRateMax = controlAbsMax; %.* [1 0.7 0.7];

stateConstraints = [
    stateAbsMax(1) stateAbsMax(1)         30/3.6  stateAbsMax(1)
    stateAbsMax(2) stateAbsMax(2) -stateAbsMax(2) stateAbsMax(2)
    stateAbsMax(3) stateAbsMax(3) -stateAbsMax(3) stateAbsMax(3)    
];
controlConstraints = [
    -controlRateMax(1) +controlRateMax(1) -controlAbsMax(1) +controlAbsMax(1)
    -controlRateMax(2) +controlRateMax(2) -controlAbsMax(2) +controlAbsMax(2)
    -controlRateMax(3) +controlRateMax(3) -controlAbsMax(3) +controlAbsMax(3)
];

%% PID gain
Kp = 0.000000005;
Ki = 5;
Kd = 0.0;

