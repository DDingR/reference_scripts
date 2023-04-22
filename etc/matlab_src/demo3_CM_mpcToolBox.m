clear 

%% Simulations Parameters
Ts = 0.01;
Vx_ref = 30 / 3.6;
%% Initial Conditions
initial_state = [0.01/3.6 0.0000 0.000000]';
initial_input = [0.000001 0.0000001 0.0000001]';
%% Dynamics Parameters
Iw = 0.589;
reff = 0.4;
load('CMdata.mat');
[A, B, C, D] = CMmodel3(initial_state, initial_input)

sys = ss(A,B,C,D);
sys = c2d(sys,Ts);
%% declare mpc
mpcObj = mpc(sys);

mpcObj.PredictionHorizon = 20;
mpcObj.ControlHorizon = 10;

%% mpc constraints and weights b
% 
% mpcObj.ManipulatedVariables(1).Min = -pi/6;
% mpcObj.ManipulatedVariables(1).Max = +pi/6;

% mpcObj.ManipulatedVariables(2).Min = 0.5;
% mpcObj.ManipulatedVariables(2).Max = +1;
% mpcObj.ManipulatedVariables(3).Min = 0.5;
% mpcObj.ManipulatedVariables(3).Max = +1;

% mpcObj.Weights.OutputVariables = [100 1 1]; % 
% mpcObj.Weights.ManipulatedVariables = [1 1];

%% PID gain
Kp = 10;
Ki = 5000;
Kd = 0;



