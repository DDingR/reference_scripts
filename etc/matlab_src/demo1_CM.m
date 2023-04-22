clear
load('CMdata.mat');

T = 100;
Ts = 0.01;
%% declare system
[A, B, C, D] = CMmodel();
system = ss(A,B,C,D);
%% declare mpc
predictionHorizon = 50;
controlHorizon = 20;

% %% mpc constraints and weights
% stateWeight = [15 10 300 100];
% controlWeight = [0];
%
% controlConstraints = [-pi pi -pi/6 pi/6];
% stateConstraints = zeros(4,4);
Iw = 0.589;
reff = 0.4;
Aw = [0 1 0; 0 0 -reff/Iw; 0 0 0];
Bw = [0 1/Iw 0]';
Cw = [1 0 0];
Dw = 0;

Kp = 0.01;
Ki = 0.0001;
%% mpc constraints and weights
mpcObj = mpc(system, Ts, predictionHorizon, controlHorizon);
% mpcObj.ManipulatedVariables(1).Min = -pi/6;
% mpcObj.ManipulatedVariables(1).Max = +pi/6;
% mpcObj.ManipulatedVariables(1).RateMin = -pi/6 * 0.001;
% mpcObj.ManipulatedVariables(1).RateMax = +pi/6 * 0.001;
%

mpcObj.Weights.OutputVariables = [15 10 150 100];
%     mpcObj.Weights.OutputVariables = [1 1 1 1];

%     mpcObj.Weights.OutputVariables = [25 10 200 5];
%     mpcObj.Weights.OutputVariables = [1 5 10 5];

mpcObj.Weights.ManipulatedVariables = 0;