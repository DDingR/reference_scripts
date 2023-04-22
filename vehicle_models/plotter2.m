function plotter2(logsout)
% PLOTTER2 Plot simulink data for model3
%   [] = plotter2(logsout)
% 
%   DESCRIPTION
%       plot simulation result of model3
% 
%   INPUT DATA  
%       XY [2,1]
%           global x coordinate
%           global y coordinate  
%       XY_ref [2,1]
%           global reference x coordinate
%           global reference y coordinate
%       state [3,1]
%           body-fixed dot x
%           body-fixed dot y
%           body-fixed dot psi    
%       state_ref [3,1]
%           reference body-fixed dot x
%           reference body-fixed dot y
%           reference body-fixed dot psi    
%       input [2,1]
%           steering angle (delta)
%           front tire force (Ffx)
%   
%   PLOT FIGURES
%       abs pos
%            position on gloabl coordinates
%       states
%       inputs

%%
XY = logsout.getElement('XY');
XY_ref = logsout.getElement('XY_ref');
state = logsout.getElement('state');
state_ref = logsout.getElement('state_ref');
input = logsout.getElement('input');

%%
XY = XY.Values;
XY_ref = XY_ref.Values;
state = state.Values;
state_ref = state_ref.Values;
input = input.Values;
% input.Data = reshape(squeeze(input.Data), [], 2);
inputData = squeeze(input.Data);
inputData = (inputData)';
%%
subplot(6,1,1)
plot(XY.Data(:,1), XY.Data(:,2),'b')
hold on
plot(XY_ref.Data(:,1), XY_ref.Data(:,2),'r')
grid on
title('abs pos')
xlabel('X')
ylabel('Y')
daspect([1 1 1])

subplot(6,1,2)
plot(state.Vx.Time, state.Vx.Data);
hold on
plot(state_ref.Time, state_ref.Data(1,:),'r')
title('Vx')
xlabel('time')
ylabel('Vx')

subplot(6,1,3)
plot(state.Vy.Time, state.Vy.Data);
hold on
plot(state_ref.Time, state_ref.Data(2,:),'r')
title('Vy')
xlabel('time')
ylabel('Vy')

subplot(6,1,4)
plot(state.dot_psi.Time, state.dot_psi.Data);
hold on
plot(state_ref.Time, state_ref.Data(3,:),'r')
title('dot psi')
xlabel('tim')
ylabel('dot psi')

subplot(6,1,5)
plot(input.Time, inputData(:,1));
title('delta')
xlabel('tim')
ylabel('delta')

subplot(6,1,6)
plot(input.Time, inputData(:,2));
title('Ffx')
xlabel('tim')
ylabel('Ffx')




