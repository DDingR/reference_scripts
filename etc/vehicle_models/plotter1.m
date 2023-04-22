function plotter1(logsout)
% PLOTTER1 Plot simulink data for model 1 and model2
%   [] = plotter1(logsout)
% 
%   DESCRIPTION
%       plot simulation result of model1 and model2
% 
%   INPUT DATA  
%       XY [2,1]
%           global x coordinate
%           global y coordinate  
%       XY_ref [2,1]
%           global reference x coordinate
%           global reference y coordinate
%       state [4,1]
%           body-fixed y
%           body-fixed dot y
%           body-fixed psi
%           body-fixed dot psi    
%       state_ref [4,1]
%           reference body-fixed y
%           reference body-fixed dot y
%           reference body-fixed psi
%           reference body-fixed dot psi    
%       input [1,1]
%           steering angle (delta)
%   
%   PLOT FIGURES
%       abs pos
%            position on gloabl coordinates
%       states
%       input

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
plot(state.y.Time, state.y.Data);
hold on
plot(state_ref.Time, state_ref.Data(1,:),'r')
title('y')
xlabel('time')
ylabel('y')

subplot(6,1,3)
plot(state.dot_y.Time, state.dot_y.Data);
hold on
plot(state_ref.Time, state_ref.Data(2,:),'r')
title('dot_y')
xlabel('time')
ylabel('dot_y')

subplot(6,1,4)
plot(state.psi.Time, state.psi.Data);
hold on
plot(state_ref.Time, state_ref.Data(3,:),'r')
title('psi')
xlabel('tim')
ylabel('psi')

subplot(6,1,5)
plot(state.dot_psi.Time, state.dot_psi.Data);
hold on
plot(state_ref.Time, state_ref.Data(4,:),'r')
title('dot_psi')
xlabel('time')
ylabel('dot_psi')

subplot(6,1,6)
plot(input.Time, squeeze(input.Data));
title('input')
xlabel('time')
ylabel('input')





