function plotter3(logsout)
% PLOTTER3 Plot simulink data for model3
%   [] = plotter3(logsout)
% 
%   DESCRIPTION
%       plot simulation result of model5
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
%           right front tire force (Ffxr)
%           left front tire force (Ffxl)
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
input.Data = squeeze(input.Data)';
%%
subplot(7,1,1)
plot(XY.Data(:,1), XY.Data(:,2),'b')
hold on
plot(XY_ref.Data(:,1), XY_ref.Data(:,2),'r')
grid on
title('abs pos')
xlabel('X')
ylabel('Y')
daspect([1 1 1])

subplot(7,1,2)
plot(state.Vx.Time, state.Vx.Data);
hold on
plot(state_ref.Time, state_ref.Data(1,:),'r')
title('Vx')
xlabel('time')
ylabel('Vx')

subplot(7,1,3)
plot(state.Vy.Time, state.Vy.Data);
hold on
plot(state_ref.Time, state_ref.Data(2,:),'r')
title('Vy')
xlabel('time')
ylabel('Vy')

subplot(7,1,4)
plot(state.dot_psi.Time, state.dot_psi.Data);
hold on
plot(state_ref.Time, state_ref.Data(3,:),'r')
title('dot psi')
xlabel('tim')
ylabel('dot psi')

subplot(7,1,5)
plot(input.Time, input.Data(:,1));
title('delta')
xlabel('tim')
ylabel('delta')

subplot(7,1,6)
plot(input.Time, input.Data(:,2));
title('Ffxr')
xlabel('tim')
ylabel('Ffxr')

subplot(7,1,7)
plot(input.Time, input.Data(:,3));
title('Ffxl')
xlabel('tim')
ylabel('Ffxl')




