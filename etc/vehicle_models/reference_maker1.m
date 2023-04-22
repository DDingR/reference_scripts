function ref_ts = reference_maker1(T, Ts, dot_psi)
% REFERENCE_MAKER1 generate reference states
%     ref_ts = reference_maker1(T, Ts, dot_psi)
% 
%     DESCRIPTION
%         reference for model 1 and model 2
%         return timeseries type data
%         radius is planned 0, dot_psi, -dot_psi, 0
%           dot_psi pi/6/12 recommended
%
%     INPUT
%         episode time (T)
%         time difference (Ts)
%
%     REFERENCE STATE
%         y
%         dot y
%         psi
%         dot psi

time = 0:Ts:T;
steps = length(time);

ref = ones(steps , 4);

ref(:,1) = ref(:,1) * 0;
ref(:,2) = ref(:,2) * 0;

q = fix(steps/4);
q_time = 0:Ts:(q-1)*Ts;

psi = q_time* dot_psi;

ref(1:q,3) = 0;
ref(q+1:q*2,3) = psi';
ref(q*2+1:q*3,3) = psi(end)-psi;
ref(q*3+1:end,3) = 0;

ref(1:q,4) = 0;
ref(q+1:q*2,4) = dot_psi';
ref(q*2+1:q*3,4) = -dot_psi;
ref(q*3+1:end,4) = 0;

ref = ref';
ref_ts = timeseries(ref, time);

end

% dot_psi = pi/6/60;
% time = 0:Ts:T;
% psi = time * dot_psi;
% ref = zeros(length(time), 4);
% ref(:,3) = psi';
% ref(:,4) = ones(1, length(time)) * dot_psi;
% ref = ref';
% ref_ts = timeseries(ref, time);