function ref_ts = reference_maker2(T, Ts, dot_psi)
% REFERENCE_MAKER2 generate reference states
%     ref_ts = reference_maker2(T, Ts, dot_psi)
% 
%     DESCRIPTION
%         reference for model 3
%         return timeseries type data
%         radius is planned 0, dot_psi, -dot_psi, 0
%           dot_psi pi/6/12 recommended
%
%     INPUT
%         episode time (T)
%         time difference (Ts)
%
%     REFERENCE STATE
%         dot x
%         dot y
%         dot psi

time = 0:Ts:T;
steps = length(time);

ref = ones(steps , 3);

ref(:,1) = ref(:,1) * 30 * 1000 / 60 / 60;
ref(:,2) = ref(:,2) * 0;

q = fix(steps/4);
ref(1:q,3) = 0;
ref(q+1:q*2,3) = dot_psi;
ref(q*2+1:q*3,3) = -dot_psi;
ref(q*3+1:end,3) = 0;

ref = ref';
ref_ts = timeseries(ref, time);

end