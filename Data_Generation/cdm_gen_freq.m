function [seq] = cdm_gen_freq(len, num)
% len: seqence의 길이
% num: layer의 수
% seq: layer 별로 다르게 위상 천이되어 생성된 sequence

% layer 별 위상 차이
alpha = 2*pi / num * (1:num-1);

% Zadoff-chu sequence 생성
if mod(len,2) == 0, N = len+1;
else N = len; end
R = max( factor(N) ) + 1;

seq = zeros(num, len);
tmp = zadoffChuSeq(R,N).';
seq(1,:) = tmp(1:len);

for i = 2:num, seq(i,:) = seq(1,:) .* exp( 1j * alpha(i-1) * (0:len-1) ); end

