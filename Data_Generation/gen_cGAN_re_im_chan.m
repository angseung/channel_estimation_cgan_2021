clc, clear, close all;

%% 전송 파라미터 설정
fft_len = 32;                % OFDM 부반송파의 수
mod_type = 2;               % 변조 차수 ex) 1 - BPSK, 2 - QPSK, 4 - 16QAM, 6 - 64QAM, 8 - 256QAM
rx_node = 1;                % 수신기의 수 (수신기의 안테나는 1개)
tx_ant = 32;                % 기지국의 안테나 수
rx_ant = 64;
snr = 10;                   % 전송 채널 SNR 범위
path = 3;
scatter = 2;
% iter = 300;               % 전송 반복 횟수
pilot_len = 8;
% num_datasets = 12000;
num_datasets = 1548 + 664;
% num_datasets = 100;
% num_datasets = 5000;

% 기본 파라미터 설정
model = SCM();
model.n_path = path;
model.n_mray = scatter;
model.fc = 2.5e9;
model.los = 1;
% model.fs = model.fc / 40;
cp_len = fft_len / 4;
% data_len = fft_len * mod_type;

% 안테나 거리 설정
model.tx_ant(3) = 1;
model.rx_ant(3) = 1;

% 채널 환경
N_d = rx_node;
N_s = rx_node;
% A = [1];

% 송수신 각 배열 안테나의 안테나 수 (수신, 송신)
model.ant(rx_ant, tx_ant);
N_tx = model.Ntx;
N_rx = model.Nrx;

% 기본 파라미터 설정
r_H = zeros(path, fft_len + cp_len, N_rx * N_d, N_tx);
t_H = zeros(path, N_rx * N_d, N_tx);
% eq_sym = zeros(N_d, fft_len);
% result_mimo = zeros(1, length(snr) );
% result_cap = zeros(1, length(snr) );

CH = zeros(num_datasets, N_rx, N_tx, path * 2);
Y = zeros(num_datasets, N_rx, pilot_len, 2);

%% Generating pilot 
% pilot_t = uniformPilotsGen(pilot_len);
% pilot_t = pilot_t{1,1};
% pilot = repmat(pilot_t, [1 tx_ant])';

%% 반복 시작
for curr_dat = 1 : num_datasets
%     fprintf("Generating %04dth data...\n", curr_dat)
    
    % 채널 계수 생성
    for d = 1:N_d
        temp = model.FD_channel(fft_len + cp_len);
        r_H(:,:,1+(d-1)*N_rx:d*N_rx,:) = temp;
    end

    t_H(:,:,:) = r_H(:,1,:,:); % (path, time, RX_tot, TX)
%     H = squeeze(sum(t_H, 1));

    % 데이터 생성
    pilot = cdm_gen_freq(pilot_len + 1, tx_ant);
    pilot = pilot(:, 1 : pilot_len);
    
    y = zeros(path, rx_ant, pilot_len);
    
    % Received signal at RX
    for p = 1 : path
        y(p, :, :) = squeeze(t_H(p, :, :)) * pilot;
    end
    
    y = squeeze(sum(y, 1));
    [y, ~] = awgn_noise(y, snr);
    
    %% Debug ONLY...
    zero_test = y(y == 0.0);
    assert(isempty(zero_test))
    
    %% Store current data to buffer
    Y(curr_dat, :, :, 1) = real(y);
    Y(curr_dat, :, :, 2) = imag(y);
    
    p_cnt = 1;
    
    for j = 1 : 2 : path * 2
        CH(curr_dat, :, :, j) = real(t_H(p_cnt, :, :));
        CH(curr_dat, :, :, j + 1) = imag(t_H(p_cnt, :, :));
        p_cnt = p_cnt + 1;
    end

end

%% Quantization Received Y
% Y_signed = zeros(size(Y));
% Y_signed(:, :, :, 1) = sign(Y(:, :, :, 1));
% Y_signed(:, :, :, 2) = sign(Y(:, :, :, 2));

%% Split data for training
trRatio = 0.7;
numTrSamples = floor(trRatio * num_datasets);
numValSamples = num_datasets - numTrSamples;

% input_da = Y_signed(1 : numTrSamples, :, :, :);
input_da = Y(1 : numTrSamples, :, :, :);
output_da = CH(1 : numTrSamples, :, :, :);

% input_da_test = Y_signed(numTrSamples + 1 : end, : , : ,:);
input_da_test = Y(numTrSamples + 1 : end, : , : ,:);
output_da_test = CH(numTrSamples + 1 : end, :, :, :);

%% Save Generated Data to mat v7.3 hd5 file...
formatSpec = "Gan_%d_dBOutdoorSCM_%dpath_%dscatter_re_im_chan_210608_v5.mat";
fname = sprintf(formatSpec, snr, path, scatter);
save("Gan_Data\" + fname,'input_da','output_da','input_da_test','output_da_test','-v7.3')