clc, clear, close all;

%% 전송 파라미터 설정... sparse
fft_len = 8;                % OFDM 부반송파의 수
mod_type = 2;               % 변조 차수 ex) 1 - BPSK, 2 - QPSK, 4 - 16QAM, 6 - 64QAM, 8 - 256QAM
rx_node = 1;                % 수신기의 수 (수신기의 안테나는 1개)
tx_ant = 32;                % 기지국의 안테나 수
rx_ant = 64;
snr = 10;                   % 전송 채널 SNR 범위
path = 3;
scatter = 2;
pilot_len = 8;
num_datasets = 1000;

% 기본 파라미터 설정
model = SCM();
model.n_path = path;
model.n_mray = scatter;
model.fc = 800e6;
model.fs = model.fc / 40;
model.los = 1;
cp_len = fft_len / 4;

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

DoF_dat = zeros(num_datasets, 3);
G_dat = zeros(num_datasets, 3);

sel_tx = 10;
sel_rx = 20;

% For Sparse...
for curr_dat = 1 : num_datasets
    
    % 채널 계수 생성
    for d = 1:N_d
        temp = model.FD_channel(fft_len + cp_len);
        r_H(:,:,1+(d-1)*N_rx:d*N_rx,:) = temp;
    end

    t_H(:,:,:) = r_H(:,1,:,:); % (path, time, RX_tot, TX)
    h = t_H(:, sel_rx, sel_tx);
    
    DoF = rank(h * h');
    G = get_gini_index(h);
    
    DoF_dat(curr_dat, 1) = DoF;
    G_dat(curr_dat, 1) = G;
    
end

%% 전송 파라미터 설정 ... semi-sparse
fft_len = 8;                % OFDM 부반송파의 수
mod_type = 2;               % 변조 차수 ex) 1 - BPSK, 2 - QPSK, 4 - 16QAM, 6 - 64QAM, 8 - 256QAM
rx_node = 1;                % 수신기의 수 (수신기의 안테나는 1개)
tx_ant = 32;                % 기지국의 안테나 수
rx_ant = 64;
snr = 10;                   % 전송 채널 SNR 범위
path = 7;
scatter = 2;
pilot_len = 8;

% 기본 파라미터 설정
model = SCM();
model.n_path = path;
model.n_mray = scatter;
model.fc = 800e6;
model.fs = model.fc / 40;
model.los = 1;
cp_len = fft_len / 4;

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

% For non_sparse...
for curr_dat = 1 : num_datasets
%     fprintf("Generating %04dth data...\n", curr_dat)
    
    % 채널 계수 생성
    for d = 1:N_d
        temp = model.FD_channel(fft_len + cp_len);
        r_H(:,:,1+(d-1)*N_rx:d*N_rx,:) = temp;
    end

    t_H(:,:,:) = r_H(:,1,:,:); % (path, time, RX_tot, TX)
%     h = mean(t_H, [2, 3]);
    h = t_H(:, sel_rx, sel_tx);
    
    DoF = rank(h * h');
    G = get_gini_index(h);
    
    DoF_dat(curr_dat, 2) = DoF;
    G_dat(curr_dat, 2) = G;
    
    
end

%% 전송 파라미터 설정 ... non-sparse
fft_len = 8;                % OFDM 부반송파의 수
mod_type = 2;               % 변조 차수 ex) 1 - BPSK, 2 - QPSK, 4 - 16QAM, 6 - 64QAM, 8 - 256QAM
rx_node = 1;                % 수신기의 수 (수신기의 안테나는 1개)
tx_ant = 32;                % 기지국의 안테나 수
rx_ant = 64;
snr = 10;                   % 전송 채널 SNR 범위
path = 12;
scatter = 20;
pilot_len = 8;

% 기본 파라미터 설정
model = SCM();
model.n_path = path;
model.n_mray = scatter;
model.fc = 800e6;
% model.fs = model.fc / 40;
model.los = 1;
cp_len = fft_len / 4;

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

% For non_sparse...
for curr_dat = 1 : num_datasets
%     fprintf("Generating %04dth data...\n", curr_dat)
    
    % 채널 계수 생성
    for d = 1:N_d
        temp = model.FD_channel(fft_len + cp_len);
        r_H(:,:,1+(d-1)*N_rx:d*N_rx,:) = temp;
    end

    t_H(:,:,:) = r_H(:,1,:,:); % (path, time, RX_tot, TX)
%     h = mean(t_H, [2, 3]);
    h = t_H(:, sel_rx, sel_tx);
    
    DoF = rank(h * h');
    G = get_gini_index(h);
    
    DoF_dat(curr_dat, 3) = DoF;
    G_dat(curr_dat, 3) = G;
    
    
end

G_val = mean(G_dat, 1)
