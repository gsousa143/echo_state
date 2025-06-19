%% 
% Realiza teste recursivo da ESN utilizando os dados de trajetoria de um
% DDMR utilizada no treino da rede.




clear;
close all;
clc;

rng('default')
rng(0)


%% Coleta de Dados

%%Dados do Qbot 2
load("dados_qbot_ofbg_zzx.mat");
x_a = dados(:,1);
y_a = dados(:,2);
theta = dados(:,3);
phi_r = dados(:,4);
phi_l = dados(:,5);
controls = [phi_r, phi_l];
states = [x_a, y_a, theta];
X_train = [states(1:end-1,:), controls(1:end-1,:)]';
Y_train = states(2:end,:)';
amostras_treino = size(X_train,2);


load("dados_qbot_ofbg_zzx.mat");
x_a = dados(:,1);
y_a = dados(:,2);
theta = dados(:,3);
phi_r = dados(:,4);
phi_l = dados(:,5);
controls = [phi_r, phi_l];
states = [x_a, y_a, theta];
X_test= [states(1:end-1,:), controls(1:end-1,:)]';
Y_test = states(2:end,:)';
amostras_teste = size(X_train,2);

%%Dados Virtuais
% load("dados_virtuais_DDMR.mat")



%% Cria ESN
nReservoir = 1000; % Tamanho do Reservatorio w
nin = size(X_train,1); % Numero de Entradas
nout = size(Y_train,1); % Numero de saidas


esn = ESN(nin, nout, nReservoir, ...
    'leakrate', 0.1, ...
    'density', 1, ...
    'inputScaling', 0.1, ...
    'biasScaling', 0.5, ...
    'noise', 1e-5, ...
    'spectralRadius', 0.9);

%% Treina ESN
warmup = 50;
% [best_error, best_reg] = esn.train_cv(X_train, Y_train, warmup, 1e-8, 1e-2);
esn.train(X_train,Y_train, warmup, 1e-8);
esn.resetState()
%% Avalia a rede com os dados de teste
Y_pred = [];
y = esn.predict(X_test(:,1));
for i = 1:amostras_teste
    x = [y; X_test(nout+(1:nin-nout), i)];
    y = esn.predict(x);
    Y_pred = [Y_pred, y];
end





estados = ["x", "y", "theta"];
figure;
hold on;
title("Trajetoria Teste")
grid on;
plot(Y_test(1,:),Y_test(2,:),'LineWidth',2,DisplayName="Target")
plot(Y_pred(1,:),Y_pred(2,:),'LineWidth',2,DisplayName="Previsão da Rede")

% figure;
% sgtitle("Estados")
% erro = abs(Y_test - Y_pred)./(max(Y_test,[],2)-min(Y_test,[],2));
% for i = 1:nout
%     subplot(2,nout,i)
%     %estados
%     hold on;
%     plot(X_train(i,:),'LineWidth',2,DisplayName="Target")
%     plot(Y_pred(i,:),'LineWidth',2,DisplayName="Previsão da Rede")
%     xlabel("Amostra");
%     ylabel(estados(i));
%     grid on;
% end
% for i = 1:nout
%     subplot(2,nout,3+i)
%     plot(erro(i,:), 'LineWidth',2)
%     xlabel("Amostra");
%     ylabel("Erro Normalizado " + estados(i));
%     grid on;
% end

