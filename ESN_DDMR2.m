


clear;
close all;
clc;


disp("Teste Recursivo, utiilzando os dados de treino para teste de rede ESN com dados de trajetoria de um Robô Movel")

rng('default')
rng(0)


%% Coleta de Dados

%%Dados do Qbot 2
load("dados_qbot_ofbg_inf.mat");
dados = dados(1:10:end,:); %reduzir o numero de amostras
x_a = dados(:,1);
y_a = dados(:,2);
theta = dados(:,3);
phi_r = dados(:,4);
phi_l = dados(:,5);
controls = [phi_r, phi_l];
states = [x_a, y_a, theta];
X_train = [states(1:end-1,:), controls(1:end-1,:)]';
Y_train = states(2:end,:)';
n_amostras = size(X_train,2);


T = 0.1;

t = T*(0:n_amostras-1);

%%Dados Virtuais
% load("dados_virtuais_DDMR.mat")



%% Cria ESN
nReservoir = 100; % Tamanho do Reservatorio w
nin = size(X_train,1); % Numero de Entradas
nout = size(Y_train,1); % Numero de saidas


esn = ESN(nin, nout, nReservoir, ...
    'leakrate', 0.1, ...
    'density', 1, ...
    'inputScaling', 0.1, ...
    'biasScaling', 0.05, ...
    'noise', 1e-5, ...
    'spectralRadius', 0.9);

%% Treina ESN
warmup = 50;
% [best_error, best_reg] = esn.train_cv(X_train, Y_train, warmup, 1e-8, 1e-2);
train_mse = esn.train(X_train,Y_train, warmup, 1e-8)
esn.resetState()
%% Avalia a rede com os dados de teste
Y_pred = [];
y = esn.predict(X_train(:,1));
for i = 1:n_amostras
    x = [y; X_train(nout+(1:nin-nout), i)];
    y = esn.predict(x);
    Y_pred = [Y_pred, y];
end

test_mse = mean((Y_pred-Y_train).^2, 'all')



estados = ["x", "y", "theta"];
figure("Name","Informações de Trajetoria");
subplot(1,2,1)
hold on;
title("Trajetoria")
grid on;
plot(X_train(1,:),X_train(2,:),'LineWidth',4,DisplayName="Treino")
plot(Y_pred(1,:),Y_pred(2,:),'LineWidth',3,DisplayName="ESN")

legend('Box','off','Location','best');
subplot(1,2,2)
hold on;
title("Erro de Trajetoria")
grid on;
erro_traj = sqrt((Y_train(1,:)-Y_pred(1,:)).^2 + (Y_train(2,:)-Y_pred(2,:)).^2);
plot(t, erro_traj,'lineWidth',3);
xlabel("Tempo (s)");
ylabel("Erro (m)");

figure("Name","Informações dos Estados");
sgtitle("Estados")
erro = abs(Y_train - Y_pred)./(max(Y_train,[],2)-min(Y_train,[],2));
for i = 1:nout
    subplot(2,nout,i)
    %estados
    hold on;
    plot(t,X_train(i,:),'LineWidth',4,DisplayName="Treino")
    plot(t, Y_pred(i,:),'LineWidth',3,DisplayName="ESN")
    xlabel("Tempo (s)");
    ylabel(estados(i));
    grid on;
end
for i = 1:nout
    subplot(2,nout,nout+i)
    plot(t,erro(i,:), 'LineWidth',3)
    xlabel("Tempo (s)");
    ylabel("Erro Normalizado em" + estados(i));
    grid on;
end


figure("Name","Manipulaveis")
sgtitle("Variaveis Manipulaveis")
subplot(2,1,1)
plot(t,X_train(4,:),'LineWidth',3)
xlabel("Tempo (s)")
ylabel("V_r (rad/s)")
subplot(2,1,2)
plot(t,X_train(5,:),'LineWidth',3)
xlabel("Tempo (s)")
ylabel("V_l (rad/s)")


