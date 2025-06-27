

clear;
close all;
clc;

disp("Teste Recursivo, avaliando a predição de uma rede ESN utilizando dados de trajetoria de um Robô Movel")

rng('default')
rng(0)


%% Coleta de Dados

%%Dados do Qbot 2
load("dados_qbot_ofbg_inf.mat");
dados = dados(1:10:end,:);
x_a = dados(:,1);
y_a = dados(:,2);
theta = dados(:,3);
phi_r = dados(:,4);
phi_l = dados(:,5);

controls = [phi_r, phi_l];
states = [x_a, y_a, theta];


X = [states(1:end-1,:), controls(1:end-1,:)]';
Y = states(2:end,:)';
T = 0.1;




% %Dados Virtuais
% load("dados_virtuais_DDMR.mat")





%% Separando dados entre treino e teste

n_amostras = size(X,2);
porcentagem_treino = 0.6;
amostras_treino = round(n_amostras*porcentagem_treino);
amostras_teste = n_amostras - amostras_treino;


X_train = X(:,1:amostras_treino);
Y_train = Y(:,1:amostras_treino);
X_test = X(:,amostras_treino+1:end);
Y_test = Y(:,amostras_treino+1:end);


t = T*(0:n_amostras-1);
t_train = t(1:amostras_treino);
t_test = t(amostras_treino+1:end);
%% Cria ESN
nReservoir = 1000; % Tamanho do Reservatorio w
nin = size(X,1); % Numero de Entradas
nout = size(Y,1); % Numero de saidas


esn = ESN(nin, nout, nReservoir, ...
    'leakrate', 0.1, ...
    'density', 0.01, ...
    'inputScaling', 0.1, ...
    'biasScaling', 0.01, ...
    'feedbackScaling', 0.0, ...
    'noise', 1e-5, ...
    'spectralRadius', 0.9);

%% Treina ESN
warmup = 100;
% [best_mse, best_reg] = esn.train_cv(X_train, Y_train, warmup, 1e-8, 1e-2, 20, 5)
train_mse = esn.train(X_train,Y_train, warmup, 1e-8);
esn.resetState();
%% Avalia a rede com os dados de teste
Y_pred = [];

%aquece a rede
X_warmup = X_train(:,end-warmup:end);
for i=1:warmup
    esn.predict(X_warmup(:,i));
end



y = esn.predict(X_test(:,1));


for i = 1:amostras_teste
    x = [y; X_test(nout+(1:nin-nout), i)];
    y = esn.predict(x);
    Y_pred = [Y_pred, y];
end


test_mse = mean((Y_pred-Y_test).^2, 'all')


estados = ["x", "y", "theta"];
figure("Name","Informações de Trajetoria");
subplot(1,2,1)
hold on;
title("Trajetoria")
grid on;
plot(X_train(1,:),X_train(2,:),'LineWidth',3,DisplayName="Treino")
plot(Y_test(1,:),Y_test(2,:),'LineWidth',4,DisplayName="Teste")
plot(Y_pred(1,:),Y_pred(2,:),'LineWidth',3,DisplayName="ESN")
axis padded
legend('Box','off','Location','best');
subplot(1,2,2)
hold on;
title("Erro de Trajetoria")
grid on;
erro_traj = sqrt((X(1,amostras_treino+1:end)-Y_pred(1,:)).^2 + (X(2,amostras_treino+1:end)-Y_pred(2,:)).^2);
plot(t_test, erro_traj,'lineWidth',2);
xlabel("Tempo (s)");
ylabel("Erro (m)");
axis padded

figure("Name","Informações dos Estados");
sgtitle("Estados")
erro = abs(Y_test - Y_pred)./(max(Y,[],2)-min(Y,[],2));
for i = 1:nout
    subplot(2,nout,i)
    %estados
    hold on;
    plot(t_train, X_train(i,:),'LineWidth',3,DisplayName="Treino")
    plot(t_test, Y_test(i,:),'LineWidth',3,DisplayName="Teste")
    plot(t_test, Y_pred(i,:),'LineWidth',3,DisplayName="ESN")
    xlabel("Tempo (s)");
    ylabel(estados(i));
    grid on;
end
axis padded
for i = 1:nout
    subplot(2,nout,nout+i)
    plot(t_test,erro(i,:), 'LineWidth',3)
    xlabel("Tempo (s)");
    ylabel("Erro Normalizado em" + estados(i));
    grid on;
end
axis padded

figure("Name","Manipulaveis")
sgtitle("Variaveis Manipulaveis")
subplot(2,1,1)
plot(t,X(4,:),"LineWidth",3)
xlabel("Tempo (s)")
ylabel("V_r (rad/s)")
subplot(2,1,2)
plot(t,X(5,:),"LineWidth",3)
xlabel("Tempo (s)")
ylabel("V_l (rad/s)")
axis padded
