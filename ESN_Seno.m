%% 
% Teste recursivo utilizando dados de um seno


clear;
close all;
clc;

rng('default')
rng(0)


%% Gera os dados
t = 0:0.01:30;
seno = sin(2*pi*0.5*t) + 0.1*randn(size(t));
X = seno(1:end-1);
Y = seno(2:end);
t = t(1:end-1);

%% Separando dados entre treino e teste

n_amostras = size(X,2);
porcentagem_treino = 0.6;
amostras_treino = n_amostras*porcentagem_treino;
amostras_teste = n_amostras - amostras_treino;


X_train = X(:,1:amostras_treino);
Y_train = Y(:,1:amostras_treino);
X_test = X(:,amostras_treino+1:end);
Y_test = Y(:,amostras_treino+1:end);

%% Cria ESN
nReservoir = 1000; % Tamanho do Reservatorio w
nin = size(X,1); % Numero de Entradas
nout = size(Y,1); % Numero de saidas


esn = ESN(nin, nout, nReservoir, ...
    'leakrate', 0.1, ...
    'density', 1, ...
    'inputScaling', 0.1, ...
    'biasScaling', 0.01, ...
    'feedbackScaling', 0, ...
    'noise', 1e-5, ...
    'spectralRadius', 0.9);

%% Treina ESN
warmup = 200;
esn.train(X_train, Y_train, warmup, 1e-6);

%% Avalia a rede com os dados de teste
Y_pred = [];
y = esn.predict(X_test(:,1));
for i = 1:amostras_teste
    x = [y; X_test(nout+(1:nin-nout), i)];
    y = esn.predict(x);
    Y_pred = [Y_pred, y];
end

subplot(1,2,1)
hold on;
title("Trajetoria")
grid on;
plot(t,X,'LineWidth',4,DisplayName="Dados completos")
plot(t(1:amostras_treino), X_train,'LineWidth',2,DisplayName="Dados treino")
plot(t(amostras_treino+1:end), Y_pred,'LineWidth',2,DisplayName="Previsão da Rede")
ylabel("sin(t)")
xlabel("Amostra")
legend('Box','off','Location','best');
subplot(1,2,2)
title("Erro Normalizado")
erro = abs(Y_pred-Y_test)/(max(Y)-min(Y));
grid on;
plot(t(amostras_treino+1:end), erro,'LineWidth',2)
ylabel("Erro normalizado")
xlabel("Amostra")
