%% 
% Teste recursivo utilizando dados de um seno


clear;
close all;
clc;


disp("Teste Recursivo de uma rede ESN utilizando dados de um seno com ruido")
rng(0)


%% Gera os dados
t = 0:0.01:30;
seno = 10*sin(2*pi*0.5*t) + 0.5*randn(size(t));
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

t_train = t(1:amostras_treino);
t_test = t(amostras_treino+1:end);

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
warmup = 100;
esn.add_data(X_train, Y_train, warmup);
[best_mse, best_reg] = esn.train_cv(1e-8, 1e-2, 20, 5)
esn.resetState();



%% Avalia a rede com os dados de teste

%aquece a rede
X_warmup = X_train(:,end-warmup:end);
for i=1:warmup
    esn.predict(X_warmup(:,i));
end


Y_pred = [];
y = esn.predict(X_test(:,1));
for i = 1:amostras_teste
    x = [y; X_test(nout+(1:nin-nout), i)];
    y = esn.predict(x);
    Y_pred = [Y_pred, y];
end


hold on;
title("Trajetoria")
grid on;
plot(t_train,Y_train,'LineWidth',3,DisplayName="Treino")
plot(t_test,Y_test,'LineWidth',4,DisplayName="Teste")
plot(t_test,Y_pred,'LineWidth',3,DisplayName="ESN")
ylabel("Seno")
xlabel("Amostra")
legend('Box','off','Location','southoutside','Orientation','horizontal');
figure
title("Erro Normalizado")
erro = abs(Y_pred-Y_test)/(max(Y)-min(Y));
grid on;
plot(t(amostras_treino+1:end), erro,'LineWidth',3)
ylabel("Erro normalizado")
xlabel("Amostra")
