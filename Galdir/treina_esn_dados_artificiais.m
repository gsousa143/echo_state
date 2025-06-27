% Script de teste para a classe ESN
% Gera uma sC)rie temporal e usa a ESN para previsC#o

% Limpa o workspace
clear all;
close all;
clc;

%% Gera dados sintC)ticos

ruido = 0.2;

t = 0:0.1:50;  % vetor de tempo
y = sin(0.4*t) + ruido*randn(size(t));  % sC)rie temporal = seno + ruC-do
data = y';  % converte para coluna

%% Prepara dados para treino e teste
N = length(data);
train_len = floor(0.7 * N);  % 70% para treino
test_len = N - train_len;    % 30% para teste

% Organiza dados para entrada/saC-da (previsC#o um passo C  frente)
X = data(1:end-1);  % entradas
Y = data(2:end);    % saC-das (alvos)

% Separa conjuntos de treino e teste
X_train = X(1:train_len);
Y_train = Y(1:train_len);
X_test = X(train_len+1:end-1);
Y_test = Y(train_len+1:end-1);

% Vetores de tempo correspondentes
t_train = t(1:train_len);
t_test = t(train_len+1:end-1);

%% Cria e configura a ESN
n_neurons = 200;     % nC:mero de neurC4nios no reservatC3rio
n_inputs = 1;        % dimensC#o da entrada
n_outputs = 1;       % dimensC#o da saC-da

% Cria a ESN com parC"metros personalizados
esn = ESN_galdir(n_neurons, n_inputs, n_outputs, ...
    'leakrate', 0.5, ...        % taxa de vazamento / esquecimento
    'ro', 0.9, ...         % raio espectral
    'psi', 0.0, ...        % esparsidade
    'in_scale', 0.1, ...   % escala de entrada
    'noise_amplitude', 1e-5, ...
    'out_scale', 0); % ruC-do pequeno para estabilidade

%% Treina a ESN
warmupdrop = 50;  % descarta primeiros estados para estabilizar

% Adiciona dados de treino
esn.add_data(X_train, Y_train, warmupdrop);

% Treina usando validaC'C#o cruzada
disp('Treinando ESN...');
[erro_cv, melhor_reg] = esn.cum_train_cv(1e-8, 1e-2, 20, 5);
fprintf('Melhor regularização %.6f\n', melhor_reg);
fprintf('Erro CV: %.6f\n', erro_cv);

% Salva o reservatC3rio treinado
esn.save_reservoir('esn_teste.mat');

% Teste de carregamento
esn_nova = ESN_galdir(n_neurons, n_inputs, n_outputs);
esn_nova.load_reservoir('esn_teste.mat');

%% Testa a ESN
disp('Testando ESN...');
Y_pred = zeros(size(X_test));

% for i = 1:length(1000)
%      esn.update(X_test(1));
% end

% PrevisC#o
for i = 1:length(X_test)
    Y_pred(i) = esn.update(X_test(i));
end

% Calcula erro de teste
erro_teste = mean((Y_test - Y_pred).^2);
fprintf('Erro de teste (MSE): %.6f\n', erro_teste);

%% VisualizaC'C#o
figure;
subplot(2,1,1);
plot(t_train(1:length(Y_train)), Y_train, 'b-', 'LineWidth', 1);
title('Dados de Treino');
xlabel('Tempo');
ylabel('Amplitude');
grid on;

subplot(2,1,2);
hold on;
plot(t_test(1:length(Y_test)), Y_test, 'b-', 'LineWidth', 1, 'DisplayName', 'Real');
plot(t_test(1:length(Y_pred)), Y_pred, 'r--', 'LineWidth', 1, 'DisplayName', 'Previsto');
title('PrevisC#o vs Real (Teste)');
xlabel('Tempo');
ylabel('Amplitude');
legend('show');
grid on;
hold off;



%% Teste adicional: previsC#o livre (geraC'C#o autC4noma)
n_previsoes = 200;
entrada_inicial = X_test(end);
previsoes = zeros(n_previsoes, 1);
entrada_atual = entrada_inicial;

% Reset do estado
%esn_nova.a = zeros(n_neurons, 1);

% Fase de aquecimento
%warmup_data = X_test(end-warmupdrop:end);
%for i = 1:length(warmup_data)
%    esn_nova.update(warmup_data(i));
%end

for i = 1:length(1000)
     esn.update(X_test(1));
end

% GeraC'C#o autC4noma
for i = 1:n_previsoes
    previsao = esn_nova.update(entrada_atual);
    previsoes(i) = previsao;
    entrada_atual = previsao;  % usa previsC#o como prC3xima entrada
end

% Plotar previsC#o livre
figure;
t_hist = t(end-100:end-1);
data_hist = data(end-100:end-1);
t_prev = (t(end-1):0.1:t(end-1)+0.1*n_previsoes)';

plot(t_hist(1:length(data_hist)), data_hist, 'b-', 'LineWidth', 1);
hold on;
plot(t_prev(1:length(previsoes)), previsoes, 'r--', 'LineWidth', 1);
title('PrevisC#o Livre');
xlabel('Tempo');
ylabel('Amplitude');
legend('HistC3rico', 'PrevisC#o');
grid on;
