% Script de teste para a classe ESN
% Gera uma série temporal e usa a ESN para previsão

% Limpa o workspace
clear all;
close all;
clc;

rng(42);

%% Gera dados sintéticos

ruido = 0.01;

t = 0:0.1:50;  % vetor de tempo
y = sin(0.4*t) + ruido*randn(size(t));  % série temporal = seno + ruído
data = y';  % converte para coluna

%% Prepara dados para treino e teste
N = length(data);
train_len = floor(0.7 * N);  % 70% para treino
test_len = N - train_len;    % 30% para teste

% Organiza dados para entrada/saída (previsão um passo à frente)
X = data(1:end-1);  % entradas
Y = data(2:end);    % saídas (alvos)

% Separa conjuntos de treino e teste
X_train = X(1:train_len);
Y_train = Y(1:train_len);
X_test = X(train_len+1:end-1);
Y_test = Y(train_len+1:end-1);

% Vetores de tempo correspondentes
t_train = t(1:train_len);
t_test = t(train_len+1:end-1);

%% Cria e configura a ESN
n_neurons = 300;     % número de neurônios no reservatório
n_inputs = 1;        % dimensão da entrada
n_outputs = 1;       % dimensão da saída

% Cria a ESN com parâmetros personalizados
esn = ESN_galdir(n_neurons, n_inputs, n_outputs, ...
    'leakrate', 0.2, ...        % taxa de vazamento
    'ro', 0.99, ...         % raio espectral
    'psi', 0.1, ...        % esparsidade
    'in_scale', 0.1, ...   % escala de entrada
    'noise_amplitude', 1e-5); % ruído pequeno para estabilidade

%% Treina a ESN
warmupdrop = 0;  % descarta primeiros estados para estabilizar

% Adiciona dados de treino
esn.add_data(X_train, Y_train, warmupdrop);

% Treina usando validação cruzada
disp('Treinando ESN...');
[erro_cv, melhor_reg] = esn.cum_train_cv(1e-8, 1e-2, 5, 50);
fprintf('Melhor regularização: %.6f\n', melhor_reg);
fprintf('Erro CV: %.6f\n', erro_cv);

%% Testa a ESN
disp('Testando ESN...');
Y_pred = zeros(size(X_test));

% Reset do estado para teste
%esn.a = zeros(n_neurons, 1);

% Fase de aquecimento usando últimos dados do treino
warmup_data = X_train(end-50:end);
for i = 1:length(warmup_data)
    esn.update(warmup_data(i));
end

%Fase de aquecimento usando últimos dados do treino
% for i = 1:300
%     esn.update(X_test(1));
% end

% Previsão
for i = 1:length(X_test)
    Y_pred(i) = esn.update(X_test(i));
end

% Calcula erro de teste
erro_teste = mean((Y_test - Y_pred).^2);
fprintf('Erro de teste (MSE): %.6f\n', erro_teste);

%% Visualização
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
title('Previsão vs Real (Teste)');
xlabel('Tempo');
ylabel('Amplitude');
legend('show');
grid on;
hold off;

% Salva o reservatório treinado
esn.save_reservoir('esn_teste.mat');

% Teste de carregamento
esn_carregada = ESN_galdir(n_neurons, n_inputs, n_outputs);
esn_carregada.load_reservoir('esn_teste.mat');

%% Teste adicional: previsão recursiva 
previsoes = zeros(length(X_test), 1);


% Fase de aquecimento usando primeiro dado de teste
for i = 1:1000
     esn_carregada.update(X_test(1));
end


% Predicao recursiva
entrada_atual = X_test(1);
for i = 1:length(X_test)
    previsao = esn_carregada.update(entrada_atual);
    previsoes(i) = previsao;
    entrada_atual = previsao;  % usa previsão como próxima entrada
end

% Calcula erro de teste recursivo
erro_teste_recursivo = mean((Y_test - previsoes).^2);
fprintf('Erro de teste recursivo (MSE): %.6f\n', erro_teste_recursivo);

% Plotar previsão recursiva
figure;
hold on;

plot(t_train, Y_train, 'k-', 'LineWidth', 1);
hold on;

plot(t_test(1:end-1), Y_test, 'b:', 'LineWidth', 1);
hold on;

plot(t_test(1:end-1), previsoes, 'r--', 'LineWidth', 1);
title('Previsão Recursiva: aquecimento com primeiro ponto de teste');
xlabel('Tempo');
ylabel('Amplitude');
legend('Histórico de Entrada', 'Histórico de Saida', 'Predicao de Saida', 'Location', 'southwest');
grid on;
hold off;


%% Teste adicional: previsão recursiva aquecimento com historico
esn_carregada = ESN_galdir(n_neurons, n_inputs, n_outputs);
esn_carregada.load_reservoir('esn_teste.mat');
previsoes = zeros(length(X_test), 1);

% Fase de aquecimento usando últimos dados do treino
% for i = 1:1000
%       esn_carregada.update(X_train(1));
% end
warmup_data = X_train;
for i = 1:length(warmup_data)
     esn_carregada.update(warmup_data(i));
end

% Predicao recursiva
entrada_atual = X_test(1);
for i = 1:length(X_test)
    previsao = esn_carregada.update(entrada_atual);
    previsoes(i) = previsao;
    entrada_atual = previsao;  % usa previsão como próxima entrada
end

% Calcula erro de teste recursivo
erro_teste_recursivo = mean((Y_test - previsoes).^2);
fprintf('Erro de teste recursivo (MSE): %.6f\n', erro_teste_recursivo);

% Plotar previsão recursiva
figure;
hold on;

plot(t_train, Y_train, 'k-', 'LineWidth', 1);
hold on;

plot(t_test(1:end-1), Y_test, 'b:', 'LineWidth', 1);
hold on;

plot(t_test(1:end-1), previsoes, 'r--', 'LineWidth', 1);
title('Previsão Recursiva: aquecimento com histórico');
xlabel('Tempo');
ylabel('Amplitude');
legend('Histórico de Entrada', 'Histórico de Saida', 'Predicao de Saida', 'Location', 'southwest');
grid on;
