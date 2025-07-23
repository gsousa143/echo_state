%% SCRIPT PRINCIPAL PARA TREINO E AVALIAÇÃO DE UMA ESN (VERSÃO ESTADO SEGUINTE)
clear all;
clc;
close all;
rng('default'); % Para reprodutibilidade

%% Parâmetros da Simulação e da ESN
nReservoir = 500;    % Tamanho do Reservatorio (neurônios)
nin = 5;             % Entradas da ESN: [x, y, theta, vr, vl]
nout = 3;            % Saídas da ESN: [x(t+T), y(t+T), theta(t+T)] %% ALTERADO
T_step = 0.1;        % Período de amostragem (deve ser o mesmo de criaDados)
rng(99999);

% Cria a Echo State Network com hiperparâmetros iniciais
esn = ESN(nin, nout, nReservoir, ...
    'leakrate', 0.5, ...
    'density', 0.01, ...
    'inputScaling', 0.1, ...
    'biasScaling', 0.05, ...
    'noise', 1e-4, ...
    'spectralRadius', 0.9);

%% Fase de Treinamento
n_trajetorias = 100;   % Número de trajetórias diferentes para treinar
n_amostras_treino = 600; % Amostras por trajetória
warmup = 0;         % Amostras a serem ignoradas no início de cada trajetória

disp('Iniciando coleta de dados para treinamento...');
for i = 1:n_trajetorias
    rng(i); % Semente diferente para cada trajetória
    % A função criaDados agora retorna o próximo estado como Y
    [X,Y] = criaDados(n_amostras_treino, T_step); 
    esn.add_data(X, Y, warmup); % Acumula estados do reservatório
end
disp('Dados coletados.');

% Treina a matriz de leitura (Wro) usando validação cruzada para achar o melhor
% parâmetro de regularização (ridge regression).
disp('Iniciando treinamento com validação cruzada...');
train_mse = esn.train_cv(1e-8, 1e-2, 20, 5);
fprintf('Treinamento concluído. Melhor erro de validação cruzada: %f\n', train_mse);
esn.save("esn_ddmr_next_state") % Salva o modelo com novo nome

%% Fase de Teste (Avaliação em Malha Fechada)
n_amostras_teste = 60/T_step; %trajetoria de 1min
rng(n_trajetorias + 1); % Usa uma semente nova, nunca vista no treino
[X_test, Y_test] = criaDados(n_amostras_teste, T_step);

disp('Iniciando avaliação em malha fechada...');

% --- LOOP DE PREDIÇÃO CORRIGIDO PARA PREVER O PRÓXIMO ESTADO --- %% ALTERADO
Y_pred = zeros(nout, n_amostras_teste); % Armazenará os estados previstos
X_sim = zeros(nout, n_amostras_teste);  % Armazenará os estados simulados [x, y, theta]

% Pega o estado inicial real dos dados de teste
x_atual_simulado = X_test(1:nout, 1); 
X_sim(:,1) = x_atual_simulado;

%% Carrega a Echo State
esn = ESN.load("esn_ddmr_next_state.mat");

for i = 1:n_amostras_teste
    % 1. Monta o vetor de entrada para a ESN com o estado ATUAL SIMULADO
    %    e os controles REAIS do conjunto de teste.
    controles_reais = X_test(nout+1:end, i);
    esn_input = [x_atual_simulado; controles_reais];
    
    % 2. Usa a ESN para prever o PRÓXIMO estado diretamente
    proximo_estado_previsto = esn.predict(esn_input);
    Y_pred(:,i) = proximo_estado_previsto;
    
    % 3. O estado previsto se torna o estado atual para a próxima iteração
    if i < n_amostras_teste
        x_atual_simulado = proximo_estado_previsto;
        X_sim(:,i+1) = x_atual_simulado;
    end
end
disp('Avaliação concluída.');

% Calcula o erro quadrático médio na previsão do PRÓXIMO ESTADO %% ALTERADO
test_mse = mean((Y_pred - Y_test).^2, 'all');
fprintf('MSE no Teste (previsão do próximo estado): %f\n', test_mse);

%% Vizualização dos Resultados
t = T_step * (0:n_amostras_teste-1);

% --- Plot das Variáveis de Controle ---
figure("Name","Variáveis de Controle");
sgtitle("Variáveis de Controle (Entradas do Sistema)")
subplot(2,1,1)
plot(t, X_test(4,:), 'LineWidth', 2)
xlabel("Tempo (s)")
ylabel("V_r (rad/s)")
grid on;
subplot(2,1,2)
plot(t, X_test(5,:), 'LineWidth', 2)
xlabel("Tempo (s)")
ylabel("V_l (rad/s)")
grid on;

% --- Plots da Trajetória (x,y) --- %% ALTERADO
% A trajetória real já está contida em X_test
X_real = X_test(1:nout, :);

figure("Name", "Informações de Trajetoria");
% Plot da trajetória no plano (x,y)
subplot(1,2,1)
hold on;
title("Trajetória no Plano (x,y)")
plot(X_real(1,:), X_real(2,:), 'LineWidth', 4, DisplayName="Real")
plot(X_sim(1,:), X_sim(2,:), 'LineWidth', 2, DisplayName="Previsão ESN")
xlabel("Posição x (m)");
ylabel("Posição y (m)");
legend('Box','off','Location','best');
grid on;
axis equal;

% Plot do erro de posição ao longo do tempo
subplot(1,2,2)
hold on;
title("Erro de Posição")
erro_posicao = sqrt((X_real(1,:) - X_sim(1,:)).^2 + (X_real(2,:) - X_sim(2,:)).^2);
plot(t, erro_posicao, 'lineWidth', 2);
xlabel("Tempo (s)");
ylabel("Erro Euclidiano (m)");
grid on;

% --- Plots dos Estados (Saídas da ESN) --- %% ALTERADO
figure("Name","Previsão dos Estados");
sgtitle("Previsão dos Estados (Saídas da ESN)")
estados = ["x (m)", "y (m)", "\theta (rad)"];
for i = 1:nout
    subplot(2,nout,i)
    hold on;
    plot(t, Y_test(i,:), 'LineWidth', 4, DisplayName="Real (próximo estado)")
    plot(t, Y_pred(i,:), 'LineWidth', 2, DisplayName="Previsão ESN")
    xlabel("Tempo (s)");
    ylabel(estados(i));
    legend('show','Location','best');
    grid on;
end

% Plot do erro normalizado de cada estado
erro_norm = abs(Y_test - Y_pred) ./ (max(Y_test,[],2) - min(Y_test,[],2));
for i = 1:nout
    subplot(2,nout,nout+i)
    plot(t, erro_norm(i,:), 'LineWidth', 2)
    xlabel("Tempo (s)");
    ylabel("Erro Norm. em " + extractBefore(estados(i)," "));
    grid on;
end


function [X, Y] = criaDados(n_amostras, T)
%CRIADADOS Gera dados de simulação para um DDMR.
%   Esta versão inclui manobras específicas: parado, movimento em linha
%   reta para frente e para trás, além do movimento aleatório original.
%
%   %% ALTERADO: A saída Y agora é o próximo estado xx(t+1), e não a derivada.

% Parâmetros do Robô
R = 0.034;      % Raio da roda (m)
L = 0.175;      % Distância entre as rodas (m)
max_u = 10;      % Maxima velocidade das rodas

% Condições Iniciais
x0 = [0;0;0];   % Estado inicial [x; y; theta]
uu = [0;0];     % Vetor de controle inicial
t_acum = 1;
nout = length(x0);

% Cinemática do DDMR (função de estado)
d_state = @(x,u) (R/2)*[cos(x(3)), cos(x(3)); sin(x(3)), sin(x(3)); 1/L, -1/L]*u;

% --- GERAÇÃO DOS SINAIS DE CONTROLE ---
while t_acum < n_amostras
    duracao = randi([5, 50]);
    tipo_manobra = randi([1, 3]); % 1:Aleatório, 2:Parado, 3:Reta
    
    switch tipo_manobra
        case 1 % Manobra Aleatória
            u_const = 2*max_u*rand(2,1) - max_u;
        case 2 % Manobra Parado
            u_const = [0; 0];
        case 3 % Manobra Linha Reta
            velocidade_reta = max_u*rand();
            if rand()<0.5
                u_const = [velocidade_reta; velocidade_reta];
            else
                u_const = [-velocidade_reta; -velocidade_reta];
            end
    end
    
    uu = [uu, repmat(u_const, 1, min(duracao, n_amostras-t_acum))];  
    t_acum = t_acum + duracao;
end

% --- SIMULAÇÃO DO MOVIMENTO DO ROBÔ ---
xx = zeros(n_amostras + 1, nout);
dxx = zeros(nout, n_amostras);
xx(1,:) = x0';

for i = 1:n_amostras
    u = uu(:,i);
    dx = d_state(xx(i,:)', u);
    xx(i+1,:) = xx(i,:) + (T*dx)';
    dxx(:,i) = dx;
end

% --- FORMATAÇÃO DOS DADOS DE SAÍDA ---
% X contém o estado e o controle em cada instante
X = [xx(1:end-1,:), uu']'; 
% Y contém o estado no instante seguinte (t+T)
Y = xx(2:end,:)'; %% ALTERADO

end
