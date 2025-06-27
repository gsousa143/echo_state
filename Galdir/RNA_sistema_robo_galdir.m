clear all;
close all;

% ParC"metros do sistema
DeltaT = 0.2; % Intervalo de tempo (s)
N = 10000; %6000;      % NC:mero de amostras
M = N - (N * 0.3); %5500;    % DivisC#o do conjunto de dados

epocas_treino = 3000;

%L= 20; % limite da malha

% Inicialização das variC!veis
rng('default');
rng(0); % Garante repetibilidade do experimento
posicaoX = zeros(1, N); %inicializa os estados em zero 
posicaoY = zeros(1, N);
Theta = zeros(1, N); 

% Geração de velocidades 
velocidadeLinear = zeros(1, N); % Velocidade linear
velocidadeAngular = zeros(1, N); % Velocidade angular
t = 2;           % Indice de tempo

%criando dados aleatorios
while t <= N
    % Define duração do bloco (entre 30 e 50 amostras)
    blocoDuracao = randi([50, 100]);
    
    % Define valores constantes para v e w no bloco
    v_const = 0.2 + 0.4 * rand();         % Velocidade linear (0.2 a 0.6 m/s)
    w_const = -pi/6 + (pi/3) * rand(); % Velocidade angular (-pi/6 a pi/6 rad/s)
    
    % Preenche os valores no intervalo do bloco
    blocoFim = min(t + blocoDuracao - 1, N);
    velocidadeLinear(t:blocoFim) = v_const;
    velocidadeAngular(t:blocoFim) = w_const;
    t = blocoFim + 1;
end

% Simulação do sistema dinC"mico do robC4
for t = 2:N
    % Atualiza os estados do sistema com base nas equaC'C5es discretizadas
    posicaoX(t) = posicaoX(t-1) + DeltaT * (velocidadeLinear(t-1) * cos(Theta(t-1)));
    posicaoY(t) = posicaoY(t-1) + DeltaT * (velocidadeLinear(t-1) * sin(Theta(t-1)));
    Theta(t) = Theta(t-1) + DeltaT * velocidadeAngular(t-1);
end

save('dados.mat', 'posicaoX', 'posicaoY', 'Theta', 'velocidadeLinear', 'velocidadeAngular');

% Dados de entrada e saC-da para a RNA
entradasTreino = [posicaoX(1:M-1); posicaoY(1:M-1); Theta(1:M-1); velocidadeLinear(1:M-1); velocidadeAngular(1:M-1)]; % Entradas
alvosTreino = [posicaoX(2:M); posicaoY(2:M); Theta(2:M)]; % SaC-da futura (t+1)

% Configuração da rede neural
net = feedforwardnet([5 5]); % Rede quantidade de neurC4nios na camada oculta - ajustC!vel 
net.trainFcn = 'trainlm';   % Algoritmo de treinamento Levenberg-Marquardt
%net.trainFcn = 'trainbr';          % Algoritmo com regularização bayesiana
%net.trainFcn = 'traingd';        

% DivisC#o dos dados
net.divideParam.trainRatio = 0.70; % 80% para treinamento
net.divideParam.valRatio = 0.30;  % 10% para validação
%net.divideParam.testRatio = 0.15; % 10% para teste

net.trainParam.max_fail = 10; % early stopping
net.trainParam.epochs = epocas_treino; % early stopping?

% Treinamento da rede
%[net, tr] = train(net, entradasTreino, alvosTreino, 'useParallel','yes','showResources','yes');
[net, tr] = train(net, entradasTreino, alvosTreino, 'showResources','yes');
save rede net     % Salva arquivo com a rede treinada

%% Validação 
% Novo conjunto de dados para validação
entradasTeste = [posicaoX(M:end-1); posicaoY(M:end-1); Theta(M:end-1); velocidadeLinear(M:end-1); velocidadeAngular(M:end-1)]; % Entradas
alvosTeste = [posicaoX(M+1:end); posicaoY(M+1:end); Theta(M+1:end)];                         % SaC-da futura (t+1)

% SaC-da da RNA para o conjunto de teste
Y_pred = net(entradasTeste);

% Visualização dos resultados
figure;
subplot(3,1,1);
plot(alvosTeste(1, :), 'b'); hold on;
plot(Y_pred(1, :), 'r--');
title('Comparação de x');
xlabel('Amostra');
ylabel('x');
legend('Real', 'Predito');

subplot(3,1,2);
plot(alvosTeste(2, :), 'b'); hold on;
plot(Y_pred(2, :), 'r--');
title('Comparação de y');
xlabel('Amostra');
ylabel('y');
legend('Real', 'Predito');

subplot(3,1,3);
plot(alvosTeste(3, :), 'b'); hold on;
plot(Y_pred(3, :), 'r--');
title('Comparação de \Theta');
xlabel('Amostra');
ylabel('\Theta');
legend('Real', 'Predito');

% Visualização das entradas de velocidade 
figure;
subplot(2,1,1);
plot(velocidadeLinear(1, :), 'b'); hold on;
title('Velocidade linear (-0.6 a 0.6 m/s)');
xlabel('Amostra');
ylabel('v');

subplot(2,1,2);
plot(velocidadeAngular(1, :), 'b'); hold on;
title('Velocidade angular (-pi/6 a pi/6 rad/s)');
xlabel('Amostra');
ylabel('w');


% Animação do movimento do robC4 (real vs predito pela RNA)
%close all;

% ConfiguraC'C5es iniciais para o plot
% figure;
% hold on;
% grid on;
% %axis([-L,L,-L,L]);
% title('Animação: Posição do RobC4 (Real vs Predito)');
% xlabel('x');
% ylabel('y');
% 
% A = 0.4; % Amplitude do vetor para plotar a orientação do robC4
% posicaoX= alvosTeste(1, :);
% posicaoY= alvosTeste(2, :);
% Theta= alvosTeste(3,:);
% % Plota os elementos iniciais
% real_pos = plot(posicaoX(1), posicaoY(1), 'bo', 'MarkerSize', 12, 'DisplayName', 'Real'); % RobC4 real
% pred_pos = plot(Y_pred(1, 1), Y_pred(2, 1), 'ro', 'MarkerSize', 12, 'DisplayName', 'Predito'); % RobC4 predito
% 
% real_dir = plot([posicaoX(1), posicaoX(1) + A * cos(Theta(1))], [posicaoY(1), posicaoY(1) + A * sin(Theta(1))], 'b-', 'LineWidth', 2); % Direção real
% pred_dir = plot([Y_pred(1, 1), Y_pred(1, 1) + A * cos(Y_pred(3, 1))], ...
%                 [Y_pred(2, 1), Y_pred(2, 1) + A * sin(Y_pred(3, 1))], 'r-', 'LineWidth', 2); % Direção predita
% 
% legend('real', 'predito');

% Animação do movimento do robC4
% for t = 1:length(alvosTeste)
%     % Atualiza a posição e direção real
%     set(real_pos, 'XData', posicaoX(t), 'YData', posicaoY(t));
%     set(real_dir, 'XData', [posicaoX(t), posicaoX(t) + A * cos(Theta(t))], ...
%                   'YData', [posicaoY(t), posicaoY(t) + A * sin(Theta(t))]);
% 
%     % Atualiza a posição e direção predita
%     if t <= size(Y_pred, 2) % Garante que nC#o exceda o tamanho das prediC'C5es
%         set(pred_pos, 'XData', Y_pred(1, t), 'YData', Y_pred(2, t));
%         set(pred_dir, 'XData', [Y_pred(1, t), Y_pred(1, t) + A * cos(Y_pred(3, t))], ...
%                       'YData', [Y_pred(2, t), Y_pred(2, t) + A * sin(Y_pred(3, t))]);
%     end
% 
%     % Atualiza o plot
%     pause(0.02);
%     drawnow;
% end
