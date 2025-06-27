
% Carrega a rede neural treinada e os dados
load('rede.mat');
load('dados.mat');

N = length(posicaoX);
M = N - (N * 0.3); 

entradasTeste = [posicaoX(M:end-1); posicaoY(M:end-1); Theta(M:end-1); velocidadeLinear(M:end-1); velocidadeAngular(M:end-1)]; % Entradas
alvosTeste = [posicaoX(M+1:end); posicaoY(M+1:end); Theta(M+1:end)];                         % SaC-da futura (t+1)

posicaoX = entradasTeste(1,:);
posicaoY = entradasTeste(2,:);
Theta = entradasTeste(3,:);
velocidadeLinear = entradasTeste(4,:);
velocidadeAngular = entradasTeste(5,:);

% Determina o nC:mero mC!ximo de passos possC-veis
numPassosMax = length(velocidadeLinear); % - M;
numPassos = min(1000, numPassosMax); % Usa o menor entre 1000 e o mC!ximo disponC-vel

% Prepara dados para predição recursiva
Y_rec = zeros(3, numPassos); % Armazena as prediC'C5es recursivas

% Inicializa com os primeiros valores reais dos dados de teste
estadoAtual = [posicaoX(1); posicaoY(1); Theta(1)];
entradaAtual = [estadoAtual; velocidadeLinear(1); velocidadeAngular(1)];

% Realizar prediC'C5es recursivas
for t = 1:numPassos
    % Faz a predição
    Y_rec(:, t) = net(entradaAtual);
    
    % Atualiza a entrada para prC3xima predição usando a saC-da atual
    if t < numPassos % Evita tentar acessar alC)m do C:ltimo C-ndice
        entradaAtual = [Y_rec(:, t); velocidadeLinear(t); velocidadeAngular(t)];
    end
end

% Comparação com os dados reais
figure;
subplot(3,1,1);
plot(posicaoX(1:numPassos), 'b', 'DisplayName', 'Real'); hold on;
plot(Y_rec(1, :), 'r--', 'DisplayName', 'Predição Recursiva');
title('Comparação de x - Predição Recursiva vs Real');
xlabel('Passo');
ylabel('x');
legend('show');

subplot(3,1,2);
plot(posicaoY(1:numPassos), 'b', 'DisplayName', 'Real'); hold on;
plot(Y_rec(2, :), 'r--', 'DisplayName', 'Predição Recursiva');
title('Comparação de y - Predição Recursiva vs Real');
xlabel('Passo');
ylabel('y');
legend('show');

subplot(3,1,3);
plot(Theta(1:numPassos), 'b', 'DisplayName', 'Real'); hold on;
plot(Y_rec(3, :), 'r--', 'DisplayName', 'Predição Recursiva');
title('Comparação de \theta - Predição Recursiva vs Real');
xlabel('Passo');
ylabel('\theta');
legend('show');

% Visualização da trajetC3ria
figure;
plot(posicaoX(1:numPassos), posicaoY(1:numPassos), 'b', 'DisplayName', 'Real'); hold on;
plot(Y_rec(1, :), Y_rec(2, :), 'r--', 'DisplayName', 'Predição Recursiva');
title('TrajetC3ria do RobC4 - Real vs Predição Recursiva');
xlabel('x');
ylabel('y');
grid on;
legend('show');

% CC!lculo do erro mC)dio quadrC!tico (RMSE) para cada variC!vel
rmse_x = sqrt(mean((posicaoX(1:numPassos) - Y_rec(1, :)).^2));
rmse_y = sqrt(mean((posicaoY(1:numPassos) - Y_rec(2, :)).^2));
rmse_theta = sqrt(mean((Theta(1:numPassos) - Y_rec(3, :)).^2));

fprintf('RMSE para x: %.4f\n', rmse_x);
fprintf('RMSE para y: %.4f\n', rmse_y);
fprintf('RMSE para theta: %.4f\n', rmse_theta);

% Exibe o nC:mero de passos utilizados
fprintf('NC:mero de passos utilizados: %d\n', numPassos);
