classdef ESN < handle
    % Echo State Network (ESN)
    % Rede neural recorrente com reservatório dinâmico de estados internos.

    properties
        nInput          % Número de entradas
        nReservoir      % Número de neurônios no reservatório
        nOutput         % Número de saídas

        spectralRadius = 0.9     % Raio espectral da matriz de recorrência (>1 Para garantir a ESP)
        inputScaling = 1.0       % Fator de escala para os pesos de entrada
        biasScaling = 0.1        % Fator de escala para os termos de viés (bias)
        leakRate = 0.5           % Taxa de vazamento (leaky integration)
        density = 0.1            % Densidade da matriz de conexões recorrentes (sparse matrix)
        noise = 1e-5             % Nível de ruído adicionado ao estado interno
        feedbackScaling = 0      % Fator de escala para os pesos de realimentação (saída → reservatório)

        Wir   % Pesos de entrada para o reservatório
        Wbr   % Pesos de viés para o reservatório
        Wrr   % Pesos de recorrência do reservatório (reservoir → reservoir)
        Wor   % Pesos de realimentação da saída para o reservatório
        Wro   % Pesos de leitura do reservatório para a saída

        state      % Estado interno atual do reservatório
        lastOutput % Última saída gerada (utilizada para feedback)
    end

    methods
        function obj = ESN(nInput, nOutput, nReservoir, varargin)
            % Construtor da ESN com parâmetros opcionais

            % Leitura dos parâmetros opcionais
            p = inputParser;
            addParameter(p, 'spectralRadius', obj.spectralRadius);
            addParameter(p, 'inputScaling', obj.inputScaling);
            addParameter(p, 'biasScaling', obj.biasScaling);
            addParameter(p, 'leakRate', obj.leakRate);
            addParameter(p, 'density', obj.density);
            addParameter(p, 'noise', obj.noise);
            addParameter(p, 'feedbackScaling', obj.feedbackScaling);
            parse(p, varargin{:});
            f = p.Results;

            % Armazenamento dos parâmetros
            obj.nInput = nInput;
            obj.nReservoir = nReservoir;
            obj.nOutput = nOutput;

            obj.spectralRadius = f.spectralRadius;
            obj.inputScaling = f.inputScaling;
            obj.biasScaling = f.biasScaling;
            obj.leakRate = f.leakRate;
            obj.density = f.density;
            obj.noise = f.noise;
            obj.feedbackScaling = f.feedbackScaling;

            % Inicialização das matrizes de pesos
            obj.Wir = (rand(nReservoir, nInput)*2 - 1)*obj.inputScaling;
            obj.Wbr = (rand(nReservoir, 1)*2 - 1)*obj.biasScaling;
            obj.Wrr = sprand(nReservoir, nReservoir, obj.density)*2 - 1;
            obj.Wor = (rand(nReservoir, nOutput)*2 - 1)*obj.feedbackScaling;

            % obj.Wir = randn(nReservoir, nInput)*obj.inputScaling;
            % obj.Wbr = randn(nReservoir, 1)*obj.biasScaling;
            % obj.Wrr = sprandn(nReservoir, nReservoir, obj.density);
            % obj.Wor = randn(nReservoir, nOutput)*obj.feedbackScaling;

            % Ajuste do raio espectral da matriz de recorrência
            lambda = abs(eigs(obj.Wrr, 1, 'LM'));
            obj.Wrr = (obj.Wrr / lambda)*obj.spectralRadius;

            % Inicialização do estado
            obj.state = zeros(nReservoir, 1);
            obj.lastOutput = zeros(nOutput, 1);
        end

function train_mse = train(obj, inputSequence, targetSequence, warmup, reg)
    T = size(inputSequence, 2);
    X = zeros(obj.nReservoir, T - warmup);
    Ytarget = targetSequence(:, warmup+1:end);
    obj.lastOutput = zeros(obj.nOutput, 1);
    obj.resetState();

    for t = 1:T
        u = inputSequence(:, t);
        y_feedback = obj.lastOutput;
        x = tanh(obj.Wir*u + obj.Wrr*obj.state + ...
            obj.Wor*y_feedback + obj.Wbr + obj.noise*randn(obj.nReservoir, 1));
        obj.state = (1 - obj.leakRate)*obj.state + obj.leakRate*x;

        if t > warmup
            X(:, t - warmup) = obj.state;
        end

        obj.lastOutput = targetSequence(:, t); % Durante o treino, usa saída real
    end

    X_aug = [X; ones(1, size(X, 2))]; % Adiciona bias

    % Treinamento via pseudo-inversa regularizada (SVD)
    [U, S, V] = svd(X_aug, 'econ');
    S_inv = diag(1 ./ (diag(S).^2 + reg)) .* diag(S);
    obj.Wro = Ytarget * V * S_inv' * U';


    %  ridge regression
    % obj.Wro = (Ytarget*X_aug') / (X_aug*X_aug' + reg*eye(size(X_aug, 1)));

    % Predição e cálculo do erro de treinamento
    Y_pred = obj.Wro * X_aug;

    train_mse = mean((Ytarget - Y_pred).^2, 'all');
end

            

        function [best_error, best_reg] = train_cv(obj, inputSequence, targetSequence, warmup, min_reg, max_reg, tests, folds)
            % Treinamento com validação cruzada para escolha da regularização ideal
            % tests: número de valores de lambda testados
            % folds: número de partições para validação cruzada

            if nargin < 7, tests = 50; end
            if nargin < 8, folds = 10; end
            if min_reg > max_reg, error("min_reg deve ser menor que max_reg"); end

            reg_list = linspace(min_reg, max_reg, tests);
            T = size(inputSequence, 2);
            valid_range = warmup+1:T;
            total_samples = length(valid_range);

            fold_size = floor(total_samples / folds);
            best_error = inf;
            best_reg = reg_list(1);

            for i = 1:tests
                reg = reg_list(i);
                fold_errors = zeros(1, folds);

                for fold = 1:folds
                    idx_val_start = (fold - 1) * fold_size + 1;
                    if fold == folds
                        idx_val_end = total_samples;
                    else
                        idx_val_end = fold * fold_size;
                    end

                    idx_val = valid_range(idx_val_start:idx_val_end);
                    idx_train = setdiff(valid_range, idx_val);

                    % Coleta de estados para treinamento
                    obj.resetState();
                    obj.lastOutput = zeros(obj.nOutput, 1);
                    X = zeros(obj.nReservoir, length(idx_train));
                    Y = targetSequence(:, idx_train);

                    t_index = 1;
                    for t = 1:T
                        u = inputSequence(:, t);
                        y_fb = obj.lastOutput;
                        x = tanh(obj.Wir*u + obj.Wrr*obj.state + ...
                            obj.Wor*y_fb + obj.Wbr + obj.noise*randn(obj.nReservoir, 1));
                        obj.state = (1 - obj.leakRate)*obj.state + obj.leakRate*x;
                        obj.lastOutput = targetSequence(:, t);

                        if ismember(t, idx_train)
                            X(:, t_index) = obj.state;
                            t_index = t_index + 1;
                        end
                    end

                    % Treinamento com dados de treino do fold
                    X_aug = [X; ones(1, size(X, 2))];
                    theta = (Y * X_aug') / (X_aug * X_aug' + reg * eye(size(X_aug, 1)));

                    % Validação
                    obj.resetState();
                    obj.lastOutput = zeros(obj.nOutput, 1);
                    X_val = zeros(obj.nReservoir, length(idx_val));
                    Y_val = targetSequence(:, idx_val);

                    t_index = 1;
                    for t = 1:T
                        u = inputSequence(:, t);
                        y_fb = obj.lastOutput;
                        x = tanh(obj.Wir*u + obj.Wrr*obj.state + ...
                            obj.Wor*y_fb + obj.Wbr + obj.noise*randn(obj.nReservoir, 1));
                        obj.state = (1 - obj.leakRate)*obj.state + obj.leakRate*x;
                        obj.lastOutput = targetSequence(:, t);

                        if ismember(t, idx_val)
                            X_val(:, t_index) = obj.state;
                            t_index = t_index + 1;
                        end
                    end

                    X_val_aug = [X_val; ones(1, size(X_val, 2))];
                    Y_pred_val = theta * X_val_aug;
                    fold_errors(fold) = mean((Y_val - Y_pred_val).^2, 'all');
                end

                mean_error = mean(fold_errors);
                if mean_error < best_error
                    best_error = mean_error;
                    best_reg = reg;
                end
            end

            % Treinamento final com melhor lambda encontrado
            obj.train(inputSequence, targetSequence, warmup, best_reg);
        end

        function y = predict(obj, u)
            % Predição de uma única amostra de entrada
            x = tanh(obj.Wir*u + obj.Wrr*obj.state + ...
                obj.Wor*obj.lastOutput + obj.Wbr);
            obj.state = (1 - obj.leakRate)*obj.state + obj.leakRate*x;
            y = obj.Wro*[obj.state; 1];
            obj.lastOutput = y;  % realimentação
        end

        function resetState(obj)
            % Reinicia o estado interno do reservatório
            obj.state = zeros(obj.nReservoir, 1);
        end
    end
end
