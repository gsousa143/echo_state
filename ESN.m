classdef ESN < handle

    properties

        nInput, nReservoir, nOutput

        spectralRadius = 0.9

        inputScaling = 1.0

        biasScaling = 0.1

        leakRate = 0.5

        density = 0.1

        noise = 1e-5

        feedbackScaling = 0

        Wir, Wbr, Wrr, Wor, Wro

        state, lastOutput

        cum_data_input, cum_data_output

    end


    methods

        function obj = ESN(nInput, nOutput, nReservoir, varargin)

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

            obj.Wir = (rand(nReservoir, nInput)*2 - 1)*obj.inputScaling;

            obj.Wbr = (rand(nReservoir, 1)*2 - 1)*obj.biasScaling;

            obj.Wrr = sprand(nReservoir, nReservoir, obj.density)*2 - 1;

            obj.Wor = (rand(nReservoir, nOutput)*2 - 1)*obj.feedbackScaling;

            lambda = abs(eigs(obj.Wrr, 1, 'LM'));

            if lambda ~= 0

                obj.Wrr = (obj.Wrr / lambda)*obj.spectralRadius;

            end

            obj.resetState();

            obj.cum_data_input = [];

            obj.cum_data_output = [];

        end


        function [best_error, best_reg] = train_cv(obj, min_reg, max_reg, tests, folds)

            if nargin < 4, tests = 50; end

            if nargin < 5, folds = 10; end

            if isempty(obj.cum_data_input)

                error("Nenhum dado acumulado. Use add_data primeiro.");

            end

            reg_list = linspace(min_reg, max_reg, tests);

            n_samples = size(obj.cum_data_input, 2);

            indices = crossvalind('Kfold', n_samples, folds);

            best_error = inf;

            best_reg = reg_list(1);

            for i = 1:tests

                reg = reg_list(i);

                fold_errors = zeros(1, folds);

                for fold = 1:folds

                    idx_val = (indices == fold);

                    idx_train = ~idx_val;

                    X_train = obj.cum_data_input(:, idx_train);

                    Y_train = obj.cum_data_output(:, idx_train);

                    X_val = obj.cum_data_input(:, idx_val);

                    Y_val = obj.cum_data_output(:, idx_val);

                    theta = (Y_train * X_train') / (X_train * X_train' + reg * eye(size(X_train, 1)));

                    Y_pred_val = theta * X_val;

                    fold_errors(fold) = mean((Y_val - Y_pred_val).^2, 'all');

                end

                mean_error = mean(fold_errors);

                if mean_error < best_error

                    best_error = mean_error;

                    best_reg = reg;

                end

            end

            X = obj.cum_data_input;

            Y = obj.cum_data_output;

            obj.Wro = (Y * X') / (X * X' + best_reg * eye(size(X,1)));

            obj.resetState();

        end


        function y = predict(obj, u)

            x = tanh(obj.Wir*u + obj.Wrr*obj.state + obj.Wor*obj.lastOutput + obj.Wbr);

            obj.state = (1 - obj.leakRate)*obj.state + obj.leakRate*x;

            y = obj.Wro*[obj.state; 1];

            obj.lastOutput = y;

        end


        function resetState(obj)

            obj.state = zeros(obj.nReservoir, 1);

            obj.lastOutput = zeros(obj.nOutput, 1);

        end


        function add_data(obj, input_data, output_data, warmup)

            if nargin < 4, warmup = 0; end

            T = size(input_data, 2);

            if T ~= size(output_data, 2)

                error('Números de amostras de entrada e saída não batem.');

            end

            X = zeros(obj.nReservoir, T - warmup);

            Y = output_data(:, warmup+1:end);

            obj.resetState();


            for t = 1:T

                u = input_data(:, t);

                x = tanh(obj.Wir*u + obj.Wrr*obj.state + obj.Wor*obj.lastOutput + obj.Wbr + obj.noise*randn(obj.nReservoir, 1));

                obj.state = (1 - obj.leakRate)*obj.state + obj.leakRate*x;

                if t > warmup

                    X(:, t - warmup) = obj.state;

                end

                obj.lastOutput = output_data(:, t);

            end

            X_aug = [X; ones(1, size(X, 2))];

            if isempty(obj.cum_data_input)

                obj.cum_data_input = X_aug;

                obj.cum_data_output = Y;

            else

                obj.cum_data_input = [obj.cum_data_input, X_aug];

                obj.cum_data_output = [obj.cum_data_output, Y];

            end

        end

        % >>>>>>>>>>>>>> NOVO MÉTODO SAVE <<<<<<<<<<<<<<

        function save(obj, filename, varargin)

            % Salva o estado do objeto ESN em um arquivo .mat

            p = inputParser;

            addParameter(p, 'IncludeData', false, @islogical);

            parse(p, varargin{:});


            saved_obj = struct();


            % Salva todas as propriedades

            props = properties(obj);

            for i = 1:length(props)

                prop_name = props{i};

                % Não salva dados acumulados a menos que solicitado

                if ~p.Results.IncludeData && (strcmp(prop_name, 'cum_data_input') || strcmp(prop_name, 'cum_data_output'))

                    saved_obj.(prop_name) = [];

                else

                    saved_obj.(prop_name) = obj.(prop_name);

                end

            end


            save(filename, '-struct', 'saved_obj');

            fprintf('ESN salva em %s\n', filename);

        end

    end


    methods (Static)

        % >>>>>>>>>>>>>> NOVO MÉTODO LOAD <<<<<<<<<<<<<<

        function obj = load(filename)

            % Carrega um objeto ESN de um arquivo .mat

            % Este é um método estático porque cria uma nova instância.

            if ~isfile(filename)

                error('Arquivo não encontrado: %s', filename);

            end


            S = load(filename);


            % Verifica se as dimensões necessárias existem

            required_fields = {'nInput', 'nReservoir', 'nOutput'};

            for i = 1:length(required_fields)

                if ~isfield(S, required_fields{i})

                    error('Arquivo .mat inválido. Faltando propriedade: %s', required_fields{i});

                end

            end


            % Cria uma nova instância

            obj = ESN(S.nInput, S.nOutput, S.nReservoir);


            % Carrega todas as propriedades salvas no objeto

            props = fieldnames(S);

            for i = 1:length(props)

                prop_name = props{i};

                if isprop(obj, prop_name)

                    obj.(prop_name) = S.(prop_name);

                end

            end

            fprintf('ESN carregada de %s\n', filename);

        end

    end

end
