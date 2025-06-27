classdef ESN_galdir < handle
    % ESN (Echo State Network) - Rede Neural de Estado de Eco
    % Esta classe implementa uma rede neural de estado de eco para modelagem de series temporais
    
    properties
        neu             % numero de neuronios
        n_in            % numero de entradas
        n_out           % numero de saidas
        psi             % fator de esparsidade
        leakrate        % taxa de vazamento
        ro              % raio espectral
        in_scale        % escala de entrada
        bias_scale      % escala do bias
        alfa            % condicao inicial do algoritmo RLS
        forget          % fator de esquecimento
        output_feedback % feedback de saida
        noise           % amplitude do ruido
        
        % Matrizes de peso
        Wrr  % matriz de pesos do reservatorio
        Wir  % matriz de pesos entrada-reservatorio
        Wbr  % matriz de pesos bias-reservatorio
        Wor  % matriz de pesos saida-reservatorio
        Wro  % matriz de pesos reservatorio-saida
        
        % Estado do reservatorio
        a    % estado atual
        P    % matriz de covariancia
        
        % Dados de treinamento
        cum_data_input        % dados de entrada acumulados
        cum_data_output       % dados de saida acumulados
        number_of_simulations % numero de simulacoes
        simulations_start     % lista que rastreia cada inicio de simulacao
        default_warmupdrop    % warmupdrop padrao
    end
    
    methods
        function obj = ESN_galdir(neu, n_in, n_out, varargin)
            % Construtor da ESN
            % Parametros:
            %   neu: numero de neuronios
            %   n_in: numero de entradas
            %   n_out: numero de saidas
            %   Parametros opcionais (nome-valor):
            %     'leakrate': taxa de vazamento (default: 0.5)
            %     'ro': raio espectral (default: 1)
            %     'psi': esparsidade (default: 0.5)
            %     'in_scale': escala de entrada (default: 0.1)
            %     'bias_scale': escala do bias (default: 0.5)
            %     'alfa': valor inicial RLS (default: 10)
            %     'forget': fator de esquecimento (default: 1)
            %     'output_feedback': feedback de saida (default: false)
            %     'noise_amplitude': amplitude do ruido (default: 0)
            %     'out_scale': escala da saida (default: 0)
            %     'default_warmupdrop': amostras descartadas (default: 100)
            
            p = inputParser;
            addParameter(p, 'leakrate', 0.5);
            addParameter(p, 'ro', 1);
            addParameter(p, 'psi', 0.5);
            addParameter(p, 'in_scale', 0.1);
            addParameter(p, 'bias_scale', 0.5);
            addParameter(p, 'alfa', 10);
            addParameter(p, 'forget', 1);
            addParameter(p, 'output_feedback', false);
            addParameter(p, 'noise_amplitude', 0);
            addParameter(p, 'out_scale', 0);
            addParameter(p, 'default_warmupdrop', 0);
            
            parse(p, varargin{:});
            
            % Inicializacao das propriedades basicas
            obj.neu = neu;
            obj.n_in = n_in;
            obj.n_out = n_out;
            obj.psi = p.Results.psi;
            
            % Inicializacao das matrizes com distribuicao normal
            obj.Wrr = obj.sparsidade(randn(neu, neu), obj.psi);
            obj.Wir = randn(neu, n_in);
            obj.Wbr = randn(neu, 1);
            obj.Wor = randn(neu, n_out);
            obj.Wro = randn(n_out, neu + 1);
            
            % Configuracao dos parametros
            obj.leakrate = p.Results.leakrate;
            obj.ro = p.Results.ro;
            obj.in_scale = p.Results.in_scale;
            obj.bias_scale = p.Results.bias_scale;
            obj.alfa = p.Results.alfa;
            obj.forget = p.Results.forget;
            obj.output_feedback = p.Results.output_feedback;
            obj.noise = p.Results.noise_amplitude;
            
            % Normalizacao do raio espectral
            eigs_val = eig(obj.Wrr);
            radius = max(abs(eigs_val));
            obj.Wrr = obj.Wrr/radius * obj.ro;
            
            % Escalonamento das matrizes
            obj.Wbr = obj.bias_scale * obj.Wbr;
            obj.Wir = obj.in_scale * obj.Wir;
            obj.Wor = p.Results.out_scale * obj.Wor;
            
            % Inicializacao do estado
            obj.a = zeros(neu, 1);
            
            % Matriz de covariancia
            obj.P = eye(neu+1)/obj.alfa;
            
            % Variaveis de acumulacao de dados
            obj.cum_data_input = [];
            obj.cum_data_output = [];
            obj.number_of_simulations = 0;
            obj.simulations_start = [];
            obj.default_warmupdrop = p.Results.default_warmupdrop;
        end
        
        function y = update(obj, inp, y_in, training)
            % Atualiza o estado da rede e retorna a saida
            % Parametros:
            %   inp: entrada da rede (n_in x 1)
            %   y_in: feedback da saida (opcional)
            %   training: modo de treinamento (opcional)
            
            % Verificacao e preparacao dos argumentos
            if nargin < 4
                training = false;
            end
            if nargin < 3 || all(y_in == 0)
                y_in = zeros(obj.n_out, 1);
            end
            
            % Garante formato correto dos dados
            inp = reshape(inp, [], 1);
            y_in = reshape(y_in, [], 1);
            
            if numel(inp) ~= obj.n_in
                error('A entrada deve ter tamanho n_in (%d), mas tem tamanho %d', obj.n_in, numel(inp));
            end
            
            % Calculo do estado
            z = obj.Wrr * obj.a + obj.Wir * inp + obj.Wbr;
            if obj.output_feedback
                z = z + obj.Wor * y_in;
            end
            if obj.noise > 0 && training
                z = z + obj.noise * randn(obj.neu, 1);
            end
            
            % Atualizacao do estado
            obj.a = (1-obj.leakrate) * obj.a + obj.leakrate * tanh(z);
            
            % Calculo da saida
            a_wbias = [1; obj.a];
            y = obj.Wro * a_wbias;
        end
        
        function add_data(obj, input_data, output_data, warmupdrop)
            % Adiciona dados de treinamento C  rede
            % Parametros:
            %   input_data: matriz de dados de entrada (amostras x n_in)
            %   output_data: matriz de dados de saida (amostras x n_out)
            %   warmupdrop: numero de amostras iniciais a descartar
            
            if nargin < 4
                warmupdrop = obj.default_warmupdrop;
            end
            
            % Verifica dimenscoes
            [n_samples, n_inputs] = size(input_data);
            if n_inputs ~= obj.n_in
                error('Numero incorreto de entradas: esperado %d, recebido %d', obj.n_in, n_inputs);
            end
            
            % Registra inicio da nova simulacao
            if isempty(obj.simulations_start)
                obj.simulations_start = 0;
            else
                obj.simulations_start(end+1) = size(obj.cum_data_input, 1);
            end
            
            % Coleta estados do reservatorio
            A = zeros(n_samples-warmupdrop, obj.neu);
            for i = 1:n_samples
                %if obj.output_feedback && i > 1
                %    obj.update(input_data(i,:)', output_data(i-1,:)', true);
                %else
                    obj.update(input_data(i,:)', [], true);
                %end
                
                if i > warmupdrop
                    A(i-warmupdrop,:) = obj.a';
                end
            end
            
            % Acumula dados
            if isempty(obj.cum_data_input)
                obj.cum_data_input = A;
                obj.cum_data_output = output_data(warmupdrop+1:end,:);
            else
                obj.cum_data_input = [obj.cum_data_input; A];
                obj.cum_data_output = [obj.cum_data_output; output_data(warmupdrop+1:end,:)];
            end
            
            obj.number_of_simulations = obj.number_of_simulations + 1;
        end
        
        function [best_error, best_reg] = cum_train_cv(obj, min_reg, max_reg, tests, folds)
            % Treina a rede usando validacao cruzada
            % Parametros:
            %   min_reg: valor minimo de regularizacao
            %   max_reg: valor maximo de regularizacao
            %   tests: numero de testes (default: 50)
            %   folds: numero de particoes (default: 10)
            
            if nargin < 4
                tests = 50;
            end
            if nargin < 5
                folds = 10;
            end
            
            if min_reg > max_reg
                error('min_reg deve ser menor que max_reg');
            end
            
            reg_list = linspace(min_reg, max_reg, tests);
            n_samples = size(obj.cum_data_input, 1);
            val_size = floor(n_samples/folds);
            
            % Adiciona bias aos dados
            A_wbias = [ones(n_samples,1) obj.cum_data_input];
            
            best_error = inf;
            best_reg = 0;
            
            % Loop principal da validacao cruzada
            for i = 1:length(reg_list)
                regularization = reg_list(i);
                fold_errors = zeros(1, folds);
                
                for fold = 1:folds
                    % Indices para validacao
                    val_start = (fold-1)*val_size + 1;
                    if fold == folds
                        val_end = n_samples;
                    else
                        val_end = fold*val_size;
                    end
                    val_idx = val_start:val_end;
                    train_idx = setdiff(1:n_samples, val_idx);
                    
                    % Separa dados de treino e validacao
                    training_A = A_wbias(train_idx,:);
                    training_y = obj.cum_data_output(train_idx,:);
                    cv_A = A_wbias(val_idx,:);
                    cv_y = obj.cum_data_output(val_idx,:);
                    
                    % Treina e avalia
                    theta = obj.reg_minimos_quadrados(training_A, training_y, regularization);
                    pred_y = cv_A * theta;
                    fold_errors(fold) = mean((cv_y - pred_y).^2, 'all');
                end
                
                mean_error = mean(fold_errors);
                if mean_error < best_error
                    best_error = mean_error;
                    best_reg = regularization;
                end
            end
            
            % Treinamento final com melhor regularizacao
            obj.Wro = obj.reg_minimos_quadrados(A_wbias, obj.cum_data_output, best_reg)';
        end
        
        function save_reservoir(obj, fileName)
            % Salva o reservatorio em arquivo .mat
            % Parametros:
            %   fileName: nome do arquivo
            
            data = struct();
            data.Wrr = obj.Wrr;
            data.Wir = obj.Wir;
            data.Wbr = obj.Wbr;
            data.Wro = obj.Wro;
            data.a0 = obj.a;
            data.leakrate = obj.leakrate;
            save(fileName, 'data');
        end
        
        function load_reservoir(obj, fileName)
            % Carrega o reservatorio de um arquivo .mat
            % Parametros:
            %   fileName: nome do arquivo
            
            data = load(fileName);
            obj.load_reservoir_from_struct(data.data);
        end
        
        function load_reservoir_from_struct(obj, data)
            % Carrega o reservatorio a partir de uma estrutura
            % Parametros:
            %   data: estrutura com os dados do reservatorio
            
            % Carrega matrizes principais
            if isfield(data, 'Wrr')
                obj.Wrr = data.Wrr;
            end
            if isfield(data, 'Wir')
                obj.Wir = data.Wir;
            end
            if isfield(data, 'Wbr')
                obj.Wbr = data.Wbr;
            end
            if isfield(data, 'Wro')
                obj.Wro = data.Wro;
            end
            
            % Carrega estado inicial se disponivel
            if isfield(data, 'a0')
                obj.a = data.a0;
            end
            
            % Carrega bias se disponivel
            if isfield(data, 'Wro_b')
                obj.Wro = [data.Wro_b obj.Wro];
            end
            
            % Carrega taxa de vazamento
            if isfield(data, 'leakrate')
                if isstruct(data.leakrate)
                    obj.leakrate = data.leakrate(1,1);
                else
                    obj.leakrate = data.leakrate;
                end
            end
            
            % Atualiza dimenscoes baseado nas matrizes carregadas
            obj.neu = size(obj.Wrr, 1);
            obj.n_in = size(obj.Wir, 2);
            obj.n_out = size(obj.Wro, 1);
        end
    end
    
    methods (Access = private)
        function M_esparsa = sparsidade(~, M, psi)
            % Cria uma matriz esparsa
            % Parametros:
            %   M: matriz de entrada
            %   psi: fator de esparsidade
            
            N = zeros(size(M));
            for linha = 1:size(N,1)
                for coluna = 1:size(N,2)
                    if rand < psi
                        N(linha,coluna) = 0;
                    else
                        N(linha,coluna) = 1;
                    end
                end
            end
            M_esparsa = N .* M;
        end
        
        function theta = reg_minimos_quadrados(~, X, Y, reg)
            % Calcula os minimos quadrados regularizados
            % Parametros:
            %   X: matriz de caracteristicas
            %   Y: matriz de saidas
            %   reg: parametro de regularizacao
            
            P = X' * X;
            R = X' * Y;
            theta = (P + reg*eye(size(P))) \ R;
        end

        % Versao 2: Usando SVD (ainda mais estavel, mas mais lento)
        function theta = reg_minimos_quadrados_svd(~, X, Y, reg)
            % Calcula minimos quadrados regularizados usando SVD
            % Esta versao e a mais estavel numericamente, especialmente para matrizes mal condicionadas
            % X: matriz de caracteristicas
            % Y: matriz de saidas
            % reg: parametro de regularizacao
            
            [U, S, V] = svd(X, 'econ');
            s = diag(S);
            
            % Fatores de regularizacao para cada valor singular
            f = s ./ (s.^2 + reg);
            
            % Calcula theta usando SVD
            theta = V * (f .* (U' * Y));
        end
    end
end
