clear all;
clc;
close all;





n_amostras = 10000;
R = 0.034;
L = 0.175;
T = 0.1;



x0 = [0;0;0];


att_state = @(x,u) T*(R/2)*[cos(x(3)), cos(x(3)); sin(x(3)), sin(x(3)); 1/L, -1/L]*u + x;

uu = [0;0];

t = 1;

while t < n_amostras

    duracao = randi([5, 500]);
    u_const = 6*rand(2,1) - 3;
    
    tamanho_uu = size(uu);
    uu = [uu, repmat(u_const,1,min(duracao, n_amostras-t))];  
    t = t+duracao;
end


xx = [];

xx = [xx; x0'];
for i = 1:n_amostras
    u = uu(:,i);
    % x0 = att_state(x0,u) + 0.01*randn(3,1);
    x0 = att_state(x0,u);
    xx = [xx; x0'];
end

X = [xx(1:end-1,:), uu']';
Y = xx(2:end,:)';

save('dados_virtuais_DDMR', "X", "Y", "T");


