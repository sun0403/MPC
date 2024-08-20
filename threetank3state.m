function [model] = threetank3state(time)
import casadi.*
model = NosnocModel();
%% Inital value
%model.x0 = [11;5;7];
%model.u0 = [4;7]; % guess for control variables
tx11 = 20;
tx21 = 16;
tx31 = 34;
tx12 = 23;
tx22 = 20;
tx32 = 30;
tchange = 100;
x_target=[tx11 - tx11*rump(time-tchange) + tx12*rump(time-tchange) ;
    tx21 - tx21*rump(time-tchange) + tx22*rump(time-tchange); 
    tx31 - tx31*rump(time-tchange) + tx32*rump(time-tchange)];
%% Variable defintion
% differential states
x1 = SX.sym('x1');
x2 = SX.sym('x2');
x3 = SX.sym('x3');
x = [x1;x2;x3];
model.x = x;
n_x = 3;

% lower and upper bounds
model.lbx = 1e-3*ones(n_x,1);
model.ubx = 100*ones(n_x,1);

%% Control
u1 = SX.sym('u1');
u2 = SX.sym('u2');
%u2 = SX.sym('u2');
u = [u1,u2];
model.u = u;
n_u = length(model.u);
% Guess and Bounds
model.lbu  = 0.1*ones(n_u,1);
model.ubu  = 100*ones(n_u,1);
% Parameters

q_c1 = 0.285;
q_c2 = 0.285;
q_o1 = 0.228;
q_o2 = 0.228;
q_o3 = 0.228;
A = 153.94;
g = 981;
p_1 = 0.75;
p_2 = 1;
%% Switching Functions
% every constraint funcion corresponds to a simplex (note that the c_i might be vector valued)
c1 = x1 - x2;
c2 = x2 - x3;
% sign matrix for the modes
S1 = [1; -1];
S2 = [1; -1];
% c = [c1;c2];
model.c = {c1,c2};
model.S = {S1,S2};

F_input = 0;
error = 1e-6;
f_1 = [1/A * (u1 - p_1 * q_o1 * sqrt(2 * g * x1) - q_c1 * sqrt(max(2 * g * ( x1 - x2 ),error)));
    1/A * ( - p_2 * q_o2 * sqrt(2 * g * x2) + q_c1 * sqrt(max(2 * g * ( x1 - x2),error))); 
    1/A * (u2 - q_o3 * sqrt(2 * g * x3))];
f_2 = [1/A * (u1 - p_1 * q_o1 * sqrt(2 * g * x1) + q_c1 * sqrt(max(2 * g * ( x2 - x1 ),error))); 
    1/A * ( - p_2 * q_o2 * sqrt(2 * g * x2) - q_c1 * sqrt(max(2 * g * ( x2 - x1),error))); 
    1/A * (u2 - q_o3 * sqrt(2 * g * x3))];
f_3 = [1/A * ( u1- p_1 * q_o1 * sqrt(2 * g * x1));
    1/A * ( - p_2 * q_o2 * sqrt(2 * g * x2)  - q_c2 * sqrt(max(2 * g * (x2 -x3), error))); 
    1/A * (u2 - q_o3 * sqrt(2 * g * x3) + q_c2 * sqrt(max(2 * g * (x2 - x3),error)))];
f_4 = [1/A * ( u1- p_1 * q_o1 * sqrt(2 * g * x1));
    1/A * ( - p_2 * q_o2 * sqrt(2 * g * x2) + q_c2 * sqrt(max(2 * g * (- x2 + x3), error))); 
      1/A * (u2 - q_o3 * sqrt(2 * g * x3) - q_c2 * sqrt(max(2 * g * (- x2 + x3),error)))];
F1 = [f_1 f_2];
F2 = [f_3 f_4];
% in matrix form
model.F = {F1,F2};

%% Objective
model.f_q =(x-x_target)'*(x-x_target);
model.f_q_T = 0;

end

function result = rump(x)
    if x<0
        result = 0;
    end
    if x>0 || x == 0
        result = 1;
    end
end

     
