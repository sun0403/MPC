import casadi.*
clear all
clc 
close all
%% settings
problem_options = NosnocProblemOptions();
solver_options = NosnocSolverOptions();
problem_options.irk_scheme = IRKSchemes.GAUSS_LEGENDRE;   
problem_options.n_s = 1;                
solver_options.N_homotopy = 8;
solver_options.homotopy_update_rule = 'linear';
x0=[0.1;0.1;0.1];
u0=[0;0];
optimal_x=x0;
optimal_u=u0;
t0=0;
t=[t0];
cal_time = [];
numberofstages = 20;
numberofT = 120;
%% Discretization parameters
problem_options.N_stages = numberofstages; % number of control intervals
problem_options.N_finite_elements = 2; % number of finite element on every control interval
problem_options.T = numberofT;    % time horizon
%% Generate Model
while t0<3000

if t0>1000
    p2=0.9;
else
    p2=0.5;
end
%model = threetank3state(t0);
model = threetank3state(t0,p2);
model.x0 = x0;
model.u0 = u0;
%lag_factor=0.9;


%% Solve OCP via nosnoc
 tic;
 mpcc = NosnocMPCC(problem_options, model);
 solver = NosnocSolver(mpcc, solver_options);
 [results,stats] = solver.solve();
 time_consume = toc;
 cal_time=[cal_time time_consume];
 x0=results.x(:,3);
 %u0 = lag_factor * u0 + (1 - lag_factor) * results.u(:,3);
 u0 = results.u(:,3);

 optimal_x=[optimal_x,x0];
 optimal_u=[optimal_u,u0];
 t0=t0+numberofT/numberofstages;
 t=[t t0];
end
%% Time record
cal_time=[cal_time time_consume];
max_time = max(cal_time);
min_time = min(cal_time);
mean_time = mean(cal_time);
plus_time = max_time -mean_time;
minus_time = mean_time - min_time;
%% Read and plot Result 
figure
subplot(211)
plot(t,optimal_x(:,:));
legend('x1','x2','x3')
hold on
subplot(212)
plot(t,optimal_u(:,:));
legend('u1','u2')
hold on
figure;
plot(t, cal_time(:,:));
hold on;
