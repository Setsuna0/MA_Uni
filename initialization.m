%%TRAINING
V_belt=0.05; %m/s
Belt_length=0.5; % m
Distance_between=0.05; %m Minimum distance between entities
Entity_length=0.05; %m

T_process=5; % second
T_switch=1; % second
num=5; % number of product
T_interval=2; % second

Ts = 1; %s the sample time
T = 1500;%s simulation duration

mdl = 'training_model_all';
open_system(mdl);
agentblk = [mdl '/RL Agent'];

% create observation info
observationInfo = rlNumericSpec([6 1],'LowerLimit',-inf*ones(6,1),'UpperLimit',inf*ones(6,1));
observationInfo.Name = 'observations';
observationInfo.Description = 'information on belts and entity';
% create action Info
actionInfo = rlFiniteSetSpec([1 2]);
actionInfo.Name = 'output_belt_num';
% define environment
env = rlSimulinkEnv(mdl,agentblk,observationInfo,actionInfo);

%env.ResetFcn = @(in)localResetFcn(in);
%rng(0);

L = 24; % number of neurons
statePath = [
    imageInputLayer([6 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(L,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(L,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(L,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];

actionPath = [
    imageInputLayer([1 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(L,'Name','fc5')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);    
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');

figure
plot(criticNetwork)

criticOptions = rlRepresentationOptions('LearnRate',1e-3,'GradientThreshold',1,'L2RegularizationFactor',1e-4);
%1e-3 1e-4
critic = rlRepresentation(criticNetwork,observationInfo,actionInfo,...
    'Observation',{'state'},'Action',{'action'},criticOptions);

agentOptions = rlDQNAgentOptions(...
    'SampleTime',Ts,...
    'UseDoubleDQN',true,...
    'TargetSmoothFactor',1e-3,...
    'DiscountFactor',0.99,...
    'ExperienceBufferLength',1e6,...
    'MiniBatchSize',64);

agent = rlDQNAgent(critic,agentOptions);

function in = localResetFcn(in)
% reset
%in = setVariable(in,'num', 5+5*rand); % random value for lateral deviation
%in = setVariable(in,'T_interval',2+3*rand); % random value for relative yaw angle
end