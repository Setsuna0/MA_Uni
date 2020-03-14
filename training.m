tStart1=tic;
C=zeros(100,3);
result=zeros(30,300);
n=1;
for i=1:3
    if i==1
        A=0;
        B=1;
    elseif i==2
        A=1;
        B=0;
    else
        A=0.5;
        B=0.5;
    end
    for m=1:100
        maxepisodes =1;
        maxsteps = ceil(T/Ts);
        trainingOpts = rlTrainingOptions(...
        'MaxEpisodes',maxepisodes,...
        'MaxStepsPerEpisode',maxsteps,...
        'Verbose',false,...
        'Plots','none',...
        'StopTrainingCriteria','EpisodeCount',...
        'StopTrainingValue',maxepisodes,...
        'SaveAgentCriteria','EpisodeCount',...
        'SaveAgentValue',maxepisodes);  
         num=5;
         trainingStats = train(agent,env,trainingOpts);
         num=15;
         sim('model_test');
         result(1:length(ans.Arrivaltime.time),n)=ans.Arrivaltime.time;
         [M,N]=find(result(:,n)~=0);
         if ~isempty(M)
         C(m,i)=sum(result(M(1):M(end),n))/M(end);
         success_rate(m,i)=M(end)/30*100;
         D(m,i)=C(m,i)/success_rate(m,i)*100;
         end
         n=n+1;
    end
end
t_simulation=toc(tStart1);
% tStart1=tic;
% maxepisodes = 100;
% maxsteps = ceil(T/Ts);
% trainingOpts = rlTrainingOptions(...
%     'MaxEpisodes',maxepisodes,...
%     'MaxStepsPerEpisode',maxsteps,...
%     'Verbose',false,...
%     'Plots','training-progress',...
%     'StopTrainingCriteria','EpisodeCount',...
%     'StopTrainingValue',maxepisodes,...
%     'SaveAgentCriteria','EpisodeCount',...
%     'SaveAgentValue',maxepisodes);
% 
% doTraining = true;
% A=1;
% B=0;
% if doTraining    
%     % Train the agent.
%     trainingStats = train(agent,env,trainingOpts);
% %    save("Agents_switch3_0.5_0.5.mat","agent");
% else
%     % Load pretrained agent for the example.
%     load('initialAgents_switch3.mat','agent')       
% end
% t_training=toc(tStart1);