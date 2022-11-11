%%
clearvars
samples = 5;
k       = [8        1       16];
p       = [400      100     800];
mr      = [0.1      0.3     0.02];
linescolor = ['k','r','b'];
pos = 1;
figure();

%%

select_k = 1;
select_p = 1;
select_mr = 1;

ax = subplot(1,2,1);hold on;
ax2 = subplot(1,2,2);hold on;


for i = 1:samples
   
    mrname = num2str(mr(select_mr));
    
    % get the file name
    file = ['p' num2str(p(select_p)) '_k' num2str(k(select_k)) '_mr' mrname '_m20_' num2str(i) '.csv'];
    
    % read the csv
    data = csvread([file],2,0);
    
    iteration = data(:,1);
    elapsed = data(:,2);
    meanO = data(:,3);
    bestO = data(:,4);
    
    title(strrep(file,'_',' '))
    plot(ax, iteration, meanO, linescolor(pos));
    plot(ax, iteration, bestO, 'g')
    plot(ax2, pos, min(bestO), 'ko', 'markerfacecolor', linescolor(pos), 'markersize', 4);
    
    
end

pos = pos + 1;