

x = xlsread('E:\algorithm\parts\parts\时间.xls','sheet1','A2:A7')';

y_proposed=xlsread('E:\algorithm\parts\parts\时间.xls','sheet1',' C2:C7')';
%plot(x,y,'Marker','^','Color','r');
%hold on
plot(x+0.5,y_proposed,'LineWidth',3,'MarkerSize',10,'Marker','^','Color',[0.968627452850342 0.588235318660736 0.274509817361832]);
ylim([15 690])
%xticklabels({'drawing2','airport','waterCubic_inside7','waterCubic_inside2','notredame15','playground'})
xticks([1 2 3 4 5 6])

set(gca,'xticklabel',{'drawing2','airport','waterCubic7','waterCubic2','notredame15','playground'})
xtb = get(gca,'XTickLabel');   % 获取横坐标轴标签句柄
xt = get(gca,'XTick');   % 获取横坐标轴刻度句柄
yt = get(gca,'YTick');    % 获取纵坐标轴刻度句柄          
xtextp=xt;    %每个标签放置位置的横坐标，这个自然应该和原来的一样了。
% 设置显示标签的位置，写法不唯一，这里其实是在为每个标签找放置位置的纵坐标                     
ytextp=yt(1)*ones(1,length(xt)); 

% rotation，正的旋转角度代表逆时针旋转，旋转轴可以由HorizontalAlignment属性来设定，
% 有3个属性值：left，right，center，这里可以改这三个值，以及rotation后的角度，这里写的是45
% 不同的角度对应不同的旋转位置了，依自己的需求而定了。
% ytextp - 0.5是让标签稍微下一移一点，显得不那么紧凑
text(xtextp+0.5,ytextp-100,xtb,'HorizontalAlignment','right','rotation',20,'fontsize',10); 
set(gca,'xticklabel','');% 将原有的标签隐去
xlim([1 7])
title({'求解单应矩阵运行时间比较'});
ylabel({'时间(ms)'});

hold on 
y_gms=xlsread('E:\algorithm\parts\parts\时间.xls','sheet1',' F2:F7')';
plot(x+0.5,y_gms,'LineWidth',3,'MarkerSize',10,'Marker','*','Color',[0.752941191196442 0.313725501298904 0.301960796117783]);

hold on 
y_orb=xlsread('E:\algorithm\parts\parts\时间.xls','sheet1',' I2:I7')';
plot(x+0.5,y_orb,'LineWidth',3,'MarkerSize',10,'Marker','s','Color',[0.607843160629272 0.733333349227905 0.34901961684227]);

hold on 
y_sift=xlsread('E:\algorithm\parts\parts\时间.xls','sheet1',' L2:L7')';
plot(x+0.5,y_sift,'LineWidth',3,'MarkerSize',10,'Marker','d','Color',[0.309803932905197 0.501960813999176 0.74117648601532]);

legend('proposed','gms','orb','sift')
