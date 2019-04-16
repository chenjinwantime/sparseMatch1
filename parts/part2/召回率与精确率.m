


x = xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','A2:A6')';
y_leuven_recall=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','H2:H6')';
plot(x,y_leuven_recall);

y_leuven_presicion=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','I2:I6')';
hold on;
plot(x,y_leuven_presicion);



y_trees_recall=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','H9:H13')';
hold on;
plot(x,y_trees_recall);

y_trees_presicion=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','I9:I13')';
hold on;
plot(x,y_trees_presicion);


y_ubc_recall=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','H17:H21')';
hold on;
plot(x,y_ubc_recall);

y_ubc_presicion=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','I17:I21')';
hold on;
plot(x,y_ubc_presicion);



x_gms = xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','A2:A6')';
y_leuven_recall_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','H2:H6')';
hold on
plot(x_gms,y_leuven_recall_gms);

y_leuven_presicion_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','I2:I6')';
hold on;
plot(x_gms,y_leuven_presicion_gms);



y_trees_recall_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','H9:H13')';
hold on;
plot(x_gms,y_trees_recall_gms);

y_trees_presicion_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','I9:I13')';
hold on;
plot(x_gms,y_trees_presicion_gms);


y_ubc_recall_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','H17:H21')';
hold on;
plot(x_gms,y_ubc_recall_gms);

y_ubc_presicion_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','I17:I21')';
hold on;
plot(x_gms,y_ubc_presicion_gms);









x = xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','A2:A6')';
x_gms = xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','A2:A6')';
y_leuven_recall=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','H2:H6')';
plot(x,y_leuven_recall);
y_leuven_recall_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','H2:H6')';
hold on
ax1=plot(x_gms,y_leuven_recall_gms);
title('recall')
legend('guide','gms');
%ylim(ax1,[0 1]);
axis([-inf inf 0 1])

y_leuven_presicion=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','I2:I6')';
figure;
plot(x,y_leuven_presicion);
y_leuven_presicion_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','I2:I6')';
hold on;
ax2=plot(x_gms,y_leuven_presicion_gms);
title('presicion')
legend('guide','gms');
%ylim(ax2,[0 1]);
axis([-inf inf 0 1])


y_trees_recall=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','H9:H13')';
figure;
plot(x,y_trees_recall);
y_trees_recall_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','H9:H13')';
hold on;
ax3=plot(x_gms,y_trees_recall_gms);
title('recall')
legend('guide','gms');
%ylim(ax3,[0 1]);
axis([-inf inf 0 1]) 

y_trees_presicion=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','I9:I13')';
figure;
plot(x,y_trees_presicion);
y_trees_presicion_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','I9:I13')';
hold on;
ax4=plot(x_gms,y_trees_presicion_gms);
title('presicion')
legend('guide','gms');
%ylim(ax4,[0 1]);
axis([-inf inf 0 1]) 


y_ubc_recall=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','H17:H21')';
figure;
plot(x,y_ubc_recall);
y_ubc_recall_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','H17:H21')';
hold on;
ax5=plot(x_gms,y_ubc_recall_gms);
title('recall')
legend('guide','gms');
%ylim(ax5,[0 1]);
axis([-inf inf 0 1]) 


y_ubc_presicion=xlsread('E:\algorithm\parts\数据表_part2.xls','sheet1','I17:I21')';
figure;
ax6=plot(x,y_ubc_presicion);
y_ubc_presicion_gms=xlsread('E:\algorithm\parts\数据表_gms.xls','sheet1','I17:I21')';
hold on;
plot(x_gms,y_ubc_presicion_gms);
title('presicion')
legend('guide','gms');

%ylim(ax6,[0 1]);
axis([-inf inf 0 1]) 





















