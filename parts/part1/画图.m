
c = categorical({'vote','brief','orb'});
%bikes
y_bikes = [
    3.4,1.1,3.4,1.1,0.01,0.67;
    0,0,3.4,1.1,0,20.8;
    3.4,1,0,0,0,17.2;
    ];
    
y_boat=[
	5.8,	4.2,6.4,	4.4,0.01	0.71;
    0,0, 5.8,	4.2	,0,51.4;
    6.4,4.4,0,0,0,51.8;
]; 
y_graf=[
	
	5.7,	3.2,3.7,	2,0.01	2.13;
    0,0,5.7,	3.2,	0,51.2;
    3.7,	2,0,0,	0,50.5;
];
y_leuven=[	
	3.3,	1.3,3.6,	1.3,0.001	0.64;
    0,0,3.6,	1.3,	0,5.30;
    3.3,	1.3,0,0,	0,6.6;
];

figure;
bikes_handle=subplot(2,2,1); 
barh(y_bikes,'stacked'); % stacks values in each row together

title('bikes')


subplot(2,2,2); 
boat_handle=barh(y_boat,'stacked'); % stacks values in each row together
title('boat')


subplot(2,2,3); 
graf_handle=barh(y_graf,'stacked'); % stacks values in each row together
title('graf')




subplot(2,2,4);    
leuven_handle=barh(y_leuven,'stacked'); % stacks values in each row together
title('leuven') 
%set(leuven_handle,'YTick',[1 2 3],'YTickLabel',{'proposed','orb','brief'});
























