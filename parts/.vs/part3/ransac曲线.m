clc
%画图
syms a n
p='0=1-(1-a^4)^n-a^4*n*(1-a^4)^(n-1)-0.95'
h_ransac=ezplot(p,[0.1,1,1,1000]);
%set(h_ransac,'Color','r','LineStyle','-.');

%画图
hold on 
syms a n
p1='0=1-(1-a^4)^n-0.95'
h_improved=ezplot(p1,[0.1,1,1,1000]);



c_ransac=get(h_ransac,'contourMatrix');
x_ransac=c_ransac(1,2:400);
y_ransac=c_ransac(2,2:400);
%figure
%plot(x_ransac,y_ransac)


c_improved=get(h_improved,'contourMatrix');
x_improved=c_improved(1,2:400);
y_improved=c_improved(2,2:400);
hold on 
%plot(x_improved,y_improved)


x=linspace(0,1,100);
y1=interp1(x_ransac,y_ransac,x);
y2=interp1(x_improved,y_improved,x);
y=y1-y2;
%figure
%plot(x,y1);
%hold on
%plot(x,y2);
hold on
plot(x,y1-y2);




hold on 
syms a n
p1='0=1-(1-a^4)^n-0.95'
h1=ezplot(p1,[0.1,1,1,1000])







hold on 
x=get(h,'XData');
y=get(h,'YData');
z=get(h,'ZData');
y1=get(h1,'YData');
plot(x,y-y1)



hold on 
syms a n
p='0=1-(1-a^4)^n-a^4*n*(1-a^4)^(n-1)-0.95'
ezplot(p,[0.1,1,1,1000])


h1=ezplot(p,[0.1,1,1,1000]);
set(h1,'Color','r');


hold on 
syms a n
p2='0=-a^4*n*(1-a^4)^(n-1)'
ezplot(p2,[0.1,1,1,1000])

clc
syms a n
p='0=n-1000*a-10'
h=ezplot(p,[0.1,1,0,2000])
x=get(h,'XData')
y=get(h,'YData')



x=[-pi:0.01:pi];
y=sin(x);
h=plot(x,y)


x=[-pi:0.01:pi];
fun='sin(x)-y';
h=ezplot(fun)