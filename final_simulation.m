clear all;
close all;
x_curr=[-50; 50];
y_curr=[40;-40];




inti=zeros(2);
di=zeros(2);

t=3.5;
r=8/t;
g(1,:)=[50,-40];
g(2,:)=[-50,40];
error_pre=zeros(2);
%error_pre(2,:)=zeros(2);
ki=1e-5;
kd=0.001;
kp=.1;

%replace v1=[a1,b1]


error(1,:) =g(1,:)-[x_curr(1),y_curr(1)];

error(2,:) =g(2,:)-[x_curr(2),y_curr(2)];

plot(x_curr(1),y_curr(1),'r--*');hold on;
plot(x_curr(2),y_curr(2),'--o');hold on;

di(1,:)=error(1,:)-error_pre(1,:);
inti(1,:)+=error(1,:);
di(2,:)=error(2,:)-error_pre(2,:);
inti(2,:)+=error(2,:);

v_opt(1,:)= kp*error(1,:) + kd*di(1,:)+ ki*inti(1,:);
v_opt(2,:)=kp*error(2,:) + kd*di(2,:) + ki*inti(2,:);

v_pref=v_opt;

x_curr=v_opt(:,1)*t+x_curr;
y_curr=v_opt(:,2)*t+y_curr;

plot(x_curr(1),y_curr(1),'r*','MarkerSize',10);hold on;
plot(x_curr(2),y_curr(2),'o','MarkerSize',10);
%hLine = line('XData',x_curr(1), 'YData',y_curr(1), 'Color','r', ...
 %   	'Marker','o', 'MarkerSize',10, 'LineWidth',2);
%hold on
%hLine1 = line('XData',x_curr(1), 'YData',y_curr(2), 'Color','r', ...
%      'Marker','--o', 'MarkerSize',10, 'LineWidth',2);
axis([-90 90 -90 90])

%%%%%%%%%%%%%%%%%%%%%%%%the above the intial start  and now the real simulation
	inc=1;
	while((norm(error(1,:))>1e-1)&(norm(error(2,:))>1e-1))	
	disp(norm(error(1,:)));
	error(1,:) =g(1,:)-[x_curr(1),y_curr(1)];
	error(2,:) =g(2,:)-[x_curr(2),y_curr(2)];
	di(1,:)=error(1,:)-error_pre(1,:);
	inti(1,:)+=error(1,:);
	di(2,:)=error(2,:)-error_pre(2,:);
	inti(2,:)+=error(2,:);	
	v_pref(1,:)= kp*error(1,:) + kd*di(1,:)+ ki*inti(1,:);
	v_pref(2,:)=kp*error(2,:) + kd*di(2,:) + ki*inti(2,:);
	
	for l=1:2


	if(l==1)
		a=(x_curr(2)-x_curr(1))/t;
		b=(y_curr(2)-y_curr(1))/t;
		v_diff_opt=v_opt(1,:)-v_opt(2,:);
	else
		a=(x_curr(1)-x_curr(2))/t;
		b=(y_curr(1)-y_curr(2))/t;
		v_diff_opt=v_opt(2,:)-v_opt(1,:);
	endif



	c=[a^2-r^2,-2*a*b,b^2-r^2];

	m=roots(c);	


	x_tan=(a+m*b)./(1+m.^2);
	y_tan=m.*x_tan;
	if(a>0)
		th=astan(atan2((y_tan-b),(x_tan-a)));
	else
		th=atan2((y_tan-b),(x_tan-a));	
	endif

	theta=linspace(th(1),th(2),100);
	x_cir=r*cos(theta)+a;
	y_cir=r*sin(theta)+b;



	if (a<0)
		x_lin=linspace(x_tan(1),-130,200);

		x_lin1=linspace(x_tan(2),-130,200);

	else
		x_lin=linspace(x_tan(1),130,200);

		x_lin1=linspace(x_tan(2),130,200);

	endif

	y_lin=m(1).*x_lin;
	y_lin1=m(2).*x_lin1;




	v_cir=[x_cir',y_cir'];
	v1=[x_lin',y_lin'];size(v1);
	v2=[x_lin1',y_lin1'];
	V=[v1;v_cir];
	V=[V;v2];


	d1=(v_diff_opt(2)-m(1)*v_diff_opt(1))*(v_diff_opt(2)-m(2)*v_diff_opt(1));
	dist=norm(v_diff_opt);
	for(i=1:size(v_cir,1))
	distance(i)=norm(v_cir(i,:));
	endfor
	min_distance_in=argmin(distance);
	min_distance=distance(min_distance_in);
	max_distance_in=argmax(distance);
	max_distance=distance(max_distance_in);
	w=1;
	temp1=dist<max_distance;
	temp2=dist>min_distance;
	if(d1<0)
		if (norm(v_diff_opt)>max_distance)
			w=0;		

		elseif(temp2*temp1==1)
			if(norm(v_diff_opt-[a,b])<r)
				w=0;		
			endif
		endif
	endif
	

	
	if(w==0)
		
		for i=1:size(V,1)
			p(i)=norm(V(i,:)-v_diff_opt);
		endfor
		
		min_dis=argmin(p);
		u=V(min_dis,:)-v_diff_opt;
		%plot(u(1),u(2),'r*')
		l=l+1;

		
		v_x=linspace(-120,120,200);
		v_y=linspace(-120,120,200);
		[x,y]=meshgrid(v_x,v_y);
		A(:,:,1)=x;

		A(:,:,2)=y;
		
		for i=1:size(A,1)
			for j=1:size(A,2)
				A(i,j,1);
				a(1)=ans;
				A(i,j,2);
				a(2)=ans;
				if(l==1)
					inj(i,j)=dot((a-(v_opt(1,:)+1/2*u)),u/norm(u))>0;
				else
					inj(i,j)=dot((a-(v_opt(2,:)+1/2*u)),u/norm(u))>0;
				endif
			endfor
		endfor

		k=1;
		for i=1:size(inj,1)
		for j=1:size(inj,2)
			if(inj(i,j)~=0)		
				x_fin(k)=A(i,j,1);
				y_fin(k)=A(i,j,2);
				k=k+1;
			endif
				
		endfor
		endfor
			
		


		v=[x_fin',y_fin'];

		for i=1:size(v,1)

			if(l==1)
				pi(i)=norm(v(i,:)-v_pref(1,:));
			else 
					pi(i)=norm(v(i,:)-v_pref(2,:));
			endif
		endfor

		index_new=argmin(pi);
		if(l==1)
			va_new(1,:)=v(index_new,:);
		else
			va_new(2,:)=v(index_new,:);	
		endif


		
	else
		if(l==1)
			va_new(1,:)=v_pref(1,:);
		else
			va_new(2,:)=v_pref(2,:);	
		endif
	endif

	
endfor
		
			v_opt=va_new;
		
		x_curr=v_opt(:,1)*t+x_curr;
		y_curr=v_opt(:,2)*t+y_curr;

		v

	plot(x_curr(1),y_curr(1),'r*','MarkerSize',10);hold on;
	plot(x_curr(2),y_curr(2),'o','MarkerSize',10);

    error_pre=error;


inc=inc+1;
endwhile
	
