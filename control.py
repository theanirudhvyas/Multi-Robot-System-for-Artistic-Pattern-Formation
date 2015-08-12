import numpy as np
import matplotlib.pyplot as plt

x_curr=[[-50],[50]]
y_curr=[[40],[-40]]

inti = np.zeros((2,2))
di = np.zeros((2,2))

t=3.5
r=8/t

g=[[50,-40],[-50,40]]
error_pre=np.zeros((2,2))

ki=1e-5
kd=0.001
kp=.1

error = [np.subtract(g[0],[x_curr[0],y_curr[0]]), np.subtract(g[1],[x_curr[1],y_curr[1]])]

# ********************* checking the valuse by plotting them
'''
plt.plot(x_curr[0],y_curr[0],'r--*')
plt.plot(x_curr[1],y_curr[1],'--o')
plt.show()
'''
#**************************8**************************************8

di=[np.subtract(error[0], error_pre[0]), np.subtract(error[1],error_pre[1])]
np.add(inti, error)

v_opt= [np.add(np.add(np.multiply(kp,error[0]), np.multiply(kd,di[0])), np.multiply(ki,inti[0])), np.add(np.add(np.multiply(kp,error[1]), np.multiply(kd,di[1])), np.multiply(ki,inti[1]))]
v_opt = np.array(v_opt)
v_pref = np.array(v_opt)

x_curr=np.add(np.multiply(v_opt[:,0], t), x_curr)
y_curr=np.add(np.multiply(v_opt[:,1], t) + y_curr)

plt.plot(x_curr[0],y_curr[0],'r*','MarkerSize',10)
plt.plot(x_curr[1],y_curr[1],'o','MarkerSize',10)
pt.display()

#************************** simulation starts from here ***********************

inc = 1

while(np.linalg.norm(error[0]> 0.1 ))&(np.linalg.norm(error[1]> 0.1 )):
	error =[np.subtract(g[0], [x_curr[0],y_curr[0]]), np.subtract(g[1], [x_curr[1],y_curr[1]])]
	
	di=[np.subtract(error[0], error_pre[0]), np.subtract(error[1],error_pre[1])]
	np.add(inti, error)

	v_pref = [np.add(np.add(np.multiply(kp,error[0]), np.multiply(kd,di[0])), np.multiply(ki,inti[0])), np.add(np.add(np.multiply(kp,error[1]), np.multiply(kd,di[1])), np.multiply(ki,inti[1]))]

	for l in range(2):

		if l == 0:
			a=np.multiply((np.subtract(x_curr[1],x_curr[0])), (1/t))
			b=np.multiply((np.subtract(y_curr[1],y_curr[0])), (1/t))
			v_diff_opt=np.subtract(v_opt[0], v_opt[1])
		else:
			a=np.multiply((np.subtract(x_curr[0],x_curr[1])), (1/t))
			b=np.multiply((np.subtract(y_curr[0],y_curr[1])), (1/t))
			v_diff_opt=np.subtract(v_opt[1], v_opt[0])

	c=[a**2-r**2,-2*a*b,b**2-r**2]
	m=np.roots(c)

	np.array(m, dtype=np.int64)

	x_tan = [0]* len(m)
	np.array(x_tan, dtype=np.int64)	
	y_tan= [0]* len(m)
	np.array(y_tan, dtype=np.int64)

	for i in np.arange(len(m)):
		x_tan[i] = (a+m[i]*b)/(1+m[i]**2)
		y_tan[i] = m[i]*x_tan[i]

	th = [0]*len(m)
	if a>0:
		for i in np.arange(len(m)):
			th[i]=astan(np.arctan2((y_tan[i]-b),(x_tan[i]-a)))
	else:
		for i in np.arange(len(m)):
			th[i]=np.arctan2((y_tan[i]-b),(x_tan[i]-a))

	theta=np.linspace(th[0],th[1],100)

	x_cir = [0]*len(theta)
	y_cir = [0]*len(theta)
	for i in np.arange(100):
		x_cir[i]=r*np.cos(theta[i])+a
		y_cir[i]=r*np.sin(theta[i])+b



	if a<0:
		x_lin=np.linspace(x_tan[0],-130,200)
		x_lin1=np.linspace(x_tan[1],-130,200)

	else:
		x_lin=np.linspace(x_tan[0],130,200)
		x_lin1=np.linspace(x_tan[1],130,200)

	y_lin=np.multiply(m[0], x_lin)
	y_lin1=np.multiply(m[1], x_lin1)

	v=[]
	v1=[]
	v2=[]
	for i in np.arange(np.size(x_cir)):
		v.append([x_cir[i], y_cir[i]])
		v1.append([x_lin[i], y_lin[i]])
		v2.append([x_lin1[i], y_lin1[i]])

	V = np.concatenate((v1, v), axis = 0)
	V = np.concatenate((V, v2), axis = 0)

	d1=np.multiply( np.subtract(v_diff_opt[1], np.multiply(m[0], v_diff_opt[0])), np.subtract(v_diff_opt[1], np.multiply(m[1], v_diff_opt[0]) ))

	dist=np.linalg.norm(v_diff_opt)

	distance = [0]*np.size(v[i])
	for i in np.arange(np.size(v[i])):
		distance[i]=np.linalg.norm(v[i])

		min_distance_in=np.argmin(distance)
		min_distance=distance[min_distance_in]
		max_distance_in=np.argmax(distance)
		max_distance=distance[max_distance_in]

		w=1
		temp1 = dist<max_distance
		temp2 = dist>min_distance

		if d1 < 0:
			if np.linalg.norm(v_diff_opt)>max_distance:
				w=0		

			elif temp2 and  temp1:
				if np.linalg.norm(np.subtract( v_diff_opt, [a,b]))<r :
					w=0

		if w == 0:

			p = [0]*np.size(v[i])
			for i in np.arange(np.size(v[i])):
				p[i]=np.linalg.norm(np.subtract( V[i], v_diff_opt))

			min_dis=np.argmin(p);
			u=np.subtract(V[min_dis],v_diff_opt)
			l+=1
			v_x=np.linspace(-120,120,200)
			v_y=np.linspace(-120,120,200)

			[x,y]=np.meshgrid(v_x,v_y)

			A = []
			A=A.append(x)
			A=A.append(y)

			for i in np.arange(np.size(A,1)):
				for j in np.arange(np.size(A,2)):
					A[i,j,0]
					a[0]=ans
					A[i,j,1]
					a[1]=ans
					if l==1:
						inj[i,j]=np.dot((a-np.add(v_opt[0],np.multiply(1/2,u))),np.divide(u, norm(u)))>0
					else:
						inj[i,j]=np.dot((a-np.add(v_opt[1],np.multiply(1/2,u))),np.divide(u, norm(u)))>0
					

			k=1
			for i in np.arange(np.size(inj,1)):
				for j in np.arange(np.size(inj,2)):
					if inj[i,j]!=0:		
						x_fin[k]=A[i,j,0]
						y_fin[k]=A[i,j,1]
						k=k+1;


			v=[np.transpose(x_fin),np.transpose(y_fin)]

			for i in np.arange(np.size(v,1)):

				if l==1 :
					pi[i]=np.linalg.norm(v[i]-v_pref[0])
				else :
						pi[i]=np.linalg.norm(v[i]-v_pref[1])

			index_new=np.argmin(pi)
			if l == 1:
				va_new[0]=v[index_new]
			else:
				va_new[1]=v[index_new]


		else:
			if l==1:
				va_new[0]=v_pref[0]
			else:
				va_new[1]=v_pref[1]


	v_opt=va_new

	x_curr=np.add(np.multiply(v_opt[:,0],t), x_curr)
	y_curr=np.add(np.multiply(v_opt[:,1],t), y_curr)

	plt.plot(x_curr[0],y_curr[1],'r*','MarkerSize',10)
	plt.plot(x_curr[0],y_curr[1],'o','MarkerSize',10)

	error_pre=error
	inc += 1
