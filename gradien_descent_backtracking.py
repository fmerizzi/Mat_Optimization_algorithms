# BACKTRACKING 

import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

lambd = 1

def f(x,y):
    b = np.array([1,1])
    lambd = 1
    return (x - b[0])**2 + (y - b[1])**2 + lambd*(x**2 + y**2)  

def grd(x):
  b = np.array([1,1])
  lambd = 1
  return 2*(x-b) + lambd*2*x 

def back(k,tau):

  alpha = tau
  ro = 0.5
  c1 = 0.5
  j = 0

  lastFactor  = np.matmul(np.transpose(grd(k)), -grd(k))
  #print("last Factor : ", c1*alpha*lastFactor)
  value = k-alpha*grd(k)
  #print("function value : ", f((k-alpha*grd(k))[0],(k-alpha*grd(k))[1]))
  #print("function second value: ", f(k[0],k[1]))

  max = 20

  while f((k-alpha*grd(k))[0],(k-alpha*grd(k))[1]) > (f(k[0],k[1]) + c1*alpha*lastFactor) and j<20:
    alpha = ro*alpha 
    j += 1
  
  print("backtracking iterations : ", j)
  return alpha

def step(k, step):
  return k - step*grd(k) 

def stopCrit(k,tau):
  #print("norm :", np.linalg.norm(grd(k)))
  if(np.linalg.norm(grd(k)) < tau) :
    return True 
  else:
    return False 

## START

start = np.array([10,10])
story = np.array([[10,10]])
stop = False
iterations = 0
max = 10
tau = 6

while (stop == False and iterations < max):
  stop = stopCrit(start,tau)

  tau = back(start,tau) 
  print("-----")
  print("backtrack alpha = ",tau)
  start = step(start,tau)
  story = np.append(story,[start],0)
  f_start = f(start[0],start[1])

  iterations += 1

  print("z = ",f_start);
  print("(x,y) = ",start)

print("--- ALGORITHM --- ")
if(stop == True):
  print("algorithm was stopped by gradient < tau, after iterations : ", iterations)
elif(iterations >= max) :
  print("algorithm was stopped by too much iterations in main cycle, iter : ",iterations)
print(" ----------------- ")
 
x = np.linspace(-3,3,100)
y = np.linspace(-3,3, 100)
xv, yv = np.meshgrid(x, y) 

#compute z values
z=f(xv,yv)


print("---story---")
print(story)
print("---story---")

fig = plt.figure()
ax = plt.axes(projection='3d')
#you plot on the axes?
ax.plot_surface(x, y, z,cmap='viridis')
ax.set_title('Surface plot')
plt.show()

contours = plt.contour(x, y, z)
plt.plot(story[1:,0], story[1:,1], '-o')
plt.show 
