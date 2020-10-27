
import numpy as np 

def reward():
	z=np.zeros((72, 96))

	z[30:42,0:2]=-0.5
	z[30:42,2:4]=-0.46
	z[30:42,4:7]=-0.44

	z[0:20,0:7]=-0.4
	z[20:30,0:7]=-0.42

	z[42:52,0:7]=-0.42
	z[52:72,0:7]=-0.40


	# zero until the middle
	c=0
	for i in range(5, 48):
	    z[:,i]=c*0.4/41 - 0.4
	    c=c+1
	    

	# until 76
	c=48
	for i in range(48, 76):
	    z[:,i]=c*0.4/41 - 0.4
	    c=c+1
	      
	# from 76 middle force
	c=76
	for i in range(76, 96):
	    z[30:42,i]=c*0.46/41 - 0.4
	    c=c+1   
	    
	# from 76 uper less
	c=76
	for i in range(76, 96):
	    z[0:30,i]=c*0.35/41 - 0.4
	    c=c+1
	    
	# from 76 lower less
	c=76
	for i in range(76, 96):
	    z[42:,i]=c*0.35/41 - 0.4
	    c=c+1
	    

	z[0:30,91:96]=0.28
	z[46:,91:96]=0.28


	z[30:42,95:96]=0.7

	return z
