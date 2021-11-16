# version 1.0.0
# author: Nicolas Honnorat
# date: November 16, 2021

import os
import numpy as np

# project a covariance matrix to produce a correlation matrix
def tocorr(m):
	return np.minimum(1,np.maximum(-1,m-np.diag(np.diag(m))+np.eye(m.shape[0]) ))

# normalize a set of time series to zero mean and unit L2 norm 
def normalization(si):	
	on=np.ones((1,si.shape[1]))
	eps=0.000000001
	si=si-np.mean(si,axis=1,keepdims=True).dot(on)
	si=np.divide(si,np.maximum(np.sqrt(np.sum(si*si,axis=1,keepdims=True)).dot(on),eps*on)   )
	return si	

# compute a Pearson correlation matrix from a set of time series
def ts2corr(ts):
	ts=normalization(ts)
	return tocorr(ts.dot(np.transpose(ts)))

# NERCOME method (see publication)
def nercome(si,prop,iters):
	si=normalization(si)
	r=[]
	m=min(si.shape[1]-2,int(si.shape[1]*prop))
	si=si*np.sqrt(si.shape[1])
	sig=np.zeros((si.shape[0],si.shape[0]))
	for ii in range(0,iters):
		sj=si[:,np.random.permutation(si.shape[1])]
		ya=sj[:,:m]
		yb=sj[:,m:]
		p1,s,v=np.linalg.svd(ya,full_matrices=False)
		sig2=np.cov(yb)
		sig=sig+p1.dot(np.diag(np.diag(np.transpose(p1).dot(sig2).dot(p1)))).dot(np.transpose(p1))
	return tocorr(sig/float(iters))
	

		
################################################################################
if __name__ == "__main__":
	from argparse import ArgumentParser, RawTextHelpFormatter
	parser = ArgumentParser(description="NERCOME (see publication)",formatter_class=RawTextHelpFormatter)
	parser.add_argument("-i", "--input",help="Text file with the matrix containing the fMRI time series (space separated, each row containing a BOLD time series)", required=True)
	parser.add_argument("-n", "--number",help="Number of matrices to combine (default is 50)", required=False, default=50,type=int)
	parser.add_argument("-p", "--proportion",help="Proportion retained for computing the S2 matrices (default is 0.9)", required=False, default=0.9,type=float)
	parser.add_argument("-o", "--output",help="Output correlation matrix after NERCOME.", required=True)
	args = parser.parse_args()
	
	if os.path.exists(args.input):
		if float(args.proportion)>0.0 and float(args.proportion)<1.0:
			if int(args.number)>1:
				ts=np.loadtxt(args.input)
				corr=nercome(ts,float(args.proportion),int(args.number))
				np.savetxt(args.output,corr,fmt='%.8f')
			else:
				print('ERROR: wrong number '+str(args.number)+' (should and integer larger than 1)')
		else:
			print('ERROR: wrong proportion '+str(args.proportion)+' (should be strictly between 0 and 1)')
	else:
		print('ERROR: file do not exist '+args.input)
		
