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

# OAS method (see publication)
def oas(ts):
	a=ts2corr(ts)
	t=ts.shape[1]
	p=a.shape[0]
	trcc=np.sum(a*a)
	trc=np.trace(a)
	la=((1.0-2.0/p)*trcc+trc*trc)/(t+1-2.0/p)/(trcc-trc*trc/p)
	la=max(0.0,min(1.0,la))
	b=(1.0-la)*a+la*np.eye(p)		
	return tocorr(b)

		
################################################################################
if __name__ == "__main__":
	from argparse import ArgumentParser, RawTextHelpFormatter
	parser = ArgumentParser(description="OAS (see publication)",formatter_class=RawTextHelpFormatter)
	parser.add_argument("-i", "--input",help="Text file with the matrix containing the fMRI time series (space separated, each row containing a BOLD time series)", required=True)
	parser.add_argument("-o", "--output",help="Output correlation matrix after OAS.", required=True)
	args = parser.parse_args()
	
	if os.path.exists(args.input):
		ts=np.loadtxt(args.input)
		corr=oas(ts)
		np.savetxt(args.output,corr,fmt='%.8f')
	else:
		print('ERROR: file do not exist '+args.input)
		
		
		