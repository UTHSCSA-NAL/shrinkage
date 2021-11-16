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

# NAS method (see publication)
def mm(x):
	return max(0.0,1.0-0.2*x*x)
def mmm(x):
	return 1.0-0.2*x*x
def nas(ts):
	ts=normalization(ts)
	cl=ts.dot(np.transpose(ts))
	p=ts.shape[0]
	n=ts.shape[1]
	
	v,la,w=np.linalg.svd(ts,full_matrices=False)
	la=la*la
	ord=np.argsort(-la)
	la=la[ord]
	v=v[:,ord]
	if p>n:
		cut=len(la)
		while la[cut-1]<la[0]*0.000001:
			cut=cut-1
		la=la[:cut]
		v=v[:,:cut]
	else:
		th=la[0]*0.001
		la[la<th]=th
	
	
	h=np.lib.scimath.power(n,-1.0/3.0)
	hj=h*la
	if n>=p:
		ff=np.zeros((p))
		hf=np.zeros((p))
		sq=np.sqrt(5.0)
		for i in range(0,p):
			for j in range(0,p):
				x=(la[i]-la[j])/hj[j]
				ff[i]=ff[i]+mm(x)*3.0/(float(p)*4.0*sq*hj[j])
				if abs(abs(x)-sq)<0.00001:
					hf[i]=hf[i]-3.0*(x)/(float(p)*10.0*np.pi*hj[j])			
				else:
					hf[i]=hf[i]-3.0*(x)/(float(p)*10.0*np.pi*hj[j])+mmm(x)*3.0/(float(p)*np.pi*4.0*sq*hj[j])*np.log(np.abs( (sq-x)/(sq+x) ))
		dd=np.zeros((p))		
		for i in range(0,p):
			a=np.pi*float(p)/float(n)*la[i]*ff[i]
			b=1.0-float(p)/float(n)-np.pi*float(p)/float(n)*la[i]*hf[i]
			dd[i]=la[i]/( a*a+b*b )
		r=v.dot( np.diag(dd) ).dot(np.transpose(v))
	else:	
		pp=len(la)
		ff=np.zeros((pp))
		hf=np.zeros((pp))
		sq=np.sqrt(5.0)
		for i in range(0,pp):
			for j in range(0,pp):
				x=(la[i]-la[j])/hj[j]
				ff[i]=ff[i]+mm(x)*3.0/(float(pp)*4.0*sq*hj[j])
				if abs(abs(x)-sq)<0.00001:
					hf[i]=hf[i]-3.0*(x)/(float(pp)*10.0*np.pi*hj[j])			
				else:
					hf[i]=hf[i]-3.0*(x)/(float(pp)*10.0*np.pi*hj[j])+mmm(x)*3.0/(float(pp)*np.pi*4.0*sq*hj[j])*np.log(np.abs( (sq-x)/(sq+x) ))
		dd=np.zeros((pp))
		ho=np.zeros((pp))
		for i in range(0,pp):
			ho[i]=1.0/la[i]
		ho=np.sum(ho)/np.pi*( 0.3/h/h+0.75/sq/h*(1.0-1.0/h/h/5.0)*np.log( (1.0+sq*h)/(1.0-sq*h) ) )/float(n)
		delta=1.0/( np.pi*(p-n)/float(n)*ho )
		for i in range(0,pp):
			dd[i]=1.0/( np.pi*np.pi*la[i]*(ff[i]*ff[i]+hf[i]*hf[i]))-delta
		r=v.dot( np.diag(dd) ).dot(np.transpose(v))+delta*np.eye(ts.shape[0])
		
	r=tocorr(r)	
	return r
	
		
################################################################################
if __name__ == "__main__":
	from argparse import ArgumentParser, RawTextHelpFormatter
	parser = ArgumentParser(description="NAS (see publication)",formatter_class=RawTextHelpFormatter)
	parser.add_argument("-i", "--input",help="Text file with the matrix containing the fMRI time series (space separated, each row containing a BOLD time series)", required=True)
	parser.add_argument("-o", "--output",help="Output correlation matrix after NAS.", required=True)
	args = parser.parse_args()
	
	if os.path.exists(args.input):
		ts=np.loadtxt(args.input)
		corr=nas(ts)
		np.savetxt(args.output,corr,fmt='%.8f')
	else:
		print('ERROR: file do not exist '+args.input)
		
