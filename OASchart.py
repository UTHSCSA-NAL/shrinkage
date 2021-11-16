import os
import numpy as np
import matplotlib.pyplot as plt

################################################################################

# compute OAS intensity, "tr" denoting the squared Frobenius norm of the matrix: Tr(S*S)
def shrinkage(n,p,tr):
	return min(1.0,  ( (1.0-2.0/p)*tr+p*p  )/( (n+1-2.0/p)*(tr-p) )    )
   
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

# compute correlation matrix density	
def corr2density(cr):
	p=cr.shape[0]
	v=np.sum(cr*cr)
	return (v-p)/(p*p-p)	

def myfmt(x):
    s = str(x)
    while s.endswith("0"):
    	s=s[:-1]
    return s


def makeColors(n):
	rr=[]
	for i in range(0,n):
		if (float(i)/float(n))<=0.5:
			r=0
			g=0
			b=int( 255.0*float(i)/float(n)*2.0 )
			he='#%02x%02x%02x' % (r,g,b)
			rr.append( he.upper() )
		else:
			s=(float(i)/float(n)-0.5)*2.0
			r=int(225.0*s)
			g=int(125.0*s)
			b=int(255.0*(1.0-s))
			he='#%02x%02x%02x' % (r,g,b)
			rr.append( he.upper() )	
	return rr[::-1]
	

def makeChart(p,n_min,n_max,d_min,d_max,levels,files,output):
	# OAS intensity on the grid
	resx=np.linspace(np.log(float(n_min)),np.log(float(n_max)),501)      
	resx=np.exp(resx)
	resy=np.linspace(np.log(d_min),np.log(d_max),501)
	resy=np.exp(resy)
	xs=np.zeros((len(resx)))
	ys=np.zeros((len(resy)))
	zs=np.zeros((len(resy),len(resx)))
	for x in range(0,len(resx)):
		n=int(resx[x])
		xs[x]=n
	for y in range(0,len(resy)):
		ys[y]=resy[y]	
	for x in range(0,len(resx)):
		n=int(resx[x])
		for y in range(0,len(resy)): 
			tr=p*(1.0-resy[y])+p*p*resy[y]
			zs[y,x]=shrinkage(n,p,tr)

	# plot the chart	
	plt.title('OAS intensity (p='+str(p)+')',fontsize=22)
	plt.xlabel('number of time points n',fontsize=18)
	plt.ylabel('Density',fontsize=18)
	plt.xscale('log')
	plt.yscale('log')
	
	tmp=args.levels.split(',')
	ls=[]
	for s in tmp:
		ls.append(float(s))
	ls.sort()
	
	
	# colors
	cols=makeColors(len(ls))
	
	# contours
	contours = plt.contour(xs,ys,zs,levels=ls,colors=cols)

	# connectomes
	if len(files)>0:
		xxs=[]
		yys=[]
		for i in range(0,len(files)):
			dat=normalization(np.loadtxt(files[i]))
			cr=tocorr(dat.dot(np.transpose(dat)))
			xxs.append(dat.shape[1])
			yys.append(corr2density(cr))
		plt.scatter(xxs,yys,marker='o',c=[1.0,0,0],s=np.ones((len(xxs)))*5.0 )

	# contour labels
	locs=[]
	cpt=0
	for line in contours.collections:
		for path in line.get_paths():
			idxs=np.argsort(path.vertices[:,1])
			idx=idxs[int( len(idxs)*0.5 )]
			locs.append(path.vertices[idx,:])
		cpt=cpt+1
	plt.clabel(contours, inline=1, fontsize=18,fmt=myfmt,inline_spacing=3, rightside_up=True,manual=locs)
	plt.xlim([np.min(resx),np.max(resx)])
	
	axes= plt.axes()
	plt.xticks(fontsize=16)
	plt.yticks(fontsize=16)
	plt.savefig(output,bbox_inches='tight')




def checkFiles(lis):
	r=[]
	fi=open(lis,'r')
	tmp=fi.readlines()
	fi.close()
	for s in tmp:
		while s.endswith('\n') or s.endswith('\r'):
			s=s[:-1]
		if os.path.exists(s):
			r.append(s)
		else:
			print('ERROR: file do not exist '+s)
	return r
		
################################################################################
if __name__ == "__main__":
	from argparse import ArgumentParser, RawTextHelpFormatter
	parser = ArgumentParser(description="OAS intensity charts (see publication)",formatter_class=RawTextHelpFormatter)
	parser.add_argument("-d", "--dimension",help="Dimension of the correlation matrices.", required=True, type=int)
	parser.add_argument("-li", "--list",help="List of files containing time series (to add corresponding Pearson correlation matrices to the chart). Not required.", required=False,default='')
	parser.add_argument("-le", "--levels",help="OAS intenstiy level sets to display, comma separated without space (default is 0.9,0.5,0.25,0.1,0.05,0.025,0.01,0.005)", required=False,default='0.9,0.5,0.25,0.1,0.05,0.025,0.01,0.005')	
	parser.add_argument("-o", "--output",help="Output plot (default is OAS_chart.png)",required=False,default='OAS_chart.png')
	
	parser.add_argument("--n_min",help="Smallest number of time points (default is 15, should be larger than 10).",required=False,default='15',type=int)
	parser.add_argument("--n_max",help="Largest number of time points (default is 1500, should be larger than two times n_min).",required=False,default='1500',type=int)
	parser.add_argument("--d_min",help="Smallest density (default is 0.01, should be larger than 0).",required=False,default='0.01',type=float)
	parser.add_argument("--d_max",help="Largest density (default is 1.0, should be smaller than 1).",required=False,default='1.0',type=float)
	
	
	args = parser.parse_args()
	
	if len(args.list)>0:
		if os.path.exists(args.list):
			files=checkFiles(args.list)
			makeChart(args.dimension,args.n_min,args.n_max,args.d_min,args.d_max,args.levels,files,args.output)
		else:
			print('ERROR: file do not exist '+args.list)
	else:
		print('Chart without data point')
		makeChart(args.dimension,args.n_min,args.n_max,args.d_min,args.d_max,args.levels,[],args.output)
		
