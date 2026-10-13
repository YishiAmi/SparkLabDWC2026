from paths import *
GRID.mkdir(exist_ok=True)
import numpy as np, pickle, os
O=str(GRID)+"/"
z=np.load(O+"case_001.npz"); hdr=[str(h).strip() for h in z["hdr"]]; base=z["a"]
nz=base.shape[1]; ndiff=np.zeros((100,nz),int); meanv=np.zeros((100,nz)); 
for n in range(1,101):
    a=np.load(O+f"case_{n:03d}.npz")["a"]
    d=~((a==base)|(np.isnan(a)&np.isnan(base)))
    ndiff[n-1]=d.sum(0)
    b=a.astype(np.float64); b[b==-999.25]=np.nan
    meanv[n-1]=np.nanmean(b,0)
pickle.dump((hdr,ndiff,meanv),open(OUT/"grid_vary.pkl","wb"))
print(f"{'column':42s} {'cells differing vs case1 (max over cases)':>40s} {'case-to-case spread of mean':>28s}")
for i,h in enumerate(hdr): print(f"{h:42s} {ndiff[:,i].max():40d} {np.ptp(meanv[:,i])/max(abs(meanv[:,i]).mean(),1e-12):28.2e}")
