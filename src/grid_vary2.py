from paths import *
GRID.mkdir(exist_ok=True)
import numpy as np, pickle, os
O=str(GRID)+"/"
U=pickle.load(open(DATA/"unc.pkl","rb")); X=np.array([r[1:] for r in U[1:]],float)
def load(n):
    a=np.load(O+f"case_{n:03d}.npz")["a"].astype(np.float64); a[a==-999.25]=np.nan; return a
b=load(1); hdr=[str(h).strip() for h in np.load(O+"case_001.npz")["hdr"]]
ix={h:i for i,h in enumerate(hdr)}
act=b[:,6]>0
cols=["Pore Volume (PORV)","Porosity (PORO)","Permeability I (PERMX)","Transmissibility I (TRANX)","Transmissibility J (TRANY)"]
out={c:[] for c in cols}; 
for n in range(2,101):
    a=load(n)
    for c in cols:
        j=ix[c]; m=act&(b[:,j]>0)&(a[:,j]>0)
        r=a[m,j]/b[m,j]; out[c].append((n,np.median(r),r.min(),r.max(),(np.abs(r-1)>1e-9).sum()))
for c in cols:
    arr=np.array(out[c]); spread=(arr[:,3]-arr[:,2]).max()
    print(c,"| per-case ratio-to-case1 uniform across cells? max(max-min)=%.3g"%spread,"| n cells changed (median)",int(np.median(arr[:,4])))
    med=arr[:,1]; 
    for k,nm in enumerate(["fault","poro_mult","perm_mult","aq_pv"]):
        pass
    A=np.column_stack([np.ones(99),X[1:,0]/X[0,0],X[1:,1]/X[0,1],X[1:,2]/X[0,2],X[1:,3]/X[0,3]]); co,res,_,_=np.linalg.lstsq(A,med,rcond=None); p=A@co
    print("   median ratio ~ linear in ratios of [fault,poro,perm,aq]: R2=%.5f"%(1-((med-p)**2).sum()/((med-med.mean())**2).sum()),"coef",np.round(co,4))
    print("   corr with each ratio:",[round(float(np.corrcoef(med,A[:,k+1])[0,1]),4) for k in range(4)])
# which cells differ: same set for PERMX and TRANX?
a=load(2)
dp=(a[:,ix["Permeability I (PERMX)"]]!=b[:,ix["Permeability I (PERMX)"]])&act
dt=(a[:,ix["Transmissibility I (TRANX)"]]!=b[:,ix["Transmissibility I (TRANX)"]])&act
dv=(a[:,ix["Pore Volume (PORV)"]]!=b[:,ix["Pore Volume (PORV)"]])&act
print("cells PERMX differ:",dp.sum(),"TRANX differ:",dt.sum(),"overlap",(dp&dt).sum(),"PORV differ",dv.sum())
print("K range of PERMX-changed cells:",np.unique(b[dp,2])[:5],"..",np.unique(b[dp,2]).size,"layers; K range of all active:",np.unique(b[act,2]).min(),np.unique(b[act,2]).max())
