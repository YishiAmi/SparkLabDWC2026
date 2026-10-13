from paths import *
GRID.mkdir(exist_ok=True)
import numpy as np, pickle, os
O=str(GRID)+"/"
U=pickle.load(open(DATA/"unc.pkl","rb")); X=np.array([r[1:] for r in U[1:]],float)
def load(n):
    a=np.load(O+f"case_{n:03d}.npz")["a"].astype(np.float64); a[a==-999.25]=np.nan; return a
b=load(1); hdr=[str(h).strip() for h in np.load(O+"case_001.npz")["hdr"]]; ix={h:i for i,h in enumerate(hdr)}
act=b[:,6]>0; res={}
a2=load(2)
def chg(c): j=ix[c]; return act&(a2[:,j]!=b[:,j])&(b[:,j]>0)&(a2[:,j]>0)
masks={"PERMX":(chg("Permeability I (PERMX)"),ix["Permeability I (PERMX)"]),"TRANX":(chg("Transmissibility I (TRANX)"),ix["Transmissibility I (TRANX)"]),
       "PORV":(act&(a2[:,6]!=b[:,6]),6)}
# PORV: split by cells whose ratio != poro ratio (aquifer-affected)
pr=a2[:,6]/np.where(b[:,6]>0,b[:,6],np.nan); porr=a2[:,7]/np.where(b[:,7]>0,b[:,7],np.nan)
aq=act&(np.abs(pr/porr-1)>1e-4); print("PORV cells with extra (non-porosity) scaling:",aq.sum(), "K layers",np.unique(b[aq,2])[:3],"...", "I range",b[aq,0].min(),b[aq,0].max())
masks["PORV_aq"]=(aq,6); masks["PORV_other"]=(act&~aq,6)
M={k:[] for k in masks}
for n in range(2,101):
    a=load(n)
    for k,(m,j) in masks.items():
        r=a[m,j]/b[m,j]; 
        if k=="PORV_aq" or k=="PORV_other": r=r/ (a[m,7]/b[m,7]) if k=="PORV_aq" else r/(a[m,7]/b[m,7])
        M[k].append(np.nanmedian(r))
R=X[1:]/X[0]
for k,v in M.items():
    v=np.array(v); print(f"{k:11s} median ratio per case: min {v.min():.4f} max {v.max():.4f} | corr with ratios [fault,poro,perm,aq]:",[round(float(np.corrcoef(v,R[:,i])[0,1]),4) if v.std()>1e-12 else None for i in range(4)])
# is the changed-cell ratio uniform across the changed cells?
a=load(50)
for k in ("PERMX","TRANX"):
    m,j=masks[k]; r=a[m,j]/b[m,j]; print(k,"case50 ratio over changed cells: min %.3f median %.3f max %.3f; perm_mult ratio=%.3f fault ratio=%.3f"%(r.min(),np.median(r),r.max(),X[49,2]/X[0,2],X[49,0]/X[0,0]))
