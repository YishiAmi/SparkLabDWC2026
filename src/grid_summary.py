from paths import *
GRID.mkdir(exist_ok=True)
import numpy as np, glob, os, pickle
O=str(GRID)+"/"; rows=[]
hdr=None
for n in range(1,101):
    z=np.load(O+f"case_{n:03d}.npz"); a=z["a"].astype(np.float64); hdr=[str(h).strip() for h in z["hdr"]]
    a[a==-999.25]=np.nan; c={h:a[:,i] for i,h in enumerate(hdr)}
    pv=np.nan_to_num(c["Pore Volume (PORV)"]); act=pv>0
    def wm(k): v=c[k]; m=act&~np.isnan(v); return (pv[m]*v[m]).sum()/pv[m].sum()
    def mm(k): v=c[k][act]; return np.nanmean(v)
    r={"case":n,"cells":int(act.sum()),"PV":pv.sum(),"poro":mm("Porosity (PORO)"),"permx":mm("Permeability I (PERMX)"),"permz":mm("Permeability K (PERMZ)"),"ntg":mm("Net-to-Gross Ratio (NTG)"),
       "tranx":mm("Transmissibility I (TRANX)"),"press":wm("Pressure (PRESSURE)"),"sw":wm("Water Saturation (SWAT)"),"so":wm("Oil Saturation (SOIL)"),"sg":wm("Gas Saturation (SGAS)"),"rs":wm("Solution Gas-Oil Ratio (RS)"),
       "oil_pv":(pv*np.nan_to_num(c["Oil Saturation (SOIL)"])).sum(),"gas_pv":(pv*np.nan_to_num(c["Gas Saturation (SGAS)"])).sum(),"wat_pv":(pv*np.nan_to_num(c["Water Saturation (SWAT)"])).sum(),
       "sowcr":mm("Residual Oil to Water (SOWCR)"),"swl":mm("Connate Water Saturation (SWL)")}
    rows.append(r)
pickle.dump(rows,open(OUT/"grid_summary.pkl","wb")); print(hdr[:3],len(rows)); print(rows[0]); print(rows[99])

import csv
U=pickle.load(open(DATA/"unc.pkl","rb")); X=[r[1:] for r in U[1:]]
with open(OUT/"grid_summary.csv","w",newline="") as f:
    w=csv.writer(f); keys=list(rows[0].keys()); w.writerow(keys+["fault_trans","poro_mult","perm_mult","aquifer_pv"])
    for r,x in zip(rows,X): w.writerow([r[k] for k in keys]+list(x))
