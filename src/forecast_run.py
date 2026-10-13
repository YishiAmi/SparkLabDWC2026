"""Select the 10 best history-matched parameter sets, forecast 2008-04-01 .. 2028-01-01 (80 quarters), fill the official template."""
from paths import *
import numpy as np, pickle, openpyxl
from proxy import *
from decline import forecast
from hm import misfit
S,mf,mfc,lo,hi,best=pickle.load(open(OUT/"hm_samples.pkl","rb")); m=pickle.load(open(MODELS/"proxy_final.pkl","rb"))
NF=80
# --- 10 diverse near-optimal parameter sets (distance > 0.2 in normalised fault/perm/aquifer space)
nrm=lambda p:(p-lo)/(hi-lo)
order=np.argsort(mf); sel=[best[0][1]]
for i in order:
    if len(sel)==10: break
    if all(np.linalg.norm(nrm(S[i])[[0,2,3]]-nrm(s)[[0,2,3]])>0.2 for s in sel): sel.append(S[i])
sel=np.array(sel); P=m.predict(sel); sp=split(P); mfs=misfit(P)
print("selected misfits",np.round(mfs,5))
def fc(j):   # oil: mean of exponential and hyperbolic decline; gas: GOR trend; water: ln(WOR) vs cumulative oil
    outs=[forecast(sp["oil_rate"][j],sp["gas_rate"][j],sp["water_rate"][j],sp["oil_cum"][j],sp["gas_cum"][j],sp["water_cum"][j],40,NF,12,mode,"wor") for mode in ("exp","hyp")]
    return [(outs[0][k]+outs[1][k])/2 for k in range(3)]       # oil, gas, water cumulative
res=np.array([fc(j) for j in range(10)])                        # 10 x 3 x 80
np.savez(OUT/"selected.npz",theta=sel,mis=mfs,fc=res[:,:,:40],hist_pred=P); np.savez(OUT/"fc20.npz",res=res)
wb=openpyxl.load_workbook(DATA/"09 Template Deliverable.xlsx")
exp=[(y if mo!=1 else y+1,mo) for y in range(2008,2028) for mo in (4,7,10,1)]
for j in range(10):
    ws=wb[f"Case {j+1}"]
    assert ws.cell(1,2).value.startswith("Gas") and ws.cell(1,3).value.startswith("Oil") and ws.cell(1,4).value.startswith("Water")
    for r in range(NF):
        v=ws.cell(r+2,1).value
        got=(int(v.split("/")[2]),int(v.split("/")[0])) if isinstance(v,str) else (v.year,v.month)
        assert got==exp[r],(j,r,v)
        ws.cell(r+2,2).value=float(res[j,1,r]); ws.cell(r+2,3).value=float(res[j,0,r]); ws.cell(r+2,4).value=float(res[j,2,r])
wb.save(OUT/"09 Template Deliverable - Filled.xlsx"); print("saved filled template (10 cases x 80 quarters)")
print("2028 oil min/max",res[:,0,-1].min().round(),res[:,0,-1].max().round())
