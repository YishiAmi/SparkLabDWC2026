from paths import *
GRID.mkdir(exist_ok=True)
import sys, os, time, re, numpy as np
from python_calamine import CalamineWorkbook
D=str(RAW/"TestCases")+"/"; O=str(GRID)+"/"
files=["03 Train Cases 1.xlsx","04 Train Cases 2.xlsx","05 Train Cases 3.xlsx","06 Train Cases 4.xlsx","07  Validation Cases.xlsx","08 Test Cases.xlsx"]
log=open(O+"progress.log","a")
for f in files:
    wb=CalamineWorkbook.from_path(D+f)
    for sn in wb.sheet_names:
        n=int(re.findall(r"\d+",sn)[0]); out=O+f"case_{n:03d}.npz"
        if os.path.exists(out): continue
        t=time.time(); rows=wb.get_sheet_by_name(sn).to_python(skip_empty_area=False)
        hdr=rows[0]; a=np.array([[np.nan if (v=="" or v is None) else v for v in r] for r in rows[1:]],dtype=np.float32)
        np.savez_compressed(out+".tmp.npz",a=a,hdr=np.array(hdr)); os.replace(out+".tmp.npz",out)
        print(f,sn,a.shape,round(time.time()-t,1),"s",file=log,flush=True); del rows,a
print("ALLDONE",file=log,flush=True)
