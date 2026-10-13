from paths import *
import numpy as np, pandas as pd, json
D=np.load(DATA/"data.npz"); S=np.load(OUT/"selected.npz"); F20=np.load(OUT/"fc20.npz")["res"]; MC=json.load(open(OUT/"model_comparison.json"))
qd=pd.to_datetime(D["qd"].astype("datetime64[D]"))            # 41 quarterly dates 1998-01-01 .. 2008-01-01
fd=pd.to_datetime([f"{y if mo!=1 else y+1}-{mo:02d}-01" for y in range(2008,2028) for mo in (4,7,10,1)])  # 80 forecast dates
dt=0.25*365.25
# ---------- observed history
obs=pd.DataFrame({"Date":qd,"Gas_Cum_MSCF":D["gas_cum"][100],"Oil_Cum_STB":D["oil_cum"][100],"Water_Cum_STB":D["water_cum"][100],
                  "Gas_Rate_MSCFd":D["gas_rate"][100],"Oil_Rate_STBd":D["oil_rate"][100],"Water_Rate_STBd":D["water_rate"][100]})
# ---------- matched cases: history (proxy) + forecast
P=S["hist_pred"]; ser=["gas_cum","oil_cum","water_cum","gas_rate","oil_rate","water_rate"]
hp={k:P[:,i*41:(i+1)*41] for i,k in enumerate(ser)}
rows=[]
for j in range(10):
    for i,d in enumerate(qd):
        rows.append(dict(Case=j+1,Period="History (matched)",Date=d,Gas_Cum_MSCF=hp["gas_cum"][j,i],Oil_Cum_STB=hp["oil_cum"][j,i],Water_Cum_STB=hp["water_cum"][j,i],
                         Gas_Rate_MSCFd=hp["gas_rate"][j,i],Oil_Rate_STBd=hp["oil_rate"][j,i],Water_Rate_STBd=hp["water_rate"][j,i]))
    prev=np.array([hp["oil_cum"][j,-1],hp["gas_cum"][j,-1],hp["water_cum"][j,-1]])
    for r,d in enumerate(fd):
        o,g,w=F20[j,0,r],F20[j,1,r],F20[j,2,r]
        qo=(o-prev[0])/dt; qg=(g-prev[1])/dt; qw=(w-prev[2])/dt; prev=np.array([o,g,w])
        rows.append(dict(Case=j+1,Period="Forecast",Date=d,Gas_Cum_MSCF=g,Oil_Cum_STB=o,Water_Cum_STB=w,Gas_Rate_MSCFd=qg,Oil_Rate_STBd=qo,Water_Rate_STBd=qw))
all_=pd.DataFrame(rows); all_["Water_Cut"]=all_.Water_Rate_STBd/(all_.Water_Rate_STBd+all_.Oil_Rate_STBd).replace(0,np.nan)
rc=["Gas_Rate_MSCFd","Oil_Rate_STBd","Water_Rate_STBd","Gas_Cum_MSCF","Oil_Cum_STB","Water_Cum_STB"]; all_[rc]=all_[rc].clip(lower=0)  # remove ~1e-4 numerical noise from proxy
all_["Year"]=all_.Date.dt.year; all_["Horizon"]=np.where(all_.Date<=pd.Timestamp("2018-01-01"),"First 10 years (to 2018)","Years 11-20 (2018-2028)")
all_.loc[all_.Period=="History (matched)","Horizon"]="History"
all_.loc[all_.Period=="History (matched)","Water_Cut"]=np.nan
fore=all_[all_.Period=="Forecast"].copy()
f10=fore[["Case","Date","Gas_Cum_MSCF","Oil_Cum_STB","Water_Cum_STB"]]  # official deliverable: 10 cases x 80 quarters to 2028-01-01
# ---------- ensemble stats across the 10 cases (forecast + matched history)
g=all_.groupby("Date")
ens=pd.concat([g[c].agg(["min","mean","max"]).add_prefix(c+"_") for c in ["Gas_Cum_MSCF","Oil_Cum_STB","Water_Cum_STB","Oil_Rate_STBd","Water_Rate_STBd"]],axis=1).reset_index()
ens["Period"]=np.where(ens.Date<=pd.Timestamp("2008-01-01"),"History (matched)","Forecast")
# ---------- all 100 simulated cases (envelope)
env=[]
for j in range(100):
    grp="Train" if j<70 else ("Validation" if j<85 else "Test")
    env.append(pd.DataFrame({"SimCase":j+1,"Set":grp,"Date":qd,"Gas_Cum_MSCF":D["gas_cum"][j],"Oil_Cum_STB":D["oil_cum"][j],"Water_Cum_STB":D["water_cum"][j],
        "Oil_Rate_STBd":D["oil_rate"][j],"Water_Rate_STBd":D["water_rate"][j]}))
env=pd.concat(env,ignore_index=True)
# ---------- parameters
th=S["theta"]; par=pd.DataFrame({"Case":range(1,11),"Fault_Transmissibility":th[:,0],"Porosity_Multiplier":th[:,1],"Permeability_Multiplier":th[:,2],"Aquifer_Pore_Volume":th[:,3],"History_Mismatch_Score":S["mis"]})
for k,col in (("Oil_Cum_STB","Oil_Cum_2018_STB"),("Gas_Cum_MSCF","Gas_Cum_2018_MSCF"),("Water_Cum_STB","Water_Cum_2018_STB")):
    par[col]=[fore[(fore.Case==c)&(fore.Date==pd.Timestamp("2018-01-01"))][k].iloc[0] for c in range(1,11)]
for k,col in (("Oil_Cum_STB","Oil_Cum_2028_STB"),("Gas_Cum_MSCF","Gas_Cum_2028_MSCF"),("Water_Cum_STB","Water_Cum_2028_STB")):
    par[col]=[fore[(fore.Case==c)&(fore.Date==pd.Timestamp("2028-01-01"))][k].iloc[0] for c in range(1,11)]
# ---------- model accuracy
acc=[]
for r in MC:
    for split in ("val","test"):
        m=r[split]; acc.append({"Model":r["model"],"Dataset":"Validation" if split=="val" else "Test","Oil_Cum_Err_pct":m["oil_cum"],"Gas_Cum_Err_pct":m["gas_cum"],"Water_Cum_Err_pct":m["water_cum"],
                                "Oil_Rate_Err_pct":m["oil_rate"],"Water_Rate_Err_pct":m["water_rate"],"Final_Oil_Cum_Err_pct":m["final_oil_cum"]})
acc=pd.DataFrame(acc)
# ---------- KPIs
o08=obs.Oil_Cum_STB.iloc[-1]; 
kpi=pd.DataFrame([
 ("Observed oil cumulative at 2008-01-01 (STB)",o08),("Observed gas cumulative at 2008-01-01 (MSCF)",obs.Gas_Cum_MSCF.iloc[-1]),("Observed water cumulative at 2008-01-01 (STB)",obs.Water_Cum_STB.iloc[-1]),
 ("Mean forecast oil cumulative 2018 (STB)",par.Oil_Cum_2018_STB.mean()),("Mean forecast gas cumulative 2018 (MSCF)",par.Gas_Cum_2018_MSCF.mean()),("Mean forecast water cumulative 2018 (STB)",par.Water_Cum_2018_STB.mean()),
 ("Mean forecast oil cumulative 2028 (STB)",par.Oil_Cum_2028_STB.mean()),("Mean forecast gas cumulative 2028 (MSCF)",par.Gas_Cum_2028_MSCF.mean()),("Mean forecast water cumulative 2028 (STB)",par.Water_Cum_2028_STB.mean()),
 ("Oil forecast spread across 10 cases at 2018 (%)",(par.Oil_Cum_2018_STB.max()-par.Oil_Cum_2018_STB.min())/par.Oil_Cum_2018_STB.mean()*100),
 ("Best proxy model test error, oil cumulative (%)",acc[(acc.Dataset=="Test")&acc.Model.str.contains("Gaussian")].Oil_Cum_Err_pct.iloc[0]),
 ("Best history-mismatch score",par.History_Mismatch_Score.min())],columns=["Metric","Value"])
readme=pd.DataFrame({"Sheet":["Forecast_Official_2028","Cases_History_Forecast","Ensemble_Summary","Observed_History","Simulated_Cases","Case_Parameters","Model_Accuracy","KPIs"],
 "Contents":["Official deliverable data: 10 matched cases, quarterly cumulative gas/oil/water 2008-04-01..2028-01-01 (same numbers as 09 Template Deliverable - Filled.xlsx, with real dates)",
 "Tidy table: matched history (1998-2008) + forecast to 2028 per case, cumulative, rates, water cut. Use for line charts with Case as legend",
 "Min / mean / max across the 10 cases for each date (for forecast band charts)","Observed field history (the target the model was matched to)",
 "All 100 simulated cases 1998-2008 (grey 'spaghetti' context chart); Set = Train / Validation / Test",
 "The 4 uncertainty settings for each of the 10 matched cases, history mismatch score, and 2018 / 2028 totals","Model comparison error table (validation and test)","Headline numbers for cards"],
 "Notes":["Dates are real dates (the template stores text mm/dd/yyyy; US date format avoids day/month mix-up)","Period column splits History vs Forecast; Horizon splits first 10 years (to 2018) from years 11-20 (to 2028)","","","","Case 1 = lowest mismatch","Errors are mean absolute % on unseen cases","Values after 2018 are extrapolations; less certain than the first 10 years"]})
with pd.ExcelWriter(OUT/"PowerBI_Data.xlsx",engine="openpyxl",datetime_format="yyyy-mm-dd") as w:
    readme.to_excel(w,sheet_name="README",index=False); f10.to_excel(w,sheet_name="Forecast_Official_2028",index=False); all_.to_excel(w,sheet_name="Cases_History_Forecast",index=False)
    ens.to_excel(w,sheet_name="Ensemble_Summary",index=False); obs.to_excel(w,sheet_name="Observed_History",index=False); env.to_excel(w,sheet_name="Simulated_Cases",index=False)
    par.to_excel(w,sheet_name="Case_Parameters",index=False); acc.to_excel(w,sheet_name="Model_Accuracy",index=False); kpi.to_excel(w,sheet_name="KPIs",index=False)
    for ws in w.book.worksheets:
        ws.freeze_panes="A2"
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width=min(max(len(str(col[0].value))+2,12),42)
print({n:len(d) for n,d in [("f10",f10),("all",all_),("ens",ens),("obs",obs),("env",env),("par",par),("acc",acc),("kpi",kpi)]})
