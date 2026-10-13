from paths import *
import numpy as np, warnings, json
warnings.filterwarnings("ignore")
from proxy import *
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
tr,va,te=slice(0,70),slice(70,85),slice(85,100)
NC=10
sx=StandardScaler().fit(feat(X[tr])); sy=StandardScaler().fit(Yall[tr]); pca=PCA(NC).fit(sy.transform(Yall[tr]))
Ztr=sx.transform(feat(X[tr])); Str=pca.transform(sy.transform(Yall[tr]))
ss=StandardScaler().fit(Str)  # scale scores so multi-output models weigh them equally
models={
 "Gaussian Process (used)":None,
 "Ridge, polynomial deg 2":make_pipeline(PolynomialFeatures(2),RidgeCV(alphas=np.logspace(-6,2,20))),
 "Ridge, polynomial deg 3":make_pipeline(PolynomialFeatures(3),RidgeCV(alphas=np.logspace(-6,2,20))),
 "Neural network (MLP)":GridSearchCV(MLPRegressor(max_iter=5000,random_state=0,solver="lbfgs"),{"hidden_layer_sizes":[(16,),(32,32),(64,64)],"alpha":[1e-4,1e-2,1e-1]},cv=5),
 "Random forest":RandomForestRegressor(300,random_state=0),
 "Gradient boosting":MultiOutputRegressor(GradientBoostingRegressor(n_estimators=400,learning_rate=0.05,max_depth=3,random_state=0)),
}
gp=Proxy(NC).fit(X[tr],Yall[tr])
def metrics(P,T):
    p,t=split(P),split(T); r={}
    for k in SER: r[k]=np.mean(np.abs(p[k]-t[k]).sum(1)/np.abs(t[k]).sum(1))*100
    r["final_oil_cum"]=np.mean(np.abs(p["oil_cum"][:,-1]-t["oil_cum"][:,-1])/t["oil_cum"][:,-1])*100
    r["final_water_cum"]=np.mean(np.abs(p["water_cum"][:,-1]-t["water_cum"][:,-1])/t["water_cum"][:,-1])*100
    return r
rows=[]
for name,mdl in models.items():
    def predict(Xn):
        if mdl is None: return gp.predict(Xn)
        S=ss.inverse_transform(np.atleast_2d(mdl.predict(sx.transform(feat(Xn)))))
        return sy.inverse_transform(pca.inverse_transform(S))
    if mdl is not None: mdl.fit(Ztr,ss.transform(Str))
    row={"model":name}
    for nm,s in (("val",va),("test",te)):
        m=metrics(predict(X[s]),Yall[s]); row[nm]=m
    rows.append(row)
print(f"{'model':28s} | {'oil cum':>7s} {'gas cum':>7s} {'wat cum':>7s} {'oil rate':>8s} {'wat rate':>8s} | {'final oil':>9s} (test set, mean abs % error)")
for r in rows:
    t=r["test"]; print(f"{r['model']:28s} | {t['oil_cum']:7.2f} {t['gas_cum']:7.2f} {t['water_cum']:7.2f} {t['oil_rate']:8.2f} {t['water_rate']:8.2f} | {t['final_oil_cum']:9.2f}")
print("--- validation set")
for r in rows:
    t=r["val"]; print(f"{r['model']:28s} | {t['oil_cum']:7.2f} {t['gas_cum']:7.2f} {t['water_cum']:7.2f} {t['oil_rate']:8.2f} {t['water_rate']:8.2f} | {t['final_oil_cum']:9.2f}")
json.dump(rows,open(OUT/"model_comparison.json","w"),indent=1)
