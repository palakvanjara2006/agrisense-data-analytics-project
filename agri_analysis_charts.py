"""Week 3: illustrative agribusiness analysis and charts for AgriSense Farms.
Data are simulated in the layout of Agmarknet / DES / NASA POWER outputs (seeded, reproducible).
Run:  python agri_analysis_charts.py     -> writes charts/*.png and stats.json
"""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
os.makedirs("charts", exist_ok=True)
GREEN, AMBER, BLUE, GREY = "#1F5C3E", "#B7791F", "#2B6CB0", "#8A8A8A"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 1. Yield (kg/ha), 2019-2024, by crop and state
years = list(range(2019, 2025))
base = {("Wheat","Gujarat"):3300, ("Wheat","Madhya Pradesh"):3150, ("Rice","Gujarat"):2600,
        ("Rice","Madhya Pradesh"):2250, ("Soybean","Gujarat"):1350, ("Soybean","Madhya Pradesh"):1050}
rows = []
for (c, s), b in base.items():
    for i, y in enumerate(years):
        rows.append((c, s, y, b * (1 + 0.012*i) + rng.normal(0, b*0.05)))
yld = pd.DataFrame(rows, columns=["crop","state","year","yield_kg_ha"])
m = yld.groupby(["crop","state"])["yield_kg_ha"].mean().unstack()
fig, ax = plt.subplots(figsize=(6.2, 3.2))
x = np.arange(len(m)); w = 0.38
ax.bar(x-w/2, m["Gujarat"], w, color=GREEN, label="Gujarat")
ax.bar(x+w/2, m["Madhya Pradesh"], w, color=AMBER, label="Madhya Pradesh")
ax.set_xticks(x); ax.set_xticklabels(m.index); ax.set_ylabel("Mean yield (kg/ha)")
ax.set_title("Mean yield by crop and state, 2019-2024", loc="left", fontweight="bold")
for i, (g, p) in enumerate(zip(m["Gujarat"], m["Madhya Pradesh"])):
    ax.text(i-w/2, g+30, f"{g:,.0f}", ha="center", fontsize=7); ax.text(i+w/2, p+30, f"{p:,.0f}", ha="center", fontsize=7)
ax.legend(frameon=False); fig.tight_layout(); fig.savefig("charts/fig1_yield_by_crop_state.png", dpi=200); plt.close()

# 2. Monthly modal prices (Rs/quintal), 36 months, with seasonality + trend + noise
months = pd.period_range("2022-01", periods=36, freq="M")
def series(base, amp, peak, trend):
    t = np.arange(36); mo = np.array([p.month for p in months])
    return base + trend*t + amp*np.cos(2*np.pi*(mo-peak)/12) + rng.normal(0, base*0.012, 36)
price = pd.DataFrame({"month": months.to_timestamp(),
    "Wheat": series(2400, 120, 8, 6), "Soybean": series(4500, 380, 9, 9)})
price["Wheat_3m"] = price["Wheat"].rolling(3).mean(); price["Soybean_3m"] = price["Soybean"].rolling(3).mean()
import matplotlib.dates as mdates
fig, axs = plt.subplots(2, 1, figsize=(6.2, 3.8), sharex=True)
for ax, c, col in zip(axs, ["Wheat","Soybean"], [GREEN, AMBER]):
    ax.plot(price.month, price[c], color=col, alpha=.35, lw=1); ax.plot(price.month, price[c+"_3m"], color=col, lw=2, label=c+" (3-month mean)")
    ax.set_ylabel("Rs/quintal"); ax.legend(frameon=False, loc="upper left", fontsize=8)
axs[0].set_title("Monthly mandi modal price, 2022-2024", loc="left", fontweight="bold")
axs[1].xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,7])); axs[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
fig.tight_layout(); fig.savefig("charts/fig2_price_trend.png", dpi=200); plt.close()
growth = {c: (price[c+"_3m"].iloc[-1]/price[c+"_3m"].dropna().iloc[0]-1)*100 for c in ["Wheat","Soybean"]}

# 3. Cost of cultivation (Rs/ha)
cost = pd.DataFrame({"Seed":[3200,4800,4200],"Fertiliser":[7800,9500,6400],"Labour":[9500,18500,8200],
    "Irrigation":[6800,8200,2100],"Plant protection":[2300,3600,3100],"Machinery":[6400,5200,5800]}, index=["Wheat","Rice","Soybean"])
cost["Total"] = cost.sum(axis=1)
fig, ax = plt.subplots(figsize=(6.2, 3.2)); cols = [GREEN, AMBER, BLUE, "#5FA3A3", "#C0504D", GREY]
left = np.zeros(3)
for c, col in zip(cost.columns[:-1], cols):
    ax.barh(cost.index, cost[c], left=left, color=col, label=c); left += cost[c].values
for i, t in enumerate(cost["Total"]): ax.text(t+300, i, f"Rs {t:,.0f}", va="center", fontsize=8)
ax.set_xlim(0, 60000); ax.set_xlabel("Cost of cultivation (Rs/ha)"); ax.invert_yaxis()
ax.set_title("Cost structure by crop", loc="left", fontweight="bold"); ax.legend(frameon=False, ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5,-0.22))
fig.tight_layout(); fig.savefig("charts/fig3_cost_structure.png", dpi=200); plt.close()
revenue = pd.Series({"Wheat":3300*0.01*2450, "Rice":2600*0.01*2200, "Soybean":1200*0.01*4600})  # kg/ha -> quintal x price
margin = (revenue - cost["Total"]).round(0)

# 4. Rainfall vs soybean yield (district-seasons)
rain = rng.uniform(450, 1100, 60)
yv = -200 + 2.8*rain - 0.0016*rain**2 + rng.normal(0, 70, 60)
r, p = stats.pearsonr(rain, yv); sl, ic, *_ = stats.linregress(rain, yv)
fig, ax = plt.subplots(figsize=(6.2, 3.2))
ax.scatter(rain, yv, color=GREEN, alpha=.75, s=22); xs = np.linspace(rain.min(), rain.max(), 50)
co = np.polyfit(rain, yv, 2); pred = np.polyval(co, rain); r2q = 1-((yv-pred)**2).sum()/((yv-yv.mean())**2).sum(); peak = -co[1]/(2*co[0])
ax.plot(xs, np.polyval(co, xs), color=AMBER, lw=2, label="Quadratic fit"); ax.axvline(peak, color=GREY, ls=":", lw=1); ax.legend(frameon=False, loc="lower right")
ax.set_xlabel("Seasonal rainfall (mm)"); ax.set_ylabel("Soybean yield (kg/ha)")
ax.set_title(f"Rainfall and soybean yield (linear r = {r:.2f}, quadratic R2 = {r2q:.2f})" if False else f"Rainfall and soybean yield (linear r = {r:.2f})", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig("charts/fig4_rain_vs_yield.png", dpi=200); plt.close()

# 5. Seasonal index of wheat price
price["mo"] = price.month.dt.month
si = (price.groupby("mo")["Wheat"].mean()/price["Wheat"].mean()*100)
fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.bar(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], si.values, color=[GREEN if v>=100 else AMBER for v in si.values])
ax.axhline(100, color="#444", lw=1, ls="--"); ax.set_ylim(90, 110); ax.set_ylabel("Index (average = 100)")
ax.set_title("Seasonal index of wheat modal price", loc="left", fontweight="bold")
fig.tight_layout(); fig.savefig("charts/fig5_seasonal_index.png", dpi=200); plt.close()

# significance test: Gujarat vs MP soybean yield
g = yld[(yld.crop=="Soybean")&(yld.state=="Gujarat")].yield_kg_ha; mp = yld[(yld.crop=="Soybean")&(yld.state=="Madhya Pradesh")].yield_kg_ha
t, tp = stats.ttest_ind(g, mp, equal_var=False)
out = {"yield_means": m.round(0).to_dict(), "price_growth_pct": {k: round(v,1) for k,v in growth.items()},
 "cost_total": cost["Total"].to_dict(), "margin": margin.to_dict(), "r": round(r,2), "r2": round(r*r,2), "p": float(p),
 "slope": round(sl,2), "r2q": round(r2q,2), "peak": round(peak), "t": round(t,2), "tp": float(tp), "gap_pct": round((g.mean()/mp.mean()-1)*100,1),
 "si_peak": [int(si.idxmax()), round(si.max(),1)], "si_low": [int(si.idxmin()), round(si.min(),1)], "n": 60}
json.dump(out, open("stats.json","w"), indent=1); print(json.dumps(out, indent=1))
