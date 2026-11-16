"""
AgriSense Farms Pvt Ltd - Week 2: Data collection and cleaning pipeline
-----------------------------------------------------------------------
Implements the plan in "Week2_Data_Collection_and_Cleaning_Plan.docx":
  1. collect_nasa_power()   - daily weather from the NASA POWER API (no key needed)
  2. clean_mandi_prices()   - the nine-step cleaning order for mandi price data
  3. profile()              - read-only data profile (Section 4)
Every change is flagged and written to a cleaning log (Appendix C).

Usage:
    python agri_cleaning_pipeline.py raw_mandi_prices.csv cleaned_mandi_prices.csv
    python agri_cleaning_pipeline.py --demo        # runs on built-in sample data
Expected raw columns: date, market, district, commodity, variety,
                      min_price, max_price, modal_price, arrivals
Prices are Rs per quintal; arrivals are in tonnes (Section 5.4).
"""
import sys
from difflib import get_close_matches

import numpy as np
import pandas as pd
import requests

# ----------------------------------------------------------------- reference
COMMODITY_REF = {"wheat": "Wheat", "rice": "Rice", "paddy": "Rice",
                 "soybean": "Soybean", "soyabean": "Soybean", "soya bean": "Soybean"}
MISSING_CODES = ["-999", "-999.0", "NA", "N/A", "na", "", "null", "-"]
MAX_FILL_DAYS = 3        # carry-forward limit for price gaps (Section 5.2)
MAD_Z_LIMIT = 3.5        # robust z-score flag threshold (Section 5.3)


# ----------------------------------------------------------------- collection
def collect_nasa_power(lat, lon, start, end, out_csv=None):
    """Daily rainfall, Tmax, Tmin, humidity for one point. start/end = 'YYYYMMDD'."""
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {"parameters": "PRECTOTCORR,T2M_MAX,T2M_MIN,RH2M", "community": "AG",
              "longitude": lon, "latitude": lat, "start": start, "end": end,
              "format": "JSON"}
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["properties"]["parameter"]
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999, np.nan)            # -999 is the fill value for missing
    df.index.name = "date"
    if out_csv:
        df.to_csv(out_csv)
    return df


# ----------------------------------------------------------------- helpers
class CleaningLog:
    """One row per rule that changes, flags or removes data (Appendix C)."""
    def __init__(self, dataset):
        self.dataset, self.rows = dataset, []

    def add(self, stage, rule, n, action):
        self.rows.append({"log_id": f"CL-{len(self.rows)+1:03d}", "dataset": self.dataset,
                          "stage": stage, "rule": rule, "rows_affected": int(n),
                          "action": action})

    def to_frame(self):
        return pd.DataFrame(self.rows)


def standardise(value, ref=COMMODITY_REF, cutoff=0.8):
    """Exact match, then fuzzy match. None goes to the manual review list."""
    v = str(value).strip().lower()
    if v in ref:
        return ref[v]
    m = get_close_matches(v, list(ref), n=1, cutoff=cutoff)
    return ref[m[0]] if m else None


def profile(df, keys):
    """Read-only profile: structure, completeness, uniqueness (Section 4)."""
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    print("Missing % per column:\n", (df.isna().mean() * 100).round(1).to_string())
    print("Duplicate key rows:", int(df.duplicated(subset=keys).sum()))
    return df.describe(include="all").T


# ----------------------------------------------------------------- cleaning
def clean_mandi_prices(raw):
    log = CleaningLog("mandi_prices v1")
    df = raw.copy()                                   # Stage 0: raw is never edited

    # Stage 1 - structural standardisation
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.replace(MISSING_CODES, np.nan)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    bad_dates = df["date"].isna().sum()
    df = df.dropna(subset=["date"])
    log.add("1. Structure", "Unparseable or impossible dates removed", bad_dates, "Removed")
    for c in ["min_price", "max_price", "modal_price", "arrivals"]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ""), errors="coerce")

    # Stage 2 - categorical standardisation
    for col in ["commodity", "market", "district"]:
        df[col] = df[col].astype(str).str.strip().str.title()
    before = df["commodity"].copy()
    df["commodity"] = df["commodity"].map(standardise)
    unmatched = df["commodity"].isna().sum()
    log.add("2. Names", "Commodity mapped to master list", (before.str.lower() !=
            df["commodity"].str.lower()).sum(), "Standardised")
    log.add("2. Names", "No match, sent to manual review", unmatched, "Quarantined")
    df = df.dropna(subset=["commodity"])

    # Stage 3 - duplicates (key = date, market, commodity, variety; keep latest)
    key = ["date", "market", "commodity", "variety"]
    n = df.duplicated(subset=key, keep="last").sum()
    df = df.drop_duplicates(subset=key, keep="last")
    log.add("3. Duplicates", "Key = date, market, commodity, variety; keep latest", n, "Removed")

    # Stage 6 (before 4/5 so rules are judged on correct units) - integrity rules
    df = df.sort_values(["commodity", "market", "date"]).reset_index(drop=True)
    ok = ((df.min_price > 0) & (df.min_price <= df.modal_price) &
          (df.modal_price <= df.max_price)) | df.modal_price.isna()
    df["rule_fail"] = ~ok
    log.add("6. Integrity", "min <= modal <= max and price > 0", (~ok).sum(), "Flagged")
    df.loc[~ok, ["min_price", "modal_price", "max_price"]] = np.nan   # fail = null + log

    # Stage 4 - missing values: carry forward max 3 days, flag, keep arrivals unknown
    grp = ["commodity", "market"]
    was_na = df.modal_price.isna()
    df["modal_price"] = df.groupby(grp)["modal_price"].transform(
        lambda s: s.ffill(limit=MAX_FILL_DAYS))
    df["price_imputed"] = was_na & df.modal_price.notna()
    log.add("4. Missing", f"Modal price carried forward <= {MAX_FILL_DAYS} days",
            df.price_imputed.sum(), "Imputed, flagged")
    df["arrivals_unknown"] = df["arrivals"].isna()    # unknown, never silently zero

    # Stage 5 - outliers: robust z-score within crop, market, calendar month
    g = df.groupby(grp + [df["date"].dt.month])["modal_price"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    z = 0.6745 * (df.modal_price - med).abs() / mad.replace(0, np.nan)
    df["outlier"] = z > MAD_Z_LIMIT
    log.add("5. Outliers", f"Robust z above {MAD_Z_LIMIT}", df.outlier.sum(),
            "Flagged (kept in cleaned layer)")

    # Stage 7 - integration keys and season label
    df["season"] = np.where(df.date.dt.month.isin([6, 7, 8, 9, 10]), "Kharif",
                    np.where(df.date.dt.month.isin([11, 12, 1, 2, 3, 4]), "Rabi", "Zaid"))
    df["ag_year"] = np.where(df.date.dt.month >= 7, df.date.dt.year, df.date.dt.year - 1)

    # Stage 8 - validation tests before publishing
    assert not df.duplicated(subset=key).any(), "duplicate keys remain"
    assert df.commodity.isin(set(COMMODITY_REF.values())).all(), "unknown commodity"
    return df, log.to_frame()


# ----------------------------------------------------------------- demo / main
def demo_data():
    rng = np.random.default_rng(1)
    days = pd.date_range("2025-01-01", periods=60, freq="D")
    d = pd.DataFrame({"Date": days.strftime("%d/%m/%Y"), "Market": " rajkot ",
                      "District": "Rajkot", "Commodity": "Soyabean", "Variety": "Yellow",
                      "Min Price": rng.normal(4300, 40, 60).round(),
                      "Max Price": rng.normal(4700, 40, 60).round(),
                      "Modal Price": rng.normal(4500, 30, 60).round(),
                      "Arrivals": rng.normal(120, 10, 60).round()}).astype(str)
    d.loc[5, "Modal Price"] = "-999"          # missing code
    d.loc[6, "Modal Price"] = "-999"
    d.loc[20, "Modal Price"] = "450000"        # unit error
    d.loc[30, "Min Price"] = "9000"           # min > modal
    d.loc[40, "Commodity"] = "Soya bean"      # spelling variant
    d.loc[45, "Commodity"] = "Mustard"        # not in master list
    d = pd.concat([d, d.iloc[[10]]])          # duplicate row
    return d


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--demo":
        raw, out = demo_data(), "cleaned_demo.csv"
    elif len(sys.argv) == 3:
        raw, out = pd.read_csv(sys.argv[1], dtype=str), sys.argv[2]
    else:
        sys.exit(__doc__)
    clean, log = clean_mandi_prices(raw)
    clean.to_csv(out, index=False)
    log.to_csv(out.replace(".csv", "_cleaning_log.csv"), index=False)
    print(log.to_string(index=False))
    print(f"\nCleaned rows: {len(clean)} -> {out}")
