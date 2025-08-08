#!/usr/bin/env python
import os, pandas as pd

OUT = os.environ.get("OUT_PATH","data/german_credit.csv")
URL = os.environ.get("SRC_URL","https://raw.githubusercontent.com/selva86/datasets/master/GermanCredit.csv")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df = pd.read_csv(URL)
# Normalize column names (no spaces)
df.columns = [c.replace(' ', '_') for c in df.columns]
df.to_csv(OUT, index=False)
print(f"Wrote {OUT} with {df.shape[0]} rows, {df.shape[1]} columns.")
