"""One panel: the frame drawn with the poly.py of the checked-out tree. Usage: panel.py frame.pkl out.png title"""
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import landau.plot as lpl

frame, out, title = sys.argv[1:4]
df = pd.read_pickle(frame)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    polys = lpl.get_polygons(df, poly_method="fasttsp")
    print(title, "| stable phases in frame:", sorted(df.query("stable").phase.unique()), "| polygons:", sorted(k[0] for k, _ in polys.items()))
    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
    lpl.plot_phase_diagram(df, poly_method="fasttsp", ax=ax)
ax.set_title(title, fontsize=10)
fig.savefig(out, dpi=150)
