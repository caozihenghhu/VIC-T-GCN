# Scheme VI: real 2009 subset

This example contains the original uncalibrated runoff/baseflow inputs and observed discharge for Ribu, Yilinchang, Dajin, Danba, and Xiaojin from 2009-01-01 through 2009-12-31. Each file contains 365 consecutive daily rows. All original node columns and numerical values are retained. No interpolation, rescaling, or synthetic replacement was applied.

The first CSV column contains dates, the second contains observed outlet discharge (m3/s), and subsequent columns contain runoff or baseflow inputs (mm/day). The two channel files for each station contain identical observed discharge. Node counts are 277 (Ribu), 176 (Yilinchang), 624 (Dajin), 805 (Danba), and 69 (Xiaojin). Column names preserve the source mapping. File checksums are recorded in subset_manifest.json.

Station connectivity is supplied in station_edges.csv, with columns upstream and downstream. It matches the original comparison code: Ribu to Dajin, Yilinchang to Dajin, Dajin to Danba, and Xiaojin to Danba. These are directed station connections, not grid D8 flow codes.

Run from the package root:

```bash
python VIC_TGCN_Schemes_II_VI.py --config configs/scheme_VI_2009.json --preflight
python VIC_TGCN_Schemes_II_VI.py --config configs/scheme_VI_2009.json --output results/scheme_VI_2009
```

The example configuration uses CPU and two epochs for a short execution test, retaining the model dimensions. It produces five station result workbooks, five test hydrographs, and one reproduction_checkpoint.pt. The truncated period and short training run do not reproduce the full-period scores in the manuscript.

Locate reproduction_checkpoint.pt inside the timestamped output directory and replace CHECKPOINT_PATH below with its path:

```bash
python VIC_TGCN_Schemes_II_VI.py --mode predict --config configs/scheme_VI_2009.json --checkpoint "CHECKPOINT_PATH" --output results/scheme_VI_2009_reload
```

Reloading the same data checks saved-model inference; it is not an additional independent scientific evaluation. For a full-period experiment, provide complete five-station inputs using configs/scheme_VI_five_stations.template.json.
