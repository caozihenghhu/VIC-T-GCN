VIC-T-GCN
=========

Code and example data for the VIC-T-GCN framework, in which a flow-direction-guided
temporal graph convolutional network (T-GCN) serves as a routing surrogate for runoff
and baseflow simulated by the Variable Infiltration Capacity (VIC) model.


1. PACKAGE CONTENTS
-------------------

  VIC_TGCN_Scheme_I.py        Scheme I: dual-graph model (surface runoff + baseflow,
                              surface + subsurface flow-direction graphs)
  VIC_TGCN_Schemes_II_VI.py   Comparison schemes II-V (grid-based) and VI
                              (station-network model)
  requirements.txt            Pinned Python dependencies (install PyTorch separately,
                              see below)
  configs/                    Executable JSON configurations for the Danba example and
                              a Scheme VI template
  data/danba/                 Example inputs for the Danba basin


2. INSTALLATION
---------------

Python 3.12. Install PyTorch 2.8.0 for your CPU/CUDA environment first
(see https://pytorch.org), then:

    pip install -r requirements.txt


3. QUICK START
--------------

    python VIC_TGCN_Scheme_I.py --self-test
    python VIC_TGCN_Schemes_II_VI.py --self-test
    python VIC_TGCN_Scheme_I.py --preflight
    python VIC_TGCN_Scheme_I.py
    python VIC_TGCN_Schemes_II_VI.py --config configs/danba.json

--self-test runs synthetic implementation checks (flow-direction encoding, physical
time ordering of the GCN-LSTM input, forward/backward passes, date causality of the
input windows, and the NSE-RMSE identity); it does not produce scientific results.
--preflight validates the inputs, graphs, data split and training-only scaling without
training a model.

Each station/scheme/seed produces one results.xlsx workbook and one test.png
hydrograph. Each training run also saves reproduction_checkpoint.pt for later
inference. The workbook contains Metrics, Train, Validation and Test sheets.
Discharge, MAE and RMSE are in m3/s; NSE and KGE are dimensionless. Metrics are
computed from unrounded predictions; Excel and plot labels display two decimals.
Only test-period observations and simulations are plotted, with all four metrics
annotated.


4. EXPERIMENTAL SETUP
---------------------

- Data are split chronologically into 65% training, 10% validation and 25% testing
  by target date.
- Min-max scaling of each input channel is fitted across all nodes using training
  input days only; the target scaling uses training targets only. For Scheme VI, the
  station-level PCA is also fitted on training input days only.
- The test period is used only for final evaluation; early stopping and
  learning-rate scheduling use the validation set.
- Default hyperparameters: GCN hidden size 32, LSTM hidden size 128, dropout 0.3,
  Adam (learning rate 0.001, weight decay 1e-4), batch size 8, up to 200 epochs,
  early-stopping patience 15, seed 42.


5. SAVED MODELS AND NEW PREDICTIONS
-----------------------------------

The checkpoint contains the selected model weights, model settings, node order, graph
edges, and fitted scaling parameters. Scheme VI also includes the fitted PCA
transformations. Use the corresponding script to load its checkpoint; inference
applies the saved preprocessing without refitting. These are inference checkpoints,
not optimizer-state checkpoints for resuming training.

For Scheme I, provide new runoff and baseflow tables with the original node column
names and consecutive dates. Observed discharge is not required:

    python VIC_TGCN_Scheme_I.py --mode predict --checkpoint path/to/reproduction_checkpoint.pt --runoff data/my_basin/runoff.xlsx --baseflow data/my_basin/baseflow.xlsx --output-dir predictions

For Schemes II-VI, provide a configuration pointing to the new input tables,
retaining the trained station names:

    python VIC_TGCN_Schemes_II_VI.py --mode predict --checkpoint path/to/reproduction_checkpoint.pt --config configs/my_basin.json --output predictions

Run a small synthetic Scheme I training-and-inference demonstration with:

    python VIC_TGCN_Scheme_I.py --mode demo --output-dir demo_results


6. PREPARE ANOTHER BASIN FOR SCHEMES I-V
----------------------------------------

Create a directory such as data/my_basin/ containing the following five files. Use
the supplied Danba files as format templates, replacing all time series and spatial
inputs with those for the intended basin.

  runoff.xlsx               Daily observed outlet discharge and surface runoff for
                            each grid node
  baseflow.xlsx             The same dates and observations, with baseflow for each
                            grid node
  nodes.xlsx                Node indices, identifiers, coordinates and input column
                            names
  surface_direction.txt     Surface flow-direction ASCII grid
  baseflow_direction.txt    Subsurface flow-direction ASCII grid

6.1 Runoff and baseflow workbooks

Put the table on the first worksheet, with headers in the first row:

  Date         Qobs                          1               2               ...
  YYYY-MM-DD   Observed outlet discharge     Node 1 flux     Node 2 flux     ...

- Column 1: actual daily dates, preferably Excel dates or YYYY-MM-DD strings.
- Column 2: observed outlet discharge in m3/s.
- Columns 3 onward: one column per grid node, in mm/day. Use surface runoff in
  runoff.xlsx and baseflow in baseflow.xlsx.
- Both workbooks must contain exactly the same dates and observed discharge values.
- Dates must be continuous and unique. Fluxes and observations must be finite,
  nonnegative numeric values without missing entries.
- Node column names must match the corresponding names in nodes.xlsx. The loader
  aligns input columns by name, not by position.

6.2 Node-coordinate workbook

Create a worksheet named Nodes, with these exact headers:

  node_index   node_id    lat        lon         surface_column   baseflow_column
  0            node_001   Latitude   Longitude   1                1
  1            node_002   Latitude   Longitude   2                2

Replace the latitude/longitude placeholders with actual numeric coordinates.

- node_index: consecutive integers from 0 to N-1, without duplicates or gaps. This
  fixes the model node order.
- node_id: a unique identifier for each node.
- lat, lon: latitude and longitude in EPSG:4326, normally at grid-cell centers.
- surface_column, baseflow_column: exact column header names in the respective flux
  workbooks. These are identifiers, not flux values.
- Each node must map to a distinct cell in each direction raster. Out-of-range
  coordinates and multiple nodes mapping to the same cell are rejected.


7. FLOW-DIRECTION FILES
-----------------------

Use ESRI ASCII-grid file structure, with six header lines followed by the integer
matrix:

    ncols         <number of columns>
    nrows         <number of rows>
    xllcorner     <western boundary longitude>
    yllcorner     <southern boundary latitude>
    cellsize      <cell size in degrees>
    NODATA_value  255
    <direction matrix: nrows rows and ncols values per row>

Rows run from north to south and columns from west to east. Nodes and rasters must
share the expected geographic coordinate system and compatible grid locations.

The supplied settings use 1=N, 2=NE, 3=E, 4=SE, 5=S, 6=SW, 7=W, 8=NW. Code 0 denotes
no assigned direction; 255 is NoData. ESRI ASCII-grid file structure does not imply
ESRI powers-of-two direction encoding: convert such direction values to the numbering
above before using the default settings. Prepare projected rasters in EPSG:4326
before using this loader.

The surface flow directions were derived from the DEM using the D8 algorithm. The
subsurface flow directions were derived from the gradient of gridded groundwater
heads (Wang et al., 2025; https://doi.org/10.6084/m9.figshare.27160197), assigning
each cell the direction of the steepest head decrease. The period of the head field,
the resampling procedure and checksums of the source data used for the supplied
Danba raster are recorded in data/danba/groundwater_provenance.json.


8. CONFIGURE AND RUN ANOTHER BASIN
----------------------------------

Copy configs/danba.json to configs/my_basin.json. Replace the "basins" list with the
following, retaining the surrounding JSON structure and other settings:

    "basins": [
      {
        "name": "my_basin",
        "runoff_excel": "../data/my_basin/runoff.xlsx",
        "baseflow_excel": "../data/my_basin/baseflow.xlsx",
        "node_table": "../data/my_basin/nodes.xlsx",
        "flow_dir_surface": "../data/my_basin/surface_direction.txt",
        "flow_dir_baseflow": "../data/my_basin/baseflow_direction.txt",
        "output_dir": "../results/my_basin/scheme_I"
      }
    ]

Relative input paths are resolved from the configuration file's directory. Run
commands from the extracted package directory and specify the basin explicitly:

    # Check inputs without training
    python VIC_TGCN_Scheme_I.py --config configs/my_basin.json --basin my_basin --preflight
    python VIC_TGCN_Schemes_II_VI.py --config configs/my_basin.json --basin my_basin --preflight

    # Run Scheme I
    python VIC_TGCN_Scheme_I.py --config configs/my_basin.json --basin my_basin

    # Run Schemes II-V
    python VIC_TGCN_Schemes_II_VI.py --config configs/my_basin.json --basin my_basin --schemes 2 3 4 5 --seeds 42 --output results/my_basin/schemes_II_V

To run only one comparison scheme, use, for example, --schemes 3. Scheme I uses the
"seed" value inside the configuration's "training" object; the supplied value is 42.
The comparison script's --output path is relative to the current working directory.


9. RUN SCHEME VI (STATION NETWORK)
----------------------------------

Scheme VI is a joint station-network model in which nodes are gauging stations and
directed edges follow upstream-downstream relationships. To run it, prepare
runoff.xlsx and baseflow.xlsx for every station in the format described in Section 6,
with the same daily date range at all stations, and supply sufficient node columns
for the 15-component PCA representation of each station.

Station connectivity is read from a CSV file referenced by "station_edges_file" in
the configuration. Its two columns are "upstream" and "downstream", and each row
defines one directed upstream-to-downstream edge; names must match the station inputs.
Relative paths are resolved from the configuration directory. The code checks unknown
names, duplicate edges, self-connections, cycles, and graph connectivity.

Update the input paths in configs/scheme_VI_five_stations.template.json, then run:

    python VIC_TGCN_Schemes_II_VI.py --config configs/scheme_VI_five_stations.template.json --schemes 6 --seeds 42 --preflight
    python VIC_TGCN_Schemes_II_VI.py --config configs/scheme_VI_five_stations.template.json --schemes 6 --seeds 42 --output results/scheme_VI

The station order is defined by STATION_ORDER in the comparison script; the station
edge CSV and the configuration's "basins" must be updated consistently with it. The
present implementation uses five stations; changing that number requires
corresponding code changes.


10. DATA
--------

The Danba basin is provided as a complete worked example in data/danba/: the gridded
VIC surface-runoff and baseflow series (2009-2020), observed outlet discharge, the
node table, and the surface and subsurface flow-direction rasters. Schemes I-V can be
run end to end with these files, and the files serve as format templates for
preparing other basins (Sections 6-8).

Third-party datasets. Meteorological forcing (CN05.1), soil (HWSD), vegetation
(UMD 1 km land cover), elevation (Geospatial Data Cloud DEM) and groundwater heads
(Wang et al., 2025; https://doi.org/10.6084/m9.figshare.27160197) are available from
their original providers, as listed in the Open Research section of the manuscript.
