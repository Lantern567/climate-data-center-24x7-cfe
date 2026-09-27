# Climate change and hourly carbon-free electricity for data centres

This repository contains the study's analysis code and country/global result tables for climate-sensitive data-centre demand, wind–solar matching, capacity planning and technology learning. Original calculation files are archived without changes; `source_manifest.json` records their SHA-256 hashes and those of the result files.

## Reproduce analyses from the included results

The aggregate analyses were verified with Python 3.12.14. Install the analysis dependencies in an isolated Python 3.12 environment:

```sh
python -m pip install -r requirements-analysis.txt
python scripts/verify_release.py
python scripts/plot_v69_figures.py
python scripts/plot_v71_figure3b.py
```

The verifier checks file hashes, the additivity of the climate decomposition, annualized cost components, and the budget/learning identities using the actual archived results. The plotting commands reproduce the technology-learning figures and the revised capacity-cost figure. Outputs are written under `output/figures/`. The second command uses the included original Figure 3 PDF as its layout source for the unchanged panels. Historical script and figure numbers are retained for traceability.

These commands use the released aggregate data. They do not rerun the original hourly climate simulations or the full capacity optimization.

## Code and result files

| Analysis | Calculation code in `research_pipeline/scripts/` | Included results under `output/` |
|---|---|---|
| Historical climate replay and future climate construction | Scripts 24–57; `research_pipeline/supply_resource_models.py` | `multimodel_analysis_20260913/` |
| Paired climate effects and matching windows | `66_compute_geographic_climate_conditioning.py`, `result1_remote_worker.py` | `result1_full_matrix_20260912/` |
| Annual volume and timing decomposition | `62_decompose_level_and_shape.py`, `69_fixed_volume_mechanisms.py`, `result234_remote_worker.py` | `multimodel_analysis_20260913/result234_mechanisms_*.csv` |
| Persistent low-matching episodes | `result1_persistence_20260914.py` | `persistence_20260914/` |
| Capacity planning and hourly matching | `greenfield_*.py`, `greenfield_kernel_20260915.cpp` | `greenfield_cfe_20260915/` |
| Commissioning-year technology learning | `commissioning_learning_20260919.py`, `complete_matching_technology_20260917.py` | `commissioning_learning_20260919/` |

The learning results include 99,615 country-pool plans across 145 pools and nine future climate states. The cost parameters are in `output/rewrite_v11/future/price_interface_2030.json`. JSON validation reports accompanying the results record checks performed during the original calculations; the separate release verifier checks the distributed tables.

In matching tables, `uncovered_mwh` is the sum of positive hourly supply deficits over a weather year and across facilities. For `worst_168h_mwh`, the largest deficit sum in a contiguous 168-hour window is first calculated separately for each facility; these facility maxima are then summed. The maximizing windows can occur at different times across facilities. Surpluses do not cancel deficits at other hours. A climate increment is the future minus historical value for paired weather years. Decomposition contributions are signed and add to that increment; a negative contribution offsets its increase. Energy columns ending in `_mwh` use MWh (divide by 1,000 for GWh). See the equations in the calculation scripts for scenario definitions.

## Full simulation inputs and environment

The original simulations use the 1995–2014 hourly weather sequence, climate-model perturbations, facility inputs and wind/solar inventories. The repository distributes country/global derived tables, rather than the large hourly arrays or third-party facility inventories. Obtain the upstream inputs under their source access conditions. Data acquisition scripts identify the sources, including ERA5, NEX-GDDP-CMIP6, Global Energy Monitor, Ember, IRENA and EIA; the original facility source survey is at https://github.com/Lantern567/global-data-center-poi-research. Technology cost metadata are recorded in the included cost-parameter JSON.

Full reruns require reconstruction of the input directory trees expected by each script, a Linux environment, and a C++ compiler for `greenfield_kernel_20260915.cpp`. Some archived scripts retain the original server filesystem root in their `ROOT` setting. Configure those paths for the new environment before running them. The commissioning-year driver accepts `--root`, `--workers` and `--stage`; it expects prepared pool metadata, hourly arrays and the compiled kernel under that root. The original replay dependency lists are retained in `research_pipeline/requirements_*.txt`; they belong to the original replay environment, separately from the analysis environment above.

The repository contains research code and derived research outputs. Upstream data remain subject to their respective source terms.
