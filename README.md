# Epitrix — Hybrid Immune Simulation Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://epitrix-kxsv2rqbm9bmcj5ywphybo.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**Live app:** https://epitrix-kxsv2rqbm9bmcj5ywphybo.streamlit.app

---

## What is Epitrix?

Epitrix is an open-source hybrid platform for vaccine formulation research that combines a trained XGBoost machine learning model for MHC-I epitope prediction (AUC-ROC 0.986 on human HLA alleles; 0.970 on mouse H-2 alleles), trained on 219,000+ peptide-MHC binding measurements from the IEDB, with a parameterised mechanistic simulation of the innate->adaptive immune cascade, with equation coefficients manually derived from 22 peer-reviewed publications, and 95% confidence intervals on all outputs derived from published inter-individual biological variability data.

Epitrix is designed for hypothesis generation in early-stage vaccine formulation work. It is not a validated clinical tool.

---

## Architecture

```
Antigen sequence  -->  XGBoost MHC-I/II model  -->  Epitope scores
                        (trained, AUC 0.986)

LNP formulation   -->  Mechanistic equations    -->  Innate activation
+ Modifications        (parameterised)               Adaptive cascade
                                                      Clinical outcomes
                                                      +/- 95% CI
```

The epitope prediction component is genuinely ML-powered. The immune cascade component is a mechanistic simulation, not an end-to-end trained model. This distinction is clearly labelled throughout the app.

---

## Key capabilities

| Module | Method | Performance |
|---|---|---|
| MHC-I epitope scan (human) | XGBoost, trained on IEDB | AUC-ROC 0.986, F1 0.930 |
| MHC-I epitope scan (mouse) | XGBoost, trained on IEDB | AUC-ROC 0.970, F1 0.986 |
| MHC-II epitope scan | IEDB NetMHCIIpan (API) + PSSM fallback | External benchmark |
| LNP physicochemical | Mechanistic equations | Fitted to published data |
| Innate pathway activation | Mechanistic equations | Parameterised |
| Adaptive immune cascade | Mechanistic equations | Parameterised |
| DC programming | Mechanistic equations | Parameterised |
| Clinical reactogenicity | Mechanistic equations | Parameterised |
| Population stratification | Published modifiers | Crooke et al. 2019 |
| Formulation optimizer | 720-combination sweep | Deterministic |

---

## ML model benchmark

Epitrix human MHC-I model vs published tools on HLA-A*02:01 9-mer prediction:

| Model | AUC-ROC | Notes |
|---|---|---|
| Epitrix XGBoost (this work) | 0.986 | Held-out IEDB split (n=32,978) |
| NetMHCpan 4.1 | 0.970 | Reynisson et al. NAR 2020 |
| MHCflurry 2.0 | 0.960 | O'Donnell et al. Cell Syst 2020 |
| NetMHCpan-EL | 0.950 | Jurtz et al. J Immunol 2017 |
| MHCnuggets | 0.930 | Shao et al. Cell Syst 2020 |
| SYFPEITHI | 0.890 | Rammensee et al. 1999 |
| PSSM baseline | 0.820 | Epitrix prior to ML upgrade |

Important caveat: The Epitrix test set was randomly split from the same IEDB download used for training. Published benchmarks use independently curated held-out sets. A fully rigorous comparison would require testing on an independent dataset such as the CAMEL benchmark or post-download publications. The scores are indicative, not definitive.

---

## Quick start

```bash
git clone https://github.com/btouray1/epitrix.git
cd epitrix
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

### IEDB connectivity (optional)

```bash
curl -s -L -X POST https://tools-cluster-interface.iedb.org/tools_api/mhci/ \
  -d "method=recommended&sequence_text=NITNLCPFGEVFNATR&allele=HLA-A*02:01&length=9" | head -2
```

If you see TSV output, IEDB is reachable. Epitrix uses it for MHC-II predictions and as a fallback if the local ML model is unavailable.

---

## Project structure

```
epitrix/
├── app.py                              Main Streamlit application
├── requirements.txt                    Runtime dependencies
├── README.md
├── .streamlit/
│   └── config.toml                     Theme (forces light mode)
└── epitrix_ml/
    ├── models/
    │   ├── mhci_xgboost_human.pkl      Trained human HLA model (1.5 MB)
    │   ├── mhci_xgboost_mouse.pkl      Trained mouse H-2 model (0.1 MB)
    │   └── mhci_xgboost_combined.pkl   Combined model (1.5 MB)
    ├── 01_download_iedb.py             Download IEDB bulk data
    ├── 02_process_mhci.py              Feature engineering pipeline
    ├── 03_train_mhci_model.py          Model training and evaluation
    ├── 04_integrate_epitrix.py         Integration test
    ├── requirements_ml.txt             ML pipeline dependencies
    └── README.md                       ML pipeline documentation
```

---

## Reproducing the ML models

```bash
pip install -r epitrix_ml/requirements_ml.txt

# Download mhc_ligand_full.csv from https://www.iedb.org/database_export_v3.php
# Place in data/raw/mhc_ligand_full.csv (~8.75 GB)

python3 epitrix_ml/02_process_mhci.py      # ~15 min, single-pass
python3 epitrix_ml/03_train_mhci_model.py  # ~20 min, trains 3 models
python3 epitrix_ml/04_integrate_epitrix.py # integration test
```

Results saved to results/human/, results/mouse/, results/combined/.

---

## Known limitations

**ML model:**
- Trained and tested on a random split of the same IEDB download. Performance on truly unseen data may differ.
- Covers 9-mer peptides only.
- Human model is HLA-A*02:01-centric. Performance varies across alleles.
- Mouse model has limited weak-binder training data (828 examples).

**Mechanistic simulation:**
- All innate->adaptive equations are simplifications of complex biology.
- Coefficients were manually extracted from a small number of publications, not statistically fitted.
- No dose, route of administration, or adjuvant variables are modelled.
- Predictions assume a single vaccination dose unless population stratification is selected.
- Formulation optimizer covers 720 combinations, a small fraction of the real design space.

**General:**
- Not validated against prospective experimental data.
- Not a medical device. Not for clinical use.
- Should not inform decisions about human subjects.

---

## Evidence base

Full citations with DOI links are in the Evidence Base tab of the app. Key sources:

- LNP formulation: Hassett et al. (NPJ Vaccines 2019), Kulkarni et al. (Nano Letters 2021)
- TLR activation: Kariko et al. (Immunity 2005), Andries et al. (Nature Biotechnology 2015)
- Innate-adaptive bridge: Miao et al. (Nature Biotechnology 2019), Liang et al. (Nature Communications 2021)
- Adaptive outcomes: Walsh et al. (NEJM 2020), Goldberg et al. (Science 2021), Crotty et al. (Science 2021)
- Epitope scoring: Reynisson et al. (NAR 2020), Rammensee et al. (Immunogenetics 1999)
- Biological variability CIs: Voysey et al. (Lancet 2021), Crooke et al. (npj Vaccines 2019)

---

## Roadmap

- [ ] Independent benchmark validation on CAMEL or post-2024 IEDB additions
- [ ] Multi-allele coverage (HLA-A*01:01, HLA-B*07:02, HLA-B*57:01)
- [ ] Replace mechanistic cascade equations with trained ML models
- [ ] Dose and route of administration as input parameters
- [ ] HLA population frequency weighting for global coverage estimates
- [ ] Prime-boost schedule modelling
- [ ] PDF export of prediction report
- [ ] ESM-2 protein embeddings to replace hand-crafted physicochemical features

---

## Citation

```bibtex
@software{touray2025epitrix,
  author = {Touray, Bubacarr},
  title  = {Epitrix: A Hybrid Mechanistic and Machine Learning Platform
            for Vaccine Formulation and Immune Simulation},
  year   = {2025},
  url    = {https://github.com/btouray1/epitrix},
  note   = {XGBoost MHC-I model: AUC-ROC 0.986 (human HLA),
            0.970 (mouse H-2). Trained on IEDB bulk MHC ligand data.}
}
```

Plain text:

> Touray, B. (2025). Epitrix: A Hybrid Mechanistic and Machine Learning Platform for Vaccine Formulation and Immune Simulation. GitHub. https://github.com/btouray1/epitrix

---

## License

MIT License. Free to use, modify, and distribute with attribution.

---

## Contact

Bubacarr Touray, PhD
Molecular Laboratory Supervisor, Exact Sciences

GitHub Issues: bug reports and feature requests. Pull requests welcome.

---

> Epitrix is a research tool built for hypothesis generation in vaccine formulation science. It is not a medical device, has not been clinically validated, and must not be used to inform decisions about human subjects. All outputs should be independently verified through experimental validation.
