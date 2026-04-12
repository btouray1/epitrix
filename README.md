# 🧬 Epitrix — Mechanistic Immune Simulation Platform

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## What is Epitrix?

Epitrix is an open-source **mechanistic immune simulation platform** for vaccine formulation research. It models the full innate→adaptive immune cascade — from antigen epitope scanning and lipid nanoparticle (LNP) formulation design through to predicted clinical immune outcomes — with 95% confidence intervals derived from published biological variability data.

> ⚠️ **Important:** Epitrix is a *parameterised mechanistic simulation*, not a trained machine learning model. Every equation coefficient was manually derived from the peer-reviewed literature listed in the Evidence Base. Outputs are hypothesis-generating approximations intended for research and educational use only. They should not inform clinical decisions.

---

## Key Capabilities

| Module | Description |
|---|---|
| 🧬 **PSSM Epitope Scanning** | Scans protein sequences for MHC-I (HLA-A\*02:01, 9-mer) and MHC-II (HLA-DR, 15-mer) binders using published anchor position weight matrices. Automatically upgrades to IEDB NetMHCpan 4.1 / NetMHCIIpan when reachable. |
| 💉 **LNP Formulation** | 13 ionizable lipids, 14 helper lipids, 12 PEG-lipids with physicochemical metadata. Custom lipid input supported. |
| 🔥 **Innate Pathway Simulation** | Predicts TLR7/8, TLR3, cGAS-STING, Complement, and Inflammasome activation from formulation chemistry. |
| 🎯 **Adaptive Immune Cascade** | Predicts Th1/Th2/Th17/Tfh polarisation, antibody kinetics, and memory formation quality. |
| 🦠 **DC Programming** | Models cDC1, cDC2, and pDC subset activation kinetics and IL-12/IL-10 output. |
| 📊 **Clinical Reactogenicity** | Predicts safety score and reactogenicity with population stratification (adult, elderly, paediatric, immunocompromised). |
| ⚗️ **Formulation Optimizer** | Sweeps 720 formulation combinations and ranks by user-chosen objective (efficacy, safety, durability, Th1 bias, or balanced). |
| 💾 **Export & Compare** | Save multiple runs, compare side-by-side with CI error bars, download as CSV. |

---

## Screenshots

> Add screenshots of the platform here after deployment.

---

## Quick Start

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/epitrix.git
cd epitrix

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Epitrix
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

### IEDB Integration

Epitrix calls the [IEDB Analysis Resource](https://tools.iedb.org) for NetMHCpan predictions when available. To verify connectivity from your machine:

```bash
curl -s -L -X POST https://tools-cluster-interface.iedb.org/tools_api/mhci/ \
  -d "method=recommended&sequence_text=NITNLCPFGEVFNATR&allele=HLA-A*02:01&length=9" | head -3
```

If you see TSV output, IEDB is reachable and Epitrix will show a green **"IEDB ✓"** badge when scanning epitopes. If IEDB is unreachable (e.g. behind a firewall), Epitrix automatically falls back to the local PSSM scanner.

---

## Project Structure

```
epitrix/
├── app.py                      # Main application (single-file Streamlit app)
├── requirements.txt            # Python dependencies
├── .streamlit/
│   └── config.toml             # Streamlit theme and server config
└── README.md                   # This file
```

---

## Deploying to Streamlit Community Cloud

1. Fork or push this repo to your GitHub account (must be **public**)
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **"New app"** → select your repo → set main file to `app.py`
4. Click **Deploy**

Your app will be live at `https://YOUR_USERNAME-epitrix-app-streamlit.app` within ~5 minutes.

---

## Deploying to Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Select **Streamlit** as the SDK
3. Upload `app.py`, `requirements.txt`, and `.streamlit/config.toml`
4. The Space builds automatically — your app is live at `https://huggingface.co/spaces/YOUR_USERNAME/epitrix`

---

## Evidence Base & Parameterisation Sources

Epitrix does not train on data — it uses manually extracted coefficients from 22 peer-reviewed publications. Key sources include:

- **LNP formulation:** Hassett et al. (NPJ Vaccines 2019), Kulkarni et al. (Nano Letters 2021)
- **TLR activation:** Karikó et al. (Immunity 2005), Andries et al. (Nature Biotechnology 2015)
- **Innate–adaptive bridge:** Miao et al. (Nature Biotechnology 2019), Liang et al. (Nature Communications 2021)
- **Adaptive outcomes:** Walsh et al. (NEJM 2020), Goldberg et al. (Science 2021), Crotty et al. (Science 2021)
- **Epitope prediction:** Reynisson et al. (NAR 2020), Rammensee et al. (Immunogenetics 1999)
- **Biological variability:** Voysey et al. (Lancet 2021), Crooke et al. (npj Vaccines 2019)

Full citations with DOI links are available in the **Evidence Base** section of the app.

---

## Roadmap

The current version is a mechanistic simulation. Future development directions:

- [ ] Replace hand-coded coefficients with gradient-boosted models trained end-to-end on the cited datasets
- [ ] Expand IEDB allele coverage (HLA-A\*01:01, HLA-B\*07:02, multi-allele population coverage)
- [ ] Dose and route of administration as input parameters
- [ ] HLA haplotype population frequency weighting
- [ ] Booster dose and prime-boost schedule modeling
- [ ] Export to PDF report

---

## Limitations

- Epitrix models biological processes that are not fully understood. All equations are simplifications.
- Predictions assume a generic healthy adult unless population stratification is selected.
- The PSSM epitope scanner covers HLA-A\*02:01 (most common HLA-A allele, ~45% of Europeans) and a generic HLA-DR allele. Other alleles require IEDB connectivity.
- Formulation optimizer covers 720 combinations — a fraction of the real design space.
- No dose, route, or adjuvant variables are currently modelled.

---

## Citation

If you use Epitrix in your research, please cite:

```
Touray, B. (2025). Epitrix: A Mechanistic Immune Simulation Platform for
Vaccine Formulation Research. GitHub. https://github.com/YOUR_USERNAME/epitrix
```

---

## License

MIT License — free to use, modify, and distribute with attribution.

---

## Contact

Built by **Bubacarr Touray**  
Molecular Laboratory Supervisor, Exact Sciences  

For questions, bug reports, or collaboration: open a GitHub Issue.

---

> *Epitrix is a research tool. It is not a medical device and has not been validated for clinical use. Always consult primary literature and conduct experimental validation before making formulation or clinical decisions.*
