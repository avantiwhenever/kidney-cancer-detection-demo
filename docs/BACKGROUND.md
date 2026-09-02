# Background

Deeper context behind this project: what it would take to turn this demo into a real clinical product, and the broader gaps in AI-assisted cancer detection today. None of this is required reading to use the repo — see the main [README](../README.md) for that.

## From demo to a real app

A working Gradio demo is maybe 5% of what a real product needs. Roughly, in order:

**1. Data & modeling rigor**
- Move past a single 2D-slice classifier: real kidney tumor assessment is volumetric (3D CT), and a full study is a stack of slices, not one image. Consider 3D CNNs or per-slice aggregation.
- Validate on data the model hasn't seen from *different hospitals/scanners* — a model that's great on one dataset's distribution often fails on another's (scanner, contrast protocol, patient population differences). The dataset used here is single-source, so external validation is essential before claiming anything.
- Establish a clinically meaningful metric bar (e.g. sensitivity target for the Tumor class) with input from a radiologist, not just "accuracy."

**2. Product engineering**
- Backend API serving the model (FastAPI/TorchServe), containerized, with versioned model artifacts.
- Real frontend (the Gradio app is a prototype, not a product UI) — likely needs DICOM support, not just JPEGs, since that's the real clinical image format.
- Data pipeline for ingesting real studies (DICOM ingestion, de-identification, storage).
- Monitoring: prediction logging, drift detection, human-in-the-loop review/override.
- Auth, audit logging, and access control if any real patient data touches the system.

**3. Regulatory & clinical**
- If this is ever used to inform an actual diagnosis, it's a medical device under FDA (or equivalent) regulation — this triggers a whole separate track: intended-use definition, clinical validation studies, quality management system, FDA 510(k)/De Novo or CE marking, post-market surveillance.
- HIPAA/data-privacy compliance for any real patient data (BAAs, encryption, access controls).
- Malpractice/liability considerations if the tool influences clinical decisions.

**4. Deployment reality**
- Hospitals integrate via PACS/EHR (e.g. HL7/FHIR, DICOM), not a standalone web app — real adoption means integrating into radiologist workflow, not asking them to use a separate tool.
- Ongoing clinical validation and monitoring post-deployment, since real-world data drifts from training data over time.

Realistically: the demo in this repo is a weekend-to-few-weeks project. A real clinical product is a multi-year, multi-discipline effort (ML, regulatory, clinical, security) — which lines up with the research-to-deployment lag described below.

## Industry gaps in AI cancer detection (as of 2026)

- **Research-to-deployment lag.** Models that hit human-level accuracy in a paper often take 3–5 additional years to clear regulatory approval, clinical integration, and physician training before real-world use — the "we published SOTA" moment is early, not late, in the process.
- **Generalizability across sites.** Most models are trained/validated on data from one or a few institutions/scanners. Performance frequently drops on out-of-distribution data from other hospitals, populations, or equipment — a core reason many promising research models never reach the clinic.
- **Data scarcity per subtype.** Conventional approaches need tens of thousands of labeled images *per cancer type/subtype* to train well; rarer subtypes and less common cancers are underserved because that scale of labeled data doesn't exist for them. Foundation/pathology models aim to fix this but still need heavy fine-tuning per tumor type in practice.
- **Evaluation ≠ clinical utility.** Papers optimize for accuracy/AUC on curated test sets; hospitals need evidence of actual clinical utility (does using this tool improve patient outcomes or workflow?), which is a different, much harder, and rarer kind of study.
- **Interpretability and trust.** Clinicians are reasonably reluctant to act on a black-box output; explainability tooling (like the Grad-CAM in this repo) is necessary but far from sufficient — sold as a nice-to-have, but functionally required for adoption.
- **Regulatory and liability uncertainty.** Especially for models that update/learn over time, current regulatory frameworks weren't built for continuously-improving software, creating friction for anyone trying to ship responsibly.
- **Access inequality.** The places with the least access to specialist pathology/radiology (low-resource and rural settings) are exactly where good AI tools could matter most, but deployment/integration efforts skew toward well-resourced health systems first.

## Roadmap ideas (not yet built)

- Swap in a volumetric model over full CT studies (e.g. KiTS23-style data — [MedOtter/kits23](https://huggingface.co/datasets/MedOtter/kits23)) for tumor segmentation, not just slice classification.
- Add a second modality: a biomedical NLP model over radiology reports, to pair image findings with report text.
- Add external validation against a second, independent kidney CT dataset.
- Host the full Gradio app (with Grad-CAM) publicly — free Hugging Face `cpu-basic` Spaces now require a PRO subscription, so the live demo today is a browser-only ONNX version without the heatmap (see `hf_space_static/`). A cheap always-on VM, or PRO, would restore the full app as a public link.

---
See also: [Run it locally](RUN_LOCALLY.md) · [Experiment with your own data](EXPERIMENT.md) · [How it works](HOW_IT_WORKS.md) · [Back to README](../README.md)
