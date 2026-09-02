# Kidney Cancer Detection — Demo

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A free tool that looks at a kidney CT scan image and predicts whether it shows a **Cyst, a Tumor, a Stone, or a Normal kidney**. Built to be usable by anyone — no coding required to try it, and open-source for anyone who wants to look under the hood or build on it.

> ⚠️ **Not a medical device.** This is a research/education project, not a clinical tool. Do not use it for real diagnosis. See [DISCLAIMER.md](DISCLAIMER.md).

## Try it — free, in your browser, no install

**👉 [huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo](https://huggingface.co/spaces/avantiwhenever/kidney-cancer-detection-demo)**

Click the link, then either upload a kidney CT image or click one of the four example pictures. It runs right there in your browser — nothing is uploaded to any server, no account needed.

## What it actually does

It was shown thousands of labeled kidney CT scan images ahead of time and learned the visual patterns that tell apart a cyst, a tumor, a kidney stone, and a healthy kidney. When you give it a new image, it compares what it sees against what it learned and gives you its best guess, along with how confident it is in each possibility.

In testing on images it hadn't seen before, it got the right answer about **98 out of 100 times**. That's a good sign the approach works — but it was only tested on images from one source, so it hasn't been proven to work reliably on scans from other hospitals or machines. That, plus a lot more clinical testing, is what separates a demo like this from something a doctor could actually rely on.

## Documentation

| If you want to... | Read |
|---|---|
| Run the full app on your own machine (includes a heatmap explaining each prediction) | **[Run it locally](docs/RUN_LOCALLY.md)** |
| Train it on your own images/dataset, or update the public demo with a new model | **[Experiment with your own data](docs/EXPERIMENT.md)** |
| Understand the model, the data, the results, and how it's built | **[How it works](docs/HOW_IT_WORKS.md)** |
| See the clinical/regulatory context and what a real medical product would require | **[Background](docs/BACKGROUND.md)** |

## Dataset & license

- Code: **MIT** — see [LICENSE](LICENSE). Free to use, modify, and redistribute.
- Training data: [mhmad240/kidney-ct-classification](https://huggingface.co/datasets/mhmad240/kidney-ct-classification) on Hugging Face — check its page for license terms before redistribution or commercial use.

## Contributing

Issues and pull requests are welcome — better models, DICOM support, and external validation on other datasets are all good directions.
