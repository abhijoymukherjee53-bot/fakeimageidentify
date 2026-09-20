# CIFAKE Analysis Lab

A small CNN + Flask web app that classifies images as **REAL** (photograph) or
**FAKE** (AI-generated), trained on the [CIFAKE dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
(60,000 real CIFAR-10 photos vs. 60,000 Stable Diffusion–generated equivalents, 32×32px).

```
cifake-classifier/
├── train.py              # trains the CNN, saves model/model.h5
├── app.py                # Flask server: renders the UI + /predict API
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
├── data/                 # you add this — see step 1 below
│   ├── train/{REAL,FAKE}/
│   └── test/{REAL,FAKE}/
└── model/                # created by train.py
    ├── model.h5
    ├── class_names.txt
    └── training_history.png
```

## 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Get the dataset

Download the **CIFAKE** dataset from Kaggle:
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

Unzip it so you end up with:

```
data/
  train/
    REAL/   (~50,000 images)
    FAKE/   (~50,000 images)
  test/
    REAL/   (~10,000 images)
    FAKE/   (~10,000 images)
```

(If you'd rather test the pipeline on a smaller set first, just put a few
hundred images in each folder — training will still run, just less accurately.)

## 3. Train the model

```bash
python train.py
```

This trains a CNN (a few conv blocks + batch norm + dropout, ~15 epochs with
early stopping) and writes:
- `model/model.h5` — the trained weights
- `model/class_names.txt` — class order used for decoding predictions
- `model/training_history.png` — accuracy/loss curves

Training on the full dataset takes a while on CPU (expect a fair bit longer
than on a GPU) — that's expected for 100k 32×32 images. Feel free to lower
`EPOCHS` in `train.py` for a quicker first run.

## 4. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5000** — drag an image onto the specimen panel and
click **Analyze image**. The report panel shows the verdict (AUTHENTIC /
SYNTHETIC), a confidence gauge, and the raw probability.

## Notes

- The model was trained on 32×32 CIFAR-style images, so any image you upload
  is resized down to 32×32 before prediction — it works best on similar
  subject matter (everyday objects/scenes) rather than, say, high-detail
  portraits.
- `app.py` loads the model lazily on the first `/predict` call, so the server
  starts fine even before you've trained anything — you'll just get a clear
  error until `model/model.h5` exists.
- Swap in your own architecture in `build_model()` (train.py) if you want to
  experiment — transfer learning, deeper nets, etc.
