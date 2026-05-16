# Kuvantunnistus omilla kuvilla

Tämä projekti on oppimisprojekti oman datan kuvantunnistuksesta. Toteutuksessa on rakennettu kolme erilaista mallia:

1. Oma CNN-malli
2. Esikoulutetun VGG16-mallin feature-extraction-malli
3. Hienosäädetty VGG16-malli

## Rakenne

- `src/`
  - `data_pipeline.py` : datan jakaminen ja ImageDataGenerator-pohjainen lataus
  - `models.py` : mallien rakentaminen ja konfigurointi
  - `train.py` : mallien koulutus ja datan jakaminen
  - `evaluate.py` : testidatan luokitusraportti ja confusion matrix
  - `inference.py` : salvettuun malliin perustuva ennustus
- `notebooks/`
  - `01_data_preparation.ipynb`
  - `02_own_cnn.ipynb`
  - `03_feature_extraction_vgg16.ipynb`
  - `04_fine_tuned_vgg16.ipynb`
- `dataset/`
  - `raw/` : alkuperäiset luokat omilla kuvilla
  - `split/` : jaettu train/val/test -data
- `report/` : raportti ja kuvat
- `requirements.txt` : riippuvuudet

## Dataset

Luo alkuperäinen data seuraavasti:

```
dataset/raw/Tiku/*.jpg
dataset/raw/Ksusha/*.jpg
```

Raw datasetissa on 214 kuvaa, joista 59 on `Ksusha`-luokassa ja 155 on `Tiku`-luokassa.
Kuvat on jaettu 70/15/15-suhteella train/val/test -dataan.

Tämän jälkeen aja koulutuskomento, joka jakaa datan automaattisesti `dataset/split`-rakenteeseen.

## Asennus

```bash
pip install -r requirements.txt
```

## Mallin koulutus

Peruskoulutus tehdään seuraavasti:

```bash
python src/train.py --mode cnn --raw_dir dataset/raw --data_dir dataset/split --epochs 20 --batch_size 16
```

Esikoulutetun VGG16:n feature extraction -mallin koulutus:

```bash
python src/train.py --mode feature --raw_dir dataset/raw --data_dir dataset/split --epochs 10 --batch_size 16
```

Hienosäädetyn VGG16:n koulutus ajetaan näin:

```bash
python src/train.py --mode finetune --raw_dir dataset/raw --data_dir dataset/split --epochs 10 --batch_size 8 --unfrozen_layers 4
```

## Testaus ja arviointi

Kun malli on koulutettu, testaa se seuraavasti:

```bash
python src/evaluate.py --model src/vgg_finetune_best.h5 --output_prefix report/vgg_finetune_eval --image_size 224 --batch_size 16
```

Voit tehdä saman myös muilla malleilla, esimerkiksi `src/aug_run_best.h5` tai `src/baseline_run_best.h5`.

## Inference

Käytä inferenssiskriptiä yksittäiselle kuvalle tai kansion kaikille kuville:

```bash
python src/inference.py --model src/vgg_finetune_best.h5 --input dataset/split/test/Tiku/IMG_4769.JPG --image_size 224
```

```bash
python src/inference.py --model src/vgg_finetune_best.h5 --input dataset/split/test --out_csv results.csv --image_size 224
```

## Notebookit

Projektissa on näitä Jupyter-notebookeja ja niiden HTML-versiot:

- `notebooks/01_data_preparation.ipynb`
- `notebooks/01_data_preparation.html`
- `notebooks/02_own_cnn.ipynb`
- `notebooks/02_own_cnn.html`
- `notebooks/03_feature_extraction_vgg16.ipynb`
- `notebooks/03_feature_extraction_vgg16.html`
- `notebooks/04_fine_tuned_vgg16.ipynb`
- `notebooks/04_fine_tuned_vgg16.html`

## Huomioita

- `src/train.py` luo `dataset/split`-kansion, jos data on määritelty `dataset/raw`-kansioon.
- `src/evaluate.py` tallentaa raportin `report/`-kansioon ja luo confusion matrix -kuvan.
- `src/inference.py` käyttää `dataset/split/train`-kansion luokkien nimitietoja.
