# Raportti

## Datasetin kuvaus

Tässä projektissa luokat ovat kaksi omaa kissaani: `Tiku` ja `Ksusha`.
Kuvat on kerätty omista kotikuvista eri kulmista, valaistusolosuhteissa ja asennoissa käyttäen puhelinta ja kameraa.
Kuvien esikäsittelyssä kaikki kuvat muutettiin 224×224 kokoisiksi, RGB:ksi ja normalisoitiin välillä 0–1.

Raw dataset sisältää yhteensä 214 kuvaa:

- Kyseessä on 59 Ksusha-kuvaa ja 155 Tiku-kuvaa
- Tämä täyttää vähintään 50 kuvaa per luokka -vaatimuksen

Datan jako tehtiin 70/15/15-suhteella:

- Train: 149 kuvaa (108 Tiku, 41 Ksusha)
- Validation: 31 kuvaa (23 Tiku, 8 Ksusha)
- Test: 34 kuvaa (24 Tiku, 10 Ksusha)

Jaettu hakemistorakenne on:

```
dataset/split/train/<luokka>/*
dataset/split/val/<luokka>/*
dataset/split/test/<luokka>/*
```

## Mallien suorituskyky

Projektissa testattiin tehtävänannon mukaisesti kolme mallia:

- Malli 1: Oma CNN-malli
- Malli 2: Esikoulutetun VGG16-mallin feature extraction
- Malli 3: Hienosäädetty VGG16-malli

### Lyhyt tulosyhteenveto

- Baseline CNN (`baseline_run_best.h5`): test-accuracy 0.71. Malli ei kyennyt erottamaan `Ksusha`-kuvia luotettavasti ja ennakoi usein `Tiku`-luokan.
- Augmentaatiolla koulutettu CNN (`aug_run_best.h5`): test-accuracy 0.88. Tämä malli paransi yleistä luokittelua ja vähensi ylisovittamisen riskiä.
- Hienosäädetty VGG16 (`vgg_finetune_best.h5`): test-accuracy 0.91. Paras tulos saavutettiin hienosäädöllä, joka vapautti viimeiset VGG16-kerrokset oppimaan oman datan piirteisiin.

### Tulosten vertailu

Mallien ero näkyy selvästi:

- Baseline-malli oppi peruskuvion mutta kärsi luokkien epätasapainosta.
- Data-augmentaatio paransi `Ksusha`-tunnistusta ja teki mallista vähemmän herkkää valaistukselle.
- VGG16:n hienosäätö toi parhaan tasapainon oikean ja väärän luokan tunnistukseen.

### Plottien upotus

Seuraavat kuvat on tallennettu `report/`-kansioon ja ne havainnollistavat mallien oppimiskäyriä:

![Augmented CNN accuracy](aug_run_accuracy.png)

![Augmented CNN loss](aug_run_loss.png)

![VGG16 finetune accuracy](vgg_finetune_accuracy.png)

![VGG16 finetune loss](vgg_finetune_loss.png)

## Luotettavuusarvio testidatalle

Testidatan arviointi paljasti myös mallien käyttökelpoisuuden reaalimaailmassa.

### Baseline CNN

- Accuracy: 0.71
- Ksusha: precision 0.00, recall 0.00
- Tiku: precision 0.71, recall 1.00

### Augmented CNN

- Accuracy: 0.88
- Ksusha: precision 0.88, recall 0.70
- Tiku: precision 0.88, recall 0.96

### Fine-tuned VGG16

- Accuracy: 0.91
- Ksusha: precision 0.89, recall 0.80
- Tiku: precision 0.92, recall 0.96

Paras malli on `vgg_finetune_best.h5`, koska sen luokkaperusteet ovat tasapainoisimmat ja se säilyttää korkean tarkkuuden myös testidatassa.

### Confusion matrix

Alla on VGG16-hienosäädön confusion matrix testidatalle.

![VGG16 confusion matrix](vgg_finetune_eval_confusion_matrix.png)

Testidatan väärin luokitellut kuvat olivat useimmiten `Ksusha`-luokkaa, joissa valaistus tai poseeraus poikkesi selkeästi harjoitusdatasta.
Tämä vahvistaa, että datan monipuolinen keruu ja augmentaatio auttoivat parantamaan mallin yleistymistä.

Lisäksi tallensin `report/baseline_eval_confusion_matrix.png` ja `report/aug_run_eval_confusion_matrix.png` eri mallien vertailua varten.

## Haasteet ja ratkaisut

- Datan määrä oli rajallinen, erityisesti `Ksusha`-luokassa. Ratkaisin tämän käyttämällä augmentaatiota ja luokkakohtaisia painoja.
- Datan epätasapaino vaikutti baseline-CNN:n oppimiseen, joten testasin myös esikoulutettua VGG16-mallia ja hienosäätöä.
- Mallin ylisovittamisen estämiseksi käytin validation-settiä ja palautin parhaat painot `EarlyStopping`-periaatteella.

## Pohdinta

- Baseline-CNN toimi nopeasti, mutta se ei riittänyt luokkien erotteluun ilman lisädataa ja augmentaatiota.
- VGG16:n feature extraction antaa hyvän lähtökohdan, mutta paras lopputulos saadaan, kun viimeisiä kerroksia hienosäädetään omalle datalle.
- Seuraavaksi voisi kokeilla lisää augmentaatiovariaatioita, pienempää oppimisnopeutta sekä erilaisia transfer learning -arkkitehtuureja kuten ResNet tai EfficientNet.

## Johtopäätös

Paras malli tässä tehtävässä on `vgg_finetune_best.h5`. Se saavutti parhaan testitarkkuuden ja osoittaa, että esikoulutetun verkon hienosäätö omassa datassa on tehokas tapa parantaa luokittelua.
