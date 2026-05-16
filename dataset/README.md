Sijoita omat kuvatiedostot tähän hakemistoon.

Rakenne:

dataset/raw/Tiku/*.jpg
dataset/raw/Ksusha/*.jpg

Kun `src/train.py` suoritetaan, data jaetaan automaattisesti seuraaviin kansioihin:

dataset/split/train/<luokka>/*
dataset/split/val/<luokka>/*
dataset/split/test/<luokka>/*

Kuvien esikäsittely: kaikki kuvat muunnetaan 224×224 kokoisiksi, RGB-muotoon ja normalisoidaan välille 0–1.
