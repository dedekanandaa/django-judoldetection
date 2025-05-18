# App Skripsi

Repositori ini berisi aplikasi skripsi. Ikuti langkah berikut untuk menjalankan aplikasi ini.

## Prasyarat

- Python
- Django 5
- Numpy
- Pillow (PIL)
- Gensim Word2Vec
- Torch, Torchvision
- Googlesearch
- Sastrawi
- Playwright dengan browser dependencies terinstall
- asgiref
- [Pytesseract](https://github.com/tesseract-ocr/tesseract/releases/) yang sudah diinstall dan dikonfigurasi beserta [Indonesia trained data](https://github.com/tesseract-ocr/tessdata/blob/main/ind.traineddata)
- [Pretrained Model](https://drive.google.com/drive/folders/1DIz0MHo4qycmkjISkRbafOTuD4RoabW6?usp=sharing) (dapat diakses menggunakan email **@pnb.ac.id**) 


## Instalasi

1. Clone repositori:
    ```bash
    git clone https://github.com/dedekanandaa/django-judoldetection.git
    ```
2. Install database:
    ```bash
    python manage.py migrate
    ```
3. Load Model:
    
    Download model di [Prasyarat](#prasyarat) dan simpan semua model pada ```media/assets/```.


## Menjalankan Aplikasi

```bash
python manage.py runserver
```

Aplikasi akan berjalan di `http://localhost:8000` (atau port yang tertera di terminal).