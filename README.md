# AeroFoil

Repositori ini digunakan sebagai pengumpulan data latih sekaligus pelatihan model
dengan *Convolutional Neural Network* (CNN).

## Sebelum Mulai
Berikut langkah-langkah yang diperlukan agar selama pengembangan dapat berjalan dengan baik, yaitu:

- Pasang Poetry sebagai Python Package Management di halaman [ini](https://python-poetry.org/docs/#installing-manually).
- Gunakan instruksi `poetry install` untuk memasang package yang diperlukan.
- Pindah ke direktori `aerofoil_datasets` di *Terminal* atau *Console* yang berbeda, lalu jalankan instruksi `tfds build`.
- Untuk menghilangkan teks atau pesan yang muncul saat menjalankan CLI di *Terminal* atau *Console*, atur 
*Environment Variable* di OS sebagai berikut `TF_CPP_MIN_LOG_LEVEL=3`.