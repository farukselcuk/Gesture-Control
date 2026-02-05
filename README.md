# gesture-control — El Hareketi Kontrol

Bu proje, bilgisayar kamerası üzerinden MediaPipe kullanarak el hareketleriyle komut çalıştıran küçük bir prototiptir.

Özellikler
- Mod 1: temel hareket bildirimleri (kalp, wolf/flag)
- Mod 2: 2 saniye sustain kuralı ile özel hareketler
  - Sol el: baş parmak + serçe açık, diğerleri kapalı → "Manifest Arıyorum"
  - Sağ el: baş parmak + serçe açık, diğerleri kapalı → Edis şarkısı
  - Sağ el: sadece işaret parmağı yukarı → Arda Güler (bu hareket korunmuştur)
- Mod 3: parmak şıklatma (snap) ile komut tetikleme
- Sağ el/sol el hareketleriyle ekran görüntüsü alma (swipe)

Fotoğraf gösterme yönlendirmeleri kaldırıldı; projede aile fotoğrafı veya gösterim dosyaları bulunmamaktadır.

Bağımlılıklar
- Python 3.8+
- opencv-python
- mediapipe
- keyboard

Kurulum
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Çalıştırma
```powershell
python main.py
```

Notlar
- `keyboard` modülünün tuş yakalama davranışı bazı sistemlerde (admin erişimi gibi) kısıtlanmış olabilir. Alternatif olarak `cv2.waitKey` üzerinden tuş değiştirme uygulanabilir.
- Eğer hareket algılama hassasiyeti düşükse, kameranıza göre eşik değerlerini (`HEART_THRESHOLD`, `SNAP_DELTA_PX`) `main.py` içinde ayarlayın.
