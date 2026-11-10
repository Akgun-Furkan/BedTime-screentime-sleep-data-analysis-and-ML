# 🌙 Gece Ekran Kullanımı ve Uyku Borcu: Ertesi Gün Yorgunluk Skoru Tahmini

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)

Bu proje, bireylerin demografik özellikleri, gece akıllı telefon kullanım alışkanlıkları, kafein tüketimleri ve uyku mimarisi verilerini kullanarak **ertesi günkü yorgunluk skorunu (1-10)** regresyon algoritmalarıyla yüksek doğrulukla tahmin eden uçtan uca bir makine öğrenmesi boru hattıdır (pipeline).

---

## 📌 İçindekiler
- [Problem Tanımı ve Amac](#-problem-tanımı-ve-amaç)
- [Veri Seti Bilgileri](#-veri-seti-bilgileri)
- [Proje Mimarısı ve Adımları](#-proje-mimarisi-ve-adımları)
- [Keşifçi Veri Analizi (EDA) Bulguları](#-keşifçi-veri-analizi-eda-bulguları)
- [Veri Ön İşleme](#-veri-ön-işleme)
- [Model Seçimi ve Eğitimi](#-model-seçimi-ve-eğitimi)
- [Model Değerlendirme ve Sonuçlar](#-model-değerlendirme-ve-sonuçlar)
- [Hata Analizi ve Aşırı Öğrenme Teşhisi](#-hata-analizi-ve-aşırı-öğrenme-teşhisi)
- [Kurulum ve Kullanım](#-kurulum-ve-kullanım)

---

## 🎯 Problem Tanımı ve Amaç

* **Girdi (Features):** Yaş, cinsiyet, meslek, kronotip, gece telefon süresi (dk), kullanılan ana uygulama, ekran parlaklığı (%), mavi ışık filtresi durumu, 17:00 sonrası kafein alımı (mg), fiziksel aktivite (dk), uykuya dalma süresi, toplam uyku süresi, derin/REM uyku yüzdeleri ve sabah alarmı erteleme sayısı.
* **Hedef Çıktı (Target):** `next_day_fatigue_score` (Ertesi Gün Yorgunluk Skoru — 1.0 ile 10.0 arası sürekli sayısal değer).
* **Gerçek Dünya İhtiyacı:** Sağlık teknolojileri ve giyilebilir cihaz uygulamalarında (ör. Apple Health, Oura, Garmin), kullanıcıların dijital ekran alışkanlıklarını düzenleyerek kronik yorgunluğu ve iş gücü kayıplarını önlemeye yönelik kişiselleştirilmiş geri bildirim mekanizmaları geliştirmek.

---

## 📊 Veri Seti Bilgileri

* **Kaynak:** [Kaggle - Sleep Debt and Screen Time: Late Night Phone Habits Dataset](https://www.kaggle.com/datasets/samartalwar/sleep-debt-and-screen-time-late-night-phone-habits)
* **Örnek Sayısı:** 8,500 bireysel gözlem kaydı.
* **Özellik Sayısı:** 15 bağımsız özellik (4 kategorik, 11 sayısal) ve 1 hedef değişken.
* **Eksik Veri (Null):** Veri setinde hiçbir eksik veya hatalı gözlem bulunmamaktadır.

| Özellik Türü | Değişken Adları |
| --- | --- |
| **Sayısal (Numeric)** | `age`, `bedtime_phone_minutes`, `screen_brightness_pct`, `blue_light_filter_active`, `caffeine_post_5pm_mg`, `physical_activity_min`, `sleep_latency_min`, `total_sleep_hours`, `deep_sleep_pct`, `rem_sleep_pct`, `morning_alarm_snoozes` |
| **Kategorik (Categorical)** | `gender`, `occupation_type`, `chronotype`, `primary_bedtime_app` |
| **Hedef (Target)** | `next_day_fatigue_score` |

---

## 🛠 Proje Mimarisi ve Adımları

```
[ Ham CSV Verisi ]
       │
       ▼
[ Keşifçi Veri Analizi (EDA) ] ──► (Histogram, Scatter Plot, Heatmap)
       │
       ▼
[ ColumnTransformer & Pipeline ]
  ├── Sayısal: StandardScaler()
  └── Kategorik: OneHotEncoder()
       │
       ▼
[ %80 Train / %20 Test Bölünmesi ]
       │
       ├──────────────────────────────┐
       ▼                              ▼
[ Baseline Model ]           [ Karmaşık Model ]
Linear Regression          Random Forest Regressor
       │                              │
       └──────────────┬───────────────┘
                      ▼
        [ Metrik Değerlendirmesi ] ──► (MAE, RMSE, R²)
                      │
                      ▼
      [ Öğrenme Eğrisi & Hata Analizi ]
```

---

## 🔍 Keşifçi Veri Analizi (EDA) Bulguları

Veri seti üzerinde yapılan korelasyon ve görselleştirme analizlerinde öne çıkan temel bulgular:

1. **Toplam Uyku Süresi (`total_sleep_hours`) $\rightarrow$ Correlation: -0.88**  
   Yorgunluk skorunu azaltan en güçlü faktördür. Toplam uyku süresi arttıkça ertesi gün yorgunluk belirgin şekilde düşmektedir.
2. **Sabah Alarmı Erteleme Sayısı (`morning_alarm_snoozes`) $\rightarrow$ Correlation: +0.88**  
   Yorgunluk hissi ile çok güçlü pozitif doğrusal ilişkiye sahiptir.
3. **Gece Telefon Süresi ve Uykuya Dalma (`bedtime_phone_minutes` & `sleep_latency_min`) $\rightarrow$ Correlation: +0.86**  
   Yatmadan önce ekrana bakılan süre uzadıkça bireylerin uykuya dalma süresi (sleep latency) uzamakta, bu da doğrudan yorgunluk skorunu (`+0.71`) artırmaktadır.

---

## ⚙️ Veri Ön İşleme

1. **Özellik Ölçekleme (Scaling):** Tüm sayısal nitelikler `StandardScaler` kullanılarak ortalaması 0, standart sapması 1 olacak şekilde dönüştürülmüştür.
2. **Kategorik Encoding:** Metinsel/kategorik değişkenler `OneHotEncoder(handle_unknown='ignore')` ile sayısal vektörlere dönüştürülmüştür.
3. **Veri Bölme:** Veri seti %80 Eğitim ($N = 6,800$) ve %20 Test ($N = 1,700$) kümesi olarak `random_state=42` ile ayrılmıştır. Bilgi sızmasını (data leakage) engellemek amacıyla tüm dönüşümler `Pipeline` içinde eğitilmiştir.

---

## 🤖 Model Seçimi ve Eğitimi

Problemin doğasına uygun olarak iki farklı regresyon yaklaşımı eğitilip karşılaştırılmıştır:

* **Linear Regression (Baseline):** Özellikler arasındaki doğrusal ilişkileri değerlendirmek üzere referans model olarak seçilmiştir.
* **Random Forest Regressor:** Gece ekran kullanımı, kafein ve fizyolojik uyku evreleri arasındaki karmaşık ve doğrusal olmayan (non-linear) etkileşimleri yakalamak amacıyla 100 karar ağacından oluşan topluluk modeli tercih edilmiştir (`n_estimators=100`, `random_state=42`).

---

## 📈 Model Değerlendirme ve Sonuçlar

Test verisi ($N = 1,700$) üzerinde elde edilen performans metrikleri aşağıda özetlenmiştir:

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | R² Skoru (Açıklanan Varyans) |
| --- | --- | --- | --- |
| **Linear Regression** | 0.6357 | 0.8006 | 0.9140 (%91.40) |
| **Random Forest Regressor** | **0.4261** | **0.6011** | **0.9515 (%95.15)** |

* **Sonuç Yorumu:** Random Forest modeli, 1-10 puanlık yorgunluk skalasında ortalama sadece **0.42 puanlık** mutlak hata (MAE) ile tahmin yapmaktadır. Model, veri setindeki toplam varyansın **%95.15'ini** başarıyla açıklamaktadır.

---

## 🧪 Hata Analizi ve Aşırı Öğrenme Teşhisi

### 1. Ezberleme (Overfitting) Kontrolü
5 katlı çapraz doğrulama (5-Fold Cross Validation) ve Öğrenme Eğrileri (Learning Curves) incelendiğinde:
* **Linear Regression:** Train RMSE: `0.788`, Validation RMSE: `0.791` $\rightarrow$ **Underfitting (Eksik Öğrenme)** eğilimi göstermektedir.
* **Random Forest:** Train RMSE: `0.230`, Validation RMSE: `0.601` $\rightarrow$ Veri sayısı arttıkça doğrulama hatası düşmeye devam etmekte ve aşırı öğrenme (overfitting) yaşanmamaktadır.

### 2. Zayıf Yönler
Model, yorgunluk skoru $9.5 - 10.0$ arasında olan aşırı (uç) değerlerde tahmin yaparken hafif derecede merkeze doğru bükülme (regress to the mean) sergilemektedir.

---

## 🚀 Kurulum ve Kullanım

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/Akgun-Furkan/BedTime-screentime-sleep-data-analysis-and-ML.git
cd bedtime-screentime-sleep-debt-ML
```

### 2. Gerekli Kütüphaneleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Modeli Çalıştırın
```bash
python main.py
```

---

## 📜 Lisans
Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.