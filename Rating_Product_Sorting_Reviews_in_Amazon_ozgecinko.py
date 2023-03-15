###################################################
# Özge Çinko
###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df = pd.read_csv("datasets/amazon_review.csv")
df.head(10)
df.info()

df["overall"].mean()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################

# Bugünün tarihi adında verisetine uyan bir tarih belirlemeye gerek yok, verisetinde bulunuyor.
# Son 30 günde yapılan puan ortalamasına bakalım.
df.loc[df["day_diff"] <= 30, "overall"].mean()

# Güncel ağırlıklı puan ortalamasına baktım.
def time_based_weighted_average(dataframe, w1=30, w2=28, w3=22, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100


time_based_weighted_average(df)

# Puanlara göre azalan şekilde sıraladım.
df.sort_values("overall", ascending=False).head(20)

# Alakasız sonuçların gelmesi gibi bir durumla karşılaştım.
# Değerlendirmenin faydalı bulunma sayısı ve değerlendirmeye verilen oy sayısına göre sıralamaya baktım.
df.sort_values("helpful_yes", ascending=False).head(20)
df.sort_values("total_vote", ascending=False).head(20)

# Sosyal ispatı biraz daha iyileştirdi ama tek başına bu iki metrik yeterli değildi.
# Üç metriği birleştirmeye karar verdim.
# overall 0-5 arası sayılardan oluşur, helpful_yes 0-1 arası değerlerden ve total_vote daha büyük sayılardan oluşur.
# üçünü direkt çarparsak overall ezilecektir, bu nedenle değerleri ölçeklendirmemiz gerekir diye düşündüm hepsini aynı ölçeğe getirdim.
# overall 1-5 arasındaki sayılardan oluşur, diğer değişkenleri 1,5 arasında ölçeklendirdim.
df["helpful_yes_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["helpful_yes"]]). \
    transform(df[["helpful_yes"]])

df.describe().T

df["total_vote_scaled"] = MinMaxScaler(feature_range=(1, 5)). \
    fit(df[["total_vote"]]). \
    transform(df[["total_vote"]])


# Ağırlıklarına göre skorlaştırdım.
def weighted_sorting_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["helpful_yes_scaled"] * w1 / 100 +
            dataframe["total_vote_scaled"] * w2 / 100 +
            dataframe["overall"] * w3 / 100)


df["weighted_sorting_score"] = weighted_sorting_score(df)


df.sort_values("weighted_sorting_score", ascending=False).head(20)


###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################

###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

# Öncelikle fonksiyonları yazdım.

def score_pos_neg_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)


def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)


##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)
# Bu veriyi yorumlayınca çok eski tarihten olan ve daha kötü yorumların getirildiğini gördüm.

df.sort_values("score_average_rating", ascending=False).head(20)
# Bu veriyi yorumlayınca çok eski tarihten olan ve daha iyi yorumların getirildiğini gördüm.

df.sort_values("score_pos_neg_diff", ascending=False).head(20)
# Bu veriyi yorumlayınca çok eski tarihten olan ve iyi ve kötü yorumların getirildiğini gördüm.