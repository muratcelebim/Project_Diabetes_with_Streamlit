################################################
# End-to-End Diabetes Machine Learning Pipeline III
################################################

##############################################################
# Kütüphanelerin Import Edilmesi
##############################################################
import yfinance as yf
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import joblib
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)



st.write(""" 
# **Welcome**""")

st.write(""" 
## **Artificial Intelligence Prediction Automation!**""")

st.write("""
Değerli kullanıcılarım uygulamaya hoşgeldiniz. :sunglasses:

Sizlere kısaca uygulamadan bahsedeceğim. Bu veri seti ABD'nin Pima Indian bölgesine ait 21 yaşından büyük kadınlardan 
alınan bilgilerle oluşturulmuştır.

Alınan bilgiler ise kısaca şunlar:
    **Hamilelik Sayısı, Glikoz Değeri, Kan Basıncı, Cilt Kalınlığı, Insulin Değeri, Vücut Kitle Indexi, 
    Soydaki Kişilerin Diyabet Olma İhtimalini Hesaplayan Fonksiyon, Yaş**'dır. Bu bilgiler doğrultusunda bu kişinin 
    diyabet hastası olup olmadığını tamin eden bir model geliştirdim. Uygulama açık değilse sol üstteki ">" 
    işareti ile gösterilen barı açarak tahmin için değerleri seçebilirsiniz. Ardından özeelikleri değiştirerek 
    sonucunuzun "Prediction" başlığı altındaki "Click for Predict" butonuna basarak görebilirsiniz.
""")


st.write("""Uygulamanın tahmin başarsı (**Accuracy Score**): %76""")








st.sidebar.header('User Input Parameters for Diabetes')
df1 = pd.read_csv("diabetes.csv")
def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies - Hamilelik Sayısı', 0.00, 17.00, 0.00)
    glucose = st.sidebar.slider('Glucose - Glikoz Değeri', 80.00, 199.00, 80.00)
    bloodpressure = st.sidebar.slider('BloodPressure - Kan Basıncı', -1.00, 123.00, 0.00)
    skinthickness = st.sidebar.slider('SkinThickness - Cilt Kalınlığı', 0.00, 99.00, 0.00)
    insulin = st.sidebar.slider('Insulin - Insulin Değeri', 0.00, 846.00, 800.00)
    bmi = st.sidebar.slider('BMI (Body Mas Indexi) - Vücut Kitle Indexi', 18.00, 67.10, 18.00)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction - Soydaki Kişilerin Diyabet Olma İhtimalini Hesaplayan Fonksiyon', 0.078, 2.42, 0.00)
    age = st.sidebar.slider('Age - Yaş', 21.00, 81.00, 21.00)

    data = {'Pregnancies': pregnancies,
            'Glucose': glucose,
            'BloodPressure': bloodpressure,
            'SkinThickness': skinthickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": dpf,
            "Age": age
            }
    features = pd.DataFrame(data, index = [df1.index[-1] + 1])
    return features

df2 =  user_input_features()

st.write("**Values**")
df2



dff = pd.concat([df1, df2]).reset_index()
dff = dff.drop("index", axis = 1)
#dff

from diabetes_pipeline import diabetes_data_prep
X, y = diabetes_data_prep(dff)
X_sample = X[-1:]
new_model = joblib.load("voting_clf.pkl")
new_model.predict(X_sample)

st.subheader('Prediction')

"""
if new_model.predict(X_sample)[0] == 0:
    print("Do Not Have Diabetes")
    st.write("Do Not Have Diabetes")
else:
    print("Have Diabetes")
    st.write("Have Diabetes")
"""

if st.button('Click for Predict') and new_model.predict(X_sample)[0] == 0:
    if new_model.predict(X_sample)[0] == 0:
        st.write("Do Not Have Diabetes")
    else:
        st.write("Have Diabetes")
else:
    st.write('For the prediction, please click the button that says **Click for Predict**.')




st.write("""
### Bana ulaşabilirsiniz:""")

st.markdown("""**[Linkedin](https://www.linkedin.com/in/muratcelebi3455)**""")
st.markdown("""**[Medium](https://medium.com/@celebim.murat)**""")
st.markdown("""**[Github](https://github.com/muratcelebim)**""")
st.markdown("""**[Kaggle](https://www.kaggle.com/clbmurat)**""")





































