import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle as pk
from sklearn.preprocessing import LabelEncoder
import numpy as np
st.header('Авто машины бодит үнийг тогтоох нь Ml')

model = pk.load(open('modelX.pkl', 'rb'))
cars_data = pd.read_csv('C:/Users/hp/Documents/Санхүү шинжилгээ/car/fdata.csv')

label_encoder = LabelEncoder()

Үйлдвэрлэгч = st.selectbox('Үйлдвэрлэгч', cars_data['Үйлдвэрлэгч'].unique())
Загвар = st.selectbox('Загвар', cars_data['Загвар'].unique())
Мотор = st.selectbox('Мотор', cars_data['Мотор'].unique())
Хурдны_хайрцаг = st.selectbox('Хурдны хайрцаг', cars_data['Хурдны_хайрцаг'].unique())
Хүрд = st.selectbox('Хүрд', cars_data['Хүрд'].unique())
Төрөл = st.selectbox('Төрөл', cars_data['Төрөл'].unique())
Өнгө = st.selectbox('Өнгө', cars_data['Өнгө'].unique())
Үйлдвэрлэсэн_он = st.slider('Үйлдвэрлэсэн он', 1982, 2023)
Орж_ирсэн_он = st.slider('Орж ирсэн он', 1984, 2023)
Хөдөлгүүр = st.selectbox('Хөдөлгүүр', cars_data['Хөдөлгүүр'].unique())
Салоны_өнгө = st.selectbox('Салоны_өнгө', cars_data['Салоны_өнгө'].unique())
Лизинг = st.selectbox('Лизинг', cars_data['Лизинг'].unique())
Хөтлөгч = st.selectbox('Хөтлөгч', cars_data['Хөтлөгч'].unique())
Явсан_км = st.text_input('Явсан_км', '0')  # '0' гэж хадгалагдсан тоо
Нөхцөл = st.selectbox('Нөхцөл', cars_data['Нөхцөл'].unique())
Хаалга = st.selectbox('Хаалга', cars_data['Хаалга'].unique())

user_input_dict = {
    'Үйлдвэрлэгч': Үйлдвэрлэгч,
    'Загвар': Загвар,
    'Мотор': Мотор,
    'Хурдны_хайрцаг': Хурдны_хайрцаг,
    'Хүрд': Хүрд,
    'Төрөл': Төрөл,
    'Өнгө': Өнгө,
    'Үйлдвэрлэсэн_он': Үйлдвэрлэсэн_он,
    'Орж_ирсэн_он': Орж_ирсэн_он,
    'Хөдөлгүүр': Хөдөлгүүр,
    'Салоны_өнгө': Салоны_өнгө,
    'Лизинг': Лизинг,
    'Хөтлөгч': Хөтлөгч,
    'Явсан_км': float(Явсан_км),
    'Нөхцөл': Нөхцөл,
    'Хаалга': Хаалга
}

user_input_df = pd.DataFrame(user_input_dict, index=[0])

categorical_columns = ['Үйлдвэрлэгч', 'Загвар', 'Мотор', 'Хурдны_хайрцаг', 'Хүрд', 'Төрөл', 'Өнгө', 
                       'Хөдөлгүүр', 'Салоны_өнгө', 'Лизинг', 'Хөтлөгч', 'Нөхцөл', 'Хаалга']

for col in categorical_columns:
    user_input_df[col] = label_encoder.fit_transform(user_input_df[col])

user_input_np = user_input_df.to_numpy()

y_pred = model.predict(user_input_np)

st.write(f"Үнийн таамаглал: {y_pred[0]}")
