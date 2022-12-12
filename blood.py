import pickle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification



#judul web
st.title('Data Mining')

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Import Data", "Preprocessing", "Modelling", "Implementation"])

with tab1:
    st.write("Dataset yang digunakan pada penelitian ini yakni Thyroid Disease Datasets")
    st.write("Thyroid Disease")
    st.write("The most common thyroid disorder is hypothyroidism. Hypo- means deficient or under(active), so hypothyroidism is a condition in which the thyroid gland is underperforming or producing too little thyroid hormone. Recognizing the symptoms of hypothyroidism is extremely important.")
    st.write("Data Set Information : From Garavan Institute, Documentation: as given by Ross Quinlan, 6 databases from the Garavan Institute in Sydney, Australia.")
    st.write("Approximately the following for each database : 2800 training (data) instances and 972 test instances, Plenty of missing data, 29 or so attributes, either Boolean or continuously-valued")
    st.write("https://www.kaggle.com/datasets/whenamancodes/blood-transfusion-dataset")
    
with tab2:
    st.write("Load Data :")
    data_file = st.file_uploader("Upload CSV",type=['csv'])
    if data_file is not None:
        file_details = {"Filename":data_file.name,"FileType":data_file.type,"FileSize":data_file.size}
        st.write(file_details)
        data = pd.read_csv(data_file)
        data = data.dropna()
        st.dataframe(data)
        
with tab3:
    st.write("Normalisasi Data")
    data.head()
    
    from sklearn.preprocessing import LabelEncoder
    enc=LabelEncoder()
    for x in data.columns:
      data[x]=enc.fit_transform(data[x])
    data.info()
    
    data.head()
    st.dataframe(data)
       
    y=data['donasidarah']
    x=data.drop(['donasidarah'],axis=1)
    st.write("Menampilkan data yang sudah dinormalisasi dan dilakukan scaled features")
    st.dataframe(data)
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.3,stratify=y)
    st.write("X_train.shape")
    st.write(X_train.shape)
    st.write("X_test.shape")
    st.write(X_test.shape)
    st.write("y_train.shape")
    st.write(y_train.shape)
    st.write("y_test.shape")
    st.write(y_test.shape)

with tab4:
    st.write("## Naive Bayes")
    # Feature Scaling to bring the variable in a single scale
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    GaussianNB(priors=None)
    
    # Fitting Naive Bayes Classification to the Training set with linear kernel
    nvklasifikasi = GaussianNB()
    nvklasifikasi = nvklasifikasi.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = nvklasifikasi.predict(X_test)
    
    #lets see the actual and predicted value side by side
    y_compare = np.vstack((y_test,y_pred)).T
    #actual value on the left side and predicted value on the right hand side
    
    # Menentukan probabilitas hasil prediksi
    nvklasifikasi.predict_proba(X_test)
    akurasi = round(100 * accuracy_score(y_test, y_pred))
    st.write('Model accuracy score: {0:0.2f}'. format(akurasi))
    
    st.write("## KNN")
    K=10
    knn=KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    
    skor_akurasi = round(100 * accuracy_score(y_test,y_pred))
    st.write("Model accuracy score : {0:0.2f}" . format(skor_akurasi))
    
    st.write("## Decision Tree")
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    # prediction
    dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)
    
    #Accuracy
    akurasi = round(100 * accuracy_score(y_test,y_pred))
    st.write('Model Accuracy Score: {0:0.2f}'.format(akurasi))
    
with tab5:
    thyroid_model = pickle.load(open('bloodmodel.sav', 'rb'))
    bulanterkini = st.text_input('Masukkan Bulan Pada Saat Ini : ')
    frekuensi = st.text_input('Frekuensi')
    CCdarah = st.text_input('Darah/c.c')
    totalbulan = st.text_input('Total Bulan')

    diagnosis = ''

    if st.button('Prediksi'):
        prediction = thyroid_model.predict([[bulanterkini, frekuensi, CCdarah, totalbulan]])

        if (prediction == 0):
            diagnosis = 'Pasien Belum Mendonorkan Darah'
        else :
            diagnosis = 'Pasien Sudah Mendonorkan Darah'
    st.success(diagnosis)
