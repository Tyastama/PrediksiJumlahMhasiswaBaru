from flask import Flask,render_template,url_for,request
import pandas as pd
import numpy as np
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/input_csv')
def csv():
	return render_template('input_csv.html')


@app.route('/input_data')
def input():
	return render_template('input_data.html')

@app.route('/hasil')
def hasil():
  return render_template('hasil.html')

@app.route('/olah', methods=['POST'])
def olah():
  #Load trained model
  NB_prdt_model = open('nbc_model.pkl','rb')
  modenbc = joblib.load(NB_prdt_model)
  #input data csv test
  data = request.files['data_file']
  df = pd.read_csv(data,  delimiter=";", encoding='utf-8')
  predicted = modenbc.predict(df)
  data1 = predicted
  df2 = pd.DataFrame(data1, columns=['Prediksi'])
  #marge dataframe
  df3 = pd.merge(df, df2, left_index=True, right_index=True)
  data =pd.DataFrame(df3)
  reg = []
  #registrasi
  TI = data[(data.JURUSAN == 541) & (data.Prediksi == 1)]
  jum_TI = len(TI)
  reg.append(jum_TI)
  SI = data[(data.JURUSAN == 561) & (data.Prediksi == 1)]
  jum_SI = len(SI)
  reg.append(jum_SI)
  TK = data[(data.JURUSAN == 331) & (data.Prediksi == 1)]
  jum_TK = len(TK)
  reg.append(jum_TK)
  MI = data[(data.JURUSAN == 311) & (data.Prediksi == 1)]
  jum_MI = len(MI)
  reg.append(jum_MI)
  KA = data[(data.JURUSAN == 321) & (data.Prediksi == 1)]
  jum_KA = len(KA)
  reg.append(jum_KA)

  tdk_reg =[]
  #tdk_registrasi
  t_TI = data[(data.JURUSAN == 541) & (data.Prediksi == 0)]
  jum_TI = len(t_TI)
  tdk_reg.append(jum_TI)
  t_SI = data[(data.JURUSAN == 561) & (data.Prediksi == 0)]
  jum_SI = len(t_SI)
  tdk_reg.append(jum_SI)
  t_TK = data[(data.JURUSAN == 331) & (data.Prediksi == 0)]
  jum_TK = len(t_TK)
  tdk_reg.append(jum_TK)
  t_MI = data[(data.JURUSAN == 311) & (data.Prediksi == 0)]
  jum_MI = len(t_MI)
  tdk_reg.append(jum_MI)
  t_KA = data[(data.JURUSAN == 321) & (data.Prediksi == 0)]
  jum_KA = len(t_KA)
  tdk_reg.append(jum_KA)
  ar = np.array(reg)
  arx = np.array(tdk_reg)
  ar2 = ar.reshape((5, 1))
  arx2 = arx.reshape((5, 1))
  ar3 = ar2.astype(int)
  arx3 = arx2.astype(int)

  a1 =pd.DataFrame(ar3, columns=['Registrasi'], index=['TI', 'SI','TK','MI','KA'])
  a2 =pd.DataFrame(arx3, columns=['Tidak Registrasi'], index=['TI', 'SI','TK','MI','KA'])
  d2 = pd.merge(a1, a2, left_index=True, right_index=True)
 
  return render_template('hasil.html', data= d2.to_html())  

@app.route('/predict', methods=['POST'])
def predict():
#Load trained model
  NB_prdt_model = open('nbc_model.pkl','rb')
  modenbc = joblib.load(NB_prdt_model)
  # Input Data 
  if request.method == 'POST':
      data1 = request.form.get('data1')
      data2 = request.form.get('data2')
      data3 = request.form['data3'] 
      data4 = request.form.get('data4') 
      data5 = request.form.get('data5') 
      data = [data1, data2, data3, data4, data5]
      arr = np.array(data)
      arr2 = arr.reshape((1, 5))
      arr3 = arr2.astype(float)
      my_prediction = modenbc.predict(arr3)
  return render_template('result.html', prediction = my_prediction)  

if __name__ == '__main__':
	app.run(debug=True)