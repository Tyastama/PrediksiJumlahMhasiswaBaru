import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

def fit(self, X, y, epsilon = 1e-10):
    self.y_classes, y_counts = np.unique(y, return_counts=True)
    self.x_classes = np.array([np.unique(x) for x in X.T])
    self.phi_y = 1.0 * y_counts/y_counts.sum()
    self.u = np.array([X[y==k].mean(axis=0) for k in self.y_classes])
    self.var_x = np.array([X[y==k].var(axis=0)  + epsilon for k in self.y_classes])
    return self
    
def predict(self, X):
    return np.apply_along_axis(lambda x: self.compute_probs(x), 1, X)
    
def compute_probs(self, x):
    probs = np.array([self.compute_prob(x, y) for y in range(len(self.y_classes))])
    return self.y_classes[np.argmax(probs)]
    
def compute_prob(self, x, y):
    c = 1.0 /np.sqrt(2.0 * np.pi * (self.var_x[y]))
    return np.prod(c * np.exp(-1.0 * np.square(x - self.u[y]) / (2.0 * self.var_x[y])))
    
def evaluate(self, X, y):
    return (self.predict(X) == y).mean()

# input data
df = pd.read_csv('dt_train.csv', sep=';')
# dropping duplicate values 
df.drop_duplicates(keep=False,inplace=True) 
# mengecek apakah ada data yang berisi null
df.isnull().values.any()
# mengecek jumlah baris data yang berisi null
len(df[pd.isnull(df).any(axis=1)])
df.dropna(inplace=True)  # untuk menghapus baris jika semua adalanya nan
# Variabel independen
x = df.drop(["STAT_REG"], axis = 1)
# Variabel dependen
y = df["STAT_REG"]
# Import train_test_split function
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 42)
# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# Mengaktifkan/memanggil/membuat fungsi klasifikasi Naive bayes
modelnb = GaussianNB()
# Memasukkan data training pada fungsi klasifikasi naive bayes
nbtrain = modelnb.fit(x_train, y_train)
print(nbtrain.class_count_)
# Menentukan hasil prediksi dari x_test
y_pred = nbtrain.predict(x_test)
#print('-----hasil prediksi-----')
print(y_pred)
# Menentukan probabilitas hasil prediksi
probabilitas = modelnb.predict_proba(x_test)
print(probabilitas)
xz = pd.DataFrame(probabilitas)
print(xz)
#import confusion_matrix model
from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test, y_pred)
#Menghitung nilai akurasi dari klasifikasi naive bayes 
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
modelnb.score(x_test,y_test)
print(modelnb.score(x_test,y_test))

# Save the training model to file
joblib.dump(modelnb , 'nbc_model.pkl')
