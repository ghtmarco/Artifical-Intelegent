import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import joblib
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Fungsi untuk menghasilkan data acak dengan deskripsi tugas
def generate_random_data(num_tasks=100):
    now = datetime.now()
    data = []
    descriptions = ["Urgent meeting", "Review document", "Prepare presentation", "Client call", "Team brainstorming"]
    
    for i in range(num_tasks):
        task_name = f"Task {i+1}"
        due_date = now + timedelta(days=np.random.randint(1, 30))
        estimated_duration = np.random.randint(1, 8)  # dalam jam
        priority = np.random.choice(['Low', 'Medium', 'High'])
        description = np.random.choice(descriptions)
        
        data.append([task_name, due_date, estimated_duration, priority, description])
    
    return data

# Membuat dataset
data = generate_random_data(500)  # Meningkatkan jumlah data
df = pd.DataFrame(data, columns=['Task', 'Due Date', 'Estimated Duration', 'Priority', 'Description'])

# Menyimpan dataset
df.to_csv('task_scheduling_dataset.csv', index=False)
print("Dataset berhasil disimpan sebagai 'task_scheduling_dataset.csv'")

# Persiapan fitur
le = LabelEncoder()
df['Priority_Encoded'] = le.fit_transform(df['Priority'])
df['Due Date'] = pd.to_datetime(df['Due Date'])
df['Days Until Due'] = (df['Due Date'] - pd.Timestamp.now()).dt.days

# NLP: Mengubah deskripsi tugas menjadi fitur
vectorizer = CountVectorizer()
description_features = vectorizer.fit_transform(df['Description'])

# Menggabungkan fitur numerik dan teks
X_numeric = df[['Estimated Duration', 'Days Until Due']].values
X_text = description_features.toarray()
X = np.hstack((X_numeric, X_text))

y = df['Priority_Encoded']

# Visualisasi distribusi prioritas
plt.figure(figsize=(10, 5))
df['Priority'].value_counts().plot(kind='bar')
plt.title('Distribusi Prioritas Tugas')
plt.xlabel('Prioritas')
plt.ylabel('Jumlah')
plt.savefig('prioritas_distribusi.png')
plt.close()

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Buat model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Proses pelatihan dengan epoch
n_epochs = 5
epoch_accuracy = []

for epoch in range(n_epochs):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    epoch_accuracy.append(accuracy)
    print(f"Epoch {epoch+1}/{n_epochs}, Akurasi: {accuracy:.4f}")

# Visualisasi akurasi per epoch
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_epochs+1), epoch_accuracy, marker='o')
plt.title('Akurasi Model per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.savefig('akurasi_per_epoch.png')
plt.close()

# Evaluasi model final
y_pred_final = model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred_final)
print(f"\nAkurasi model final: {final_accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_final)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()

# Simpan model dan vectorizer
joblib.dump(model, 'task_scheduling_model.joblib')
joblib.dump(vectorizer, 'description_vectorizer.joblib')
print("Model dan vectorizer berhasil disimpan.")

# Fungsi untuk memprediksi prioritas
def predict_priority(estimated_duration, days_until_due, description):
    description_vector = vectorizer.transform([description]).toarray()
    features = np.hstack(([[estimated_duration, days_until_due]], description_vector))
    prediction = model.predict(features)
    return le.inverse_transform(prediction)[0]

model = joblib.load('task_scheduling_model.joblib')
vectorizer = joblib.load('description_vectorizer.joblib')

# Contoh penggunaan
print("\nContoh prediksi:")
print(predict_priority(5, 10, "Urgent meeting with client"))