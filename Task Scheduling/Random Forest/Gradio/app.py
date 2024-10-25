import pandas as pd
import numpy as np
import gradio as gr
import joblib
from datetime import datetime

# Load model, scaler, dan feature names
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')['feature_names']

def predict_task_priority(task_name, duration, deadline_str):
    try:
        # Parse deadline string to calculate days
        start_date = datetime.now()
        try:
            deadline = datetime.strptime(deadline_str, '%Y-%m-%d')
        except:
            return "Error: Format tanggal harus YYYY-MM-DD (contoh: 2024-12-31)"
            
        deadline_days = (deadline - start_date).days
        
        if deadline_days < 0:
            return "Error: Deadline tidak boleh di masa lalu"
        
        # Buat DataFrame dengan feature names yang sesuai
        input_data = pd.DataFrame({
            'duration_hours': [duration],
            'deadline_days': [deadline_days]
        })
        
        # Transform menggunakan scaler
        input_scaled = scaler.transform(input_data)
        
        # Predict
        priority = model.predict(input_scaled)[0]
        
        priority_map = {
            1: "Rendah",
            2: "Sedang",
            3: "Tinggi"
        }
        
        # Generate response
        response = f"Analisis Tugas: {task_name}\n"
        response += f"Durasi: {duration} jam\n"
        response += f"Deadline: {deadline_days} hari lagi\n"
        response += f"Prioritas: {priority_map[priority]}\n\n"
        
        # Add recommendations
        if priority == 3:
            response += "Rekomendasi: Kerjakan segera! Deadline dekat dan membutuhkan waktu lama."
        elif priority == 2:
            response += "Rekomendasi: Buatlah jadwal yang tepat dan mulai kerjakan secara bertahap."
        else:
            response += "Rekomendasi: Dapat dikerjakan dengan lebih santai, tapi tetap pantau progress."
            
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_task_priority,
    inputs=[
        gr.Dropdown(
            choices=[
                "Meeting", 
                "Bekerja", 
                "Belajar", 
                "Tugas Kuliah", 
                "Proyek"
            ],
            label="Nama Tugas"
        ),
        gr.Slider(
            minimum=1,
            maximum=10,
            value=5,
            step=0.5,
            label="Durasi Tugas (dalam jam)"
        ),
        gr.Textbox(
            label="Deadline (YYYY-MM-DD)",
            placeholder="Contoh: 2024-12-31",
            info="Masukkan tanggal dalam format YYYY-MM-DD"
        )
    ],
    outputs=gr.Textbox(label="Hasil Analisis", lines=6),
    title="Sistem Prioritas Tugas",
    description="""
    Sistem ini akan membantu Anda menentukan prioritas tugas berdasarkan:
    1. Durasi pengerjaan tugas
    2. Jarak waktu ke deadline
    
    Hasil analisis akan memberikan rekomendasi pengelolaan waktu yang sesuai.
    """
)

if __name__ == "__main__":
    iface.launch()