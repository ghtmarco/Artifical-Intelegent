import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Fungsi untuk menghasilkan data acak
def generate_random_data(num_tasks=100):
    now = datetime.now()
    data = []
    
    for i in range(num_tasks):
        task_name = f"Task {i+1}"
        due_date = now + timedelta(days=np.random.randint(1, 30))
        estimated_duration = np.random.randint(1, 8)  # dalam jam
        priority = np.random.choice(['Low', 'Medium', 'High'])
        
        data.append([task_name, due_date, estimated_duration, priority])
    
    return data

# Membuat dataset
data = generate_random_data()
df = pd.DataFrame(data, columns=['Task', 'Due Date', 'Estimated Duration', 'Priority'])

# Menyimpan dataset
df.to_csv('task_scheduling_dataset.csv', index=False)

print(df.head())
print(f"\nDataset berhasil dibuat dengan {len(df)} tugas.")