import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_dataset(n_samples=1000):
    np.random.seed(42)
    data = []
    
    task_names = [
        "Meeting",
        "Tugas Kuliah",
        "Belajar",
        "Proyek",
        "Presentasi"
    ]
    
    start_date = datetime(2024, 10, 20)
    
    for _ in range(n_samples):
        # Generate random data
        task_name = np.random.choice(task_names)
        duration = round(np.random.uniform(1, 10), 1)
        deadline_days = np.random.randint(1, 31)
        
        # Calculate priority
        if deadline_days <= 7 and duration >= 7:
            priority = 3  # High
        elif deadline_days >= 20 and duration <= 3:
            priority = 1  # Low
        else:
            priority = 2  # Medium
            
        data.append({
            'task_name': task_name,
            'duration_hours': duration,
            'deadline_days': deadline_days,
            'priority_level': priority
        })
    
    df = pd.DataFrame(data)
    df.to_csv('task_scheduling_dataset.csv', index=False)
    
if __name__ == "__main__":
    generate_dataset()