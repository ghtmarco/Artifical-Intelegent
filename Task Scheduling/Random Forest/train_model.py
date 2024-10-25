import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def create_priority_labels(row):
    
    if row['deadline_days'] <= 3:  # deadline dekat
        if row['duration_hours'] >= 6:  # durasi panjang
            return 1  # High priority
        else:
            return 2  # Medium priority
    elif row['deadline_days'] <= 7:  # deadline menengah
        if row['duration_hours'] >= 6:
            return 2  # Medium priority
        else:
            return 3  # Low priority
    else:  # deadline jauh
        return 3  # Low priority

def train_and_predict():
    try:
        # Load dataset
        df = pd.read_csv('Dataset/task_scheduling_dataset.csv')
        
        # Buat label prioritas baru berdasarkan logika yang benar
        df['corrected_priority'] = df.apply(create_priority_labels, axis=1)
        
        # Prepare features
        X = df[['duration_hours', 'deadline_days']]
        y = df['corrected_priority']  # Gunakan prioritas yang sudah dikoreksi
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(model, 'Models/random_forest_model.pkl')
        joblib.dump(scaler, 'Models/scaler.pkl')
        
        # Predict on full dataset
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        
        # Add predictions to dataframe
        df['predicted_priority'] = predictions
        
        print("\n" + "="*60)
        print("TASK SCHEDULING AI - PRIORITY PREDICTION")
        print("="*60)
        print("\nAnalyzed at:", datetime.now().strftime('%Y-%m-%d %H:%M'))
        print("\nTASK PRIORITY CLASSIFICATION:")
        print("-"*60)
        
        priority_labels = {
            1: "High",
            2: "Medium",
            3: "Low"
        }
        
        for priority in sorted(priority_labels.keys()):
            print(f"\n{priority_labels[priority]} Priority Tasks:")
            tasks = df[df['predicted_priority'] == priority].head(1)
            
            if len(tasks) == 0:
                print("No tasks in this category")
            else:
                for _, task in tasks.iterrows():
                    print(f"""
Task: {task['task_name']}
Duration: {task['duration_hours']} hours
Deadline: {task['deadline_days']} days
                    """)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
    print("="*60)

if __name__ == "__main__":
    train_and_predict()