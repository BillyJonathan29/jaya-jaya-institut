"""
Jaya Jaya Institut - Student Dropout & Academic Performance Binary Predictor CLI
Script inferensi mandiri untuk mendeteksi risiko mahasiswa: Dropout (1) vs Graduate (0).
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_student_model.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "target_encoder.json")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.json")

class StudentDropoutPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Pastikan model sudah dilatih.")
        
        self.model = joblib.load(MODEL_PATH)
        with open(ENCODER_PATH, 'r') as f:
            self.target_encoder = json.load(f)
        with open(FEATURES_PATH, 'r') as f:
            self.feature_names = json.load(f)

    def predict(self, df_input: pd.DataFrame) -> pd.DataFrame:
        df = df_input.copy()
        
        # Drop Status or Target column if present in input
        for col in ['Status', 'Target']:
            if col in df.columns:
                df = df.drop(columns=[col])

        # Ensure all features exist
        for feat in self.feature_names:
            if feat not in df.columns:
                df[feat] = 0

        df_ordered = df[self.feature_names]
        
        preds = self.model.predict(df_ordered)
        probs = self.model.predict_proba(df_ordered)
        
        inv_map = {int(k): v for k, v in self.target_encoder['inverse_mapping'].items()}
        
        result_df = df_input.copy()
        result_df['Predicted_Status'] = [inv_map.get(p, str(p)) for p in preds]
        result_df['Graduate_Probability (%)'] = np.round(probs[:, 0] * 100, 2)
        result_df['Dropout_Probability (%)'] = np.round(probs[:, 1] * 100, 2)
        
        # Categorize Dropout Risk Level
        risk_levels = []
        for p in probs[:, 1]:
            if p >= 0.60:
                risk_levels.append('TINGGI (High Dropout Risk)')
            elif p >= 0.35:
                risk_levels.append('SEDANG (Moderate Risk)')
            else:
                risk_levels.append('RENDAH (Safe / Low Risk)')
        result_df['Dropout_Risk_Level'] = risk_levels
        
        return result_df

def main():
    parser = argparse.ArgumentParser(description="Jaya Jaya Institut - Student Dropout Binary Predictor CLI")
    parser.add_argument("--file", type=str, help="Path ke file CSV data mahasiswa")
    parser.add_argument("--sample", action="store_true", help="Jalankan prediksi pada sampel mahasiswa aktif (Enrolled)")
    parser.add_argument("--output", type=str, default="student_predictions.csv", help="Nama file output CSV")

    args = parser.parse_args()

    print("==================================================================")
    print("   JAYA JAYA INSTITUT - STUDENT DROPOUT BINARY PREDICTOR CLI")
    print("==================================================================")

    predictor = StudentDropoutPredictor()

    if args.sample or not args.file:
        enrolled_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unlabeled_enrolled_students.csv")
        data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.csv")
        
        target_file = enrolled_path if os.path.exists(enrolled_path) else data_path
        if not os.path.exists(target_file):
            print(f"File data tidak ditemukan.")
            sys.exit(1)
            
        raw_df = pd.read_csv(target_file, sep=';' if ';' in open(target_file).readline() else ',')
        sample_df = raw_df.head(10).copy()
        print(f"\n[INFO] Mengambil 10 sampel data mahasiswa aktif dari {os.path.basename(target_file)}...")

        predictions = predictor.predict(sample_df)
        
        display_cols = ['Course', 'Tuition_fees_up_to_date', 'Scholarship_holder', 'Age_at_enrollment',
                        'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
                        'Predicted_Status', 'Dropout_Probability (%)', 'Dropout_Risk_Level']
        display_cols = [c for c in display_cols if c in predictions.columns]
        
        print("\n--- HASIL PREDIKSI RISIKO DROPOUT MAHASISWA AKTIF ---")
        print(predictions[display_cols].to_string(index=False))
        
        predictions.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Hasil prediksi lengkap disimpan ke: {args.output}")

    elif args.file:
        if not os.path.exists(args.file):
            print(f"Error: File '{args.file}' tidak ditemukan.")
            sys.exit(1)

        print(f"\n[INFO] Membaca data mahasiswa dari: {args.file}")
        input_df = pd.read_csv(args.file, sep=';' if ';' in open(args.file).readline() else ',')
        predictions = predictor.predict(input_df)
        
        display_cols = ['Course', 'Tuition_fees_up_to_date', 'Scholarship_holder', 'Age_at_enrollment',
                        'Curricular_units_1st_sem_approved', 'Curricular_units_2nd_sem_approved',
                        'Predicted_Status', 'Dropout_Probability (%)', 'Dropout_Risk_Level']
        display_cols = [c for c in display_cols if c in predictions.columns]
        
        print("\n--- RINGKASAN HASIL PREDIKSI (Top 10) ---")
        print(predictions[display_cols].head(10).to_string(index=False))
        
        predictions.to_csv(args.output, index=False)
        print(f"\n[SUCCESS] Seluruh hasil prediksi disimpan ke: {args.output}")

if __name__ == '__main__':
    main()
