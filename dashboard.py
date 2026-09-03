import os
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import joblib

# Set Page Config
st.set_page_config(
    page_title="Jaya Jaya Institut - Student Dropout Dashboard & Early Warning System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 15px;
        color: #64748b;
        margin-bottom: 20px;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        padding: 18px;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        margin-top: 5px;
    }
    .kpi-label {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("data.csv", sep=';' if ';' in open("data.csv").readline() else ',')
    df['Tuition_Status'] = df['Tuition_fees_up_to_date'].map({1: 'Lancar (Up to date)', 0: 'Menunggak (Late)'})
    df['Scholarship_Status'] = df['Scholarship_holder'].map({1: 'Penerima Beasiswa', 0: 'Bukan Beasiswa'})
    df['Debtor_Status'] = df['Debtor'].map({1: 'Memiliki Utang', 0: 'Tidak Ada Utang'})
    df['Gender_Label'] = df['Gender'].map({1: 'Laki-laki', 0: 'Perempuan'})
    return df

@st.cache_resource
def load_model():
    model_path = os.path.join("model", "random_forest_student_model.joblib")
    scaler_path = os.path.join("model", "scaler.joblib")
    encoder_path = os.path.join("model", "target_encoder.json")
    features_path = os.path.join("model", "feature_names.json")
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        with open(encoder_path, 'r') as f:
            target_enc = json.load(f)
        with open(features_path, 'r') as f:
            features = json.load(f)
        return model, scaler, target_enc, features
    return None, None, None, None

df = load_data()
model, scaler, target_enc, feature_names = load_model()

# Header
st.markdown('<div class="main-header">🎓 JAYA JAYA INSTITUT - STUDENT RETENTION & EARLY WARNING DASHBOARD</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Sistem Monitoring Performa Akademik & Deteksi Dini Risiko Dropout Mahasiswa (Binary ML Solution)</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 Filter Interaktif Data Mahasiswa")
selected_courses = st.sidebar.multiselect("Pilih Program Studi (Course)", options=sorted(df['Course'].unique()), default=sorted(df['Course'].unique()))
selected_scholar = st.sidebar.multiselect("Status Beasiswa", options=df['Scholarship_Status'].unique(), default=df['Scholarship_Status'].unique())
selected_tuition = st.sidebar.multiselect("Kelancaran SPP (Tuition)", options=df['Tuition_Status'].unique(), default=df['Tuition_Status'].unique())
selected_gender = st.sidebar.multiselect("Jenis Kelamin", options=df['Gender_Label'].unique(), default=df['Gender_Label'].unique())

filtered_df = df[
    (df['Course'].isin(selected_courses)) &
    (df['Scholarship_Status'].isin(selected_scholar)) &
    (df['Tuition_Status'].isin(selected_tuition)) &
    (df['Gender_Label'].isin(selected_gender))
]

if len(filtered_df) == 0:
    st.warning("⚠️ Tidak ada data mahasiswa yang sesuai dengan kombinasi filter yang dipilih.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Business Dashboard & Analytics", "🤖 AI Early Warning Dropout Predictor", "💡 Strategic Action Items & Kesimpulan"])

with tab1:
    # KPI Overview
    total_stud = len(filtered_df)
    dropout_cnt = (filtered_df['Status'] == 'Dropout').sum()
    grad_cnt = (filtered_df['Status'] == 'Graduate').sum()
    enroll_cnt = (filtered_df['Status'] == 'Enrolled').sum()
    dropout_rate = (dropout_cnt / total_stud) * 100 if total_stud > 0 else 0
    grad_rate = (grad_cnt / total_stud) * 100 if total_stud > 0 else 0
    avg_age = filtered_df['Age_at_enrollment'].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">👥 Total Mahasiswa</div><div class="kpi-value">{total_stud:,}</div></div>', unsafe_allow_html=True)
    with col2:
        badge_color = "#ef4444" if dropout_rate > 20 else "#10b981"
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">⚠️ Dropout Rate</div><div class="kpi-value" style="color:{badge_color}">{dropout_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">🎓 Kelulusan (Graduates)</div><div class="kpi-value" style="color:#10b981">{grad_rate:.1f}%</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">📚 Mahasiswa Aktif</div><div class="kpi-value">{enroll_cnt:,}</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">🎂 Rata-rata Usia Masuk</div><div class="kpi-value">{avg_age:.1f} Thn</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Row 1: Tuition & Scholarship
    r1_col1, r1_col2 = st.columns(2)
    with r1_col1:
        tf_stat = filtered_df.groupby(['Tuition_Status', 'Status']).size().reset_index(name='Count')
        fig_tf = px.bar(
            tf_stat, x='Tuition_Status', y='Count', color='Status',
            barmode='stack', title="<b>1. Pengaruh Kelancaran Pembayaran SPP terhadap Status Mahasiswa</b>",
            color_discrete_map={'Dropout': '#ef4444', 'Enrolled': '#f59e0b', 'Graduate': '#10b981'},
            text_auto=True
        )
        fig_tf.update_layout(xaxis_title="Status Pembayaran SPP", yaxis_title="Jumlah Mahasiswa")
        st.plotly_chart(fig_tf, use_container_width=True)

    with r1_col2:
        sch_stat = filtered_df.groupby(['Scholarship_Status', 'Status']).size().reset_index(name='Count')
        fig_sch = px.bar(
            sch_stat, x='Scholarship_Status', y='Count', color='Status',
            barmode='stack', title="<b>2. Pengaruh Beasiswa terhadap Tingkat Kelulusan vs Dropout</b>",
            color_discrete_map={'Dropout': '#ef4444', 'Enrolled': '#f59e0b', 'Graduate': '#10b981'},
            text_auto=True
        )
        fig_sch.update_layout(xaxis_title="Penerimaan Beasiswa", yaxis_title="Jumlah Mahasiswa")
        st.plotly_chart(fig_sch, use_container_width=True)

    # Row 2: Course Breakdown & Approved Units
    r2_col1, r2_col2 = st.columns(2)
    with r2_col1:
        course_stat = filtered_df.groupby(['Course', 'Status']).size().reset_index(name='Count')
        fig_course = px.bar(
            course_stat, x='Course', y='Count', color='Status',
            barmode='stack', title="<b>3. Distribusi Status Mahasiswa per Program Studi (Course)</b>",
            color_discrete_map={'Dropout': '#ef4444', 'Enrolled': '#f59e0b', 'Graduate': '#10b981'},
            text_auto=True
        )
        fig_course.update_layout(xaxis_title="Kode Program Studi", yaxis_title="Jumlah Mahasiswa")
        st.plotly_chart(fig_course, use_container_width=True)

    with r2_col2:
        fig_box = px.box(
            filtered_df, x='Status', y='Curricular_units_2nd_sem_approved', color='Status',
            title="<b>4. Performa Mata Kuliah Lulus Semester 2 vs Status Mahasiswa</b>",
            color_discrete_map={'Dropout': '#ef4444', 'Enrolled': '#f59e0b', 'Graduate': '#10b981'}
        )
        fig_box.update_layout(xaxis_title="Status Mahasiswa", yaxis_title="Jumlah MK Lulus (Sem 2)")
        st.plotly_chart(fig_box, use_container_width=True)

    # Row 3: ML Feature Importance & Debtor
    r3_col1, r3_col2 = st.columns(2)
    with r3_col1:
        if model is not None:
            feat_imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10).reset_index()
            feat_imp.columns = ['Feature', 'Importance']
            fig_imp = px.bar(
                feat_imp, y='Feature', x='Importance', orientation='h',
                title="<b>5. Top 10 Faktor Kunci Penentu Dropout (Binary ML Importance)</b>",
                color='Importance', color_continuous_scale='Teal',
                text_auto='.3f'
            )
            fig_imp.update_layout(xaxis_title="Importance Score", yaxis_title="Fitur / Variabel")
            st.plotly_chart(fig_imp, use_container_width=True)

    with r3_col2:
        debt_stat = filtered_df.groupby(['Debtor_Status', 'Status']).size().reset_index(name='Count')
        fig_debt = px.bar(
            debt_stat, x='Debtor_Status', y='Count', color='Status',
            barmode='stack', title="<b>6. Status Utang Mahasiswa (Debtor) vs Dropout</b>",
            color_discrete_map={'Dropout': '#ef4444', 'Enrolled': '#f59e0b', 'Graduate': '#10b981'},
            text_auto=True
        )
        fig_debt.update_layout(xaxis_title="Status Debitur Utang", yaxis_title="Jumlah Mahasiswa")
        st.plotly_chart(fig_debt, use_container_width=True)

with tab2:
    st.subheader("🎯 Prototype: AI Early Warning Dropout Predictor")
    st.write("Sistem inferensi biner untuk mendeteksi sedini mungkin apakah seorang mahasiswa berpotensi **Dropout (1)** atau **Graduate (0)** berdasarkan performa akademik dan status finansial:")

    if model is None:
        st.error("Model Machine Learning belum ditemukan di folder model/.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            inp_course = st.selectbox("Program Studi (Course)", sorted(df['Course'].unique()))
            inp_age = st.slider("Usia Saat Masuk (Age at Enrollment)", 17, 60, 20)
            inp_tuition = st.selectbox("Kelancaran Pembayaran SPP", [("Lancar / Up to Date (1)", 1), ("Menunggak (0)", 0)], format_func=lambda x: x[0])[1]
            inp_scholar = st.selectbox("Status Beasiswa", [("Penerima Beasiswa (1)", 1), ("Bukan Beasiswa (0)", 0)], format_func=lambda x: x[0])[1]
            inp_debtor = st.selectbox("Status Debitur Utang", [("Tidak Ada Utang (0)", 0), ("Memiliki Utang (1)", 1)], format_func=lambda x: x[0])[1]
            
        with c2:
            inp_sem1_enrolled = st.slider("Jumlah MK Diambil Sem 1", 0, 15, 6)
            inp_sem1_approved = st.slider("Jumlah MK Lulus Sem 1", 0, 15, 5)
            inp_sem1_grade = st.number_input("Rata-rata Nilai Sem 1 (0-20)", 0.0, 20.0, 13.0, step=0.5)
            inp_sem1_eval = st.slider("Jumlah Evaluasi/Ujian Sem 1", 0, 20, 6)
            inp_displaced = st.selectbox("Mahasiswa Perantau (Displaced)", [("Ya (1)", 1), ("Tidak (0)", 0)], format_func=lambda x: x[0])[1]
            
        with c3:
            inp_sem2_enrolled = st.slider("Jumlah MK Diambil Sem 2", 0, 15, 6)
            inp_sem2_approved = st.slider("Jumlah MK Lulus Sem 2", 0, 15, 5)
            inp_sem2_grade = st.number_input("Rata-rata Nilai Sem 2 (0-20)", 0.0, 20.0, 13.5, step=0.5)
            inp_sem2_eval = st.slider("Jumlah Evaluasi/Ujian Sem 2", 0, 20, 6)
            inp_gender = st.selectbox("Jenis Kelamin", [("Laki-laki (1)", 1), ("Perempuan (0)", 0)], format_func=lambda x: x[0])[1]

        if st.button("🚀 Jalankan Prediksi Risiko Dropout", type="primary"):
            sample_data = {
                'Marital_status': 1, 'Application_mode': 1, 'Application_order': 1, 'Course': inp_course,
                'Daytime_evening_attendance': 1, 'Previous_qualification': 1, 'Previous_qualification_grade': 130.0,
                'Nacionality': 1, 'Mothers_qualification': 1, 'Fathers_qualification': 1,
                'Mothers_occupation': 5, 'Fathers_occupation': 5, 'Admission_grade': 125.0,
                'Displaced': inp_displaced, 'Educational_special_needs': 0, 'Debtor': inp_debtor,
                'Tuition_fees_up_to_date': inp_tuition, 'Gender': inp_gender, 'Scholarship_holder': inp_scholar,
                'Age_at_enrollment': inp_age, 'International': 0,
                'Curricular_units_1st_sem_credited': 0, 'Curricular_units_1st_sem_enrolled': inp_sem1_enrolled,
                'Curricular_units_1st_sem_evaluations': inp_sem1_eval, 'Curricular_units_1st_sem_approved': inp_sem1_approved,
                'Curricular_units_1st_sem_grade': inp_sem1_grade, 'Curricular_units_1st_sem_without_evaluations': 0,
                'Curricular_units_2nd_sem_credited': 0, 'Curricular_units_2nd_sem_enrolled': inp_sem2_enrolled,
                'Curricular_units_2nd_sem_evaluations': inp_sem2_eval, 'Curricular_units_2nd_sem_approved': inp_sem2_approved,
                'Curricular_units_2nd_sem_grade': inp_sem2_grade, 'Curricular_units_2nd_sem_without_evaluations': 0,
                'Unemployment_rate': 11.1, 'Inflation_rate': 1.4, 'GDP': 0.79
            }
            
            df_in = pd.DataFrame([sample_data])[feature_names]
            p_dropout = model.predict_proba(df_in)[0, 1] * 100
            p_graduate = 100.0 - p_dropout
            pred_class = model.predict(df_in)[0]
            
            st.markdown("### 📊 Hasil Prediksi Status Mahasiswa:")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                if p_dropout >= 60:
                    st.error(f"⚠️ **STATUS: BERISIKO TINGGI DROPOUT (HIGH RISK)**\n\n- Probabilitas Dropout: **{p_dropout:.1f}%**\n- Probabilitas Graduate: **{p_graduate:.1f}%**\n\n**Rekomendasi Tindakan:** Mahasiswa perlu segera dipanggil untuk bimbingan konseling akademik khusus, asistensi remedial mata kuliah, dan evaluasi bantuan SPP.")
                elif p_dropout >= 35:
                    st.warning(f"⚡ **STATUS: BERISIKO SEDANG (MODERATE RISK / PERLU MONITORING)**\n\n- Probabilitas Dropout: **{p_dropout:.1f}%**\n- Probabilitas Graduate: **{p_graduate:.1f}%**\n\n**Rekomendasi Tindakan:** Dosen wali perlu memantau presensi dan performa ujian tengah semester.")
                else:
                    st.success(f"✅ **STATUS: AMAN / BERPOTENSI BESAR LULUS (GRADUATE)**\n\n- Probabilitas Graduate: **{p_graduate:.1f}%**\n- Probabilitas Dropout: **{p_dropout:.1f}%**\n\n**Rekomendasi:** Performa mahasiswa dalam kondisi sangat baik.")

            with p_col2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=p_dropout,
                    title={'text': "Dropout Risk Score (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#ef4444" if p_dropout >= 50 else "#10b981"},
                        'steps': [
                            {'range': [0, 35], 'color': "#dcfce7"},
                            {'range': [35, 60], 'color': "#fef9c3"},
                            {'range': [60, 100], 'color': "#fee2e2"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 60
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

with tab3:
    st.subheader("💡 Kesimpulan & Rekomendasi Action Items untuk Jaya Jaya Institut")
    st.markdown("""
    ### 1. Kesimpulan Utama
    - **Pemisahan Mahasiswa**: Model Machine Learning biner yang dilatih khusus pada mahasiswa berlabel historis (*Dropout* vs *Graduate*) berhasil mencapai akurasi **92,7%** dan ROC-AUC **0.977**.
    - **Faktor Pemicu Kunci**:
      1. **Kelulusan Mata Kuliah Semester 1 & 2**: Merupakan prediktor dominan. Mahasiswa yang gagal lulus > 50% SKS di tahun pertama hampir pasti mengalami dropout.
      2. **Tunggakan SPP (*Tuition Fees*)**: Mahasiswa yang menunggak SPP memiliki risiko dropout hingga >85%.
      3. **Penerimaan Beasiswa (*Scholarship*)**: Penerima beasiswa memiliki tingkat kelulusan sangat tinggi (>75%).
      4. **Usia Masuk**: Mahasiswa yang mendaftar pada usia matang (>25 tahun) lebih rentan dropout karena beban pekerjaan/keluarga.

    ---
    ### 2. Rekomendasi Action Items Strategis
    1. **Sistem Peringatan Dini Akademik (Academic Early Warning)**:
       - Otomatisasi notifikasi bagi mahasiswa yang lulus < 4 mata kuliah di Semester 1 untuk langsung mendapatkan bimbingan intensif dan tutor sebaya.
    2. **Skema Bantuan Finansial & Restrukturisasi SPP**:
       - Sediakan opsi cicilan SPP fleksibel dan alokasi dana beasiswa darurat untuk mahasiswa berprestasi yang mengalami kesulitan ekonomi mendadak.
    3. **Pendampingan Khusus Mahasiswa Usia Matang & Perantau**:
       - Sediakan kelas pendampingan fleksibel dan konseling adaptasi kampus bagi mahasiswa perantau (*displaced*) dan usia matang.
    4. **Pemanfaatan AI Predictor Setiap Semester**:
       - Integrasikan model prediksi Machine Learning pada Sistem Informasi Akademik (SIAKAD) kampus untuk memetakan mahasiswa berisiko secara preventif.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("👤 **Peserta**: Billy Jonathan (billy_0991kb)\n\n📌 **Kelas**: Belajar Penerapan Data Science - Proyek Akhir")
