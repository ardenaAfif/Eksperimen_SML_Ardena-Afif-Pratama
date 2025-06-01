import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Direktori tempat script ini berada (preprocessing/)
RAW_DATA_INPUT_PATH = os.path.join(BASE_DIR, '..', 'shopping_trends_raw', 'shopping_trends.csv')
PROCESSED_DATA_OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'preprocessing')
PROCESSED_DATA_OUTPUT_FILE = os.path.join(PROCESSED_DATA_OUTPUT_DIR, 'processed_shopping_trends.csv')

def load_data(file_path):
    """
    Memuat data dari file CSV.
    """
    print(f"Memuat data dari: {file_path}")
    try:
        df = pd.read_csv(file_path)
        print("Data berhasil dimuat.")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {file_path}")
        return None
    except Exception as e:
        print(f"Error saat memuat data: {e}")
        return None
    
def preprocess_data_for_classification(df, target_column_name='Category'):
    """
    Melakukan preprocessing pada DataFrame untuk tugas klasifikasi.
    Target default adalah 'Category'.
    """
    if df is None:
        return None, None # Mengembalikan None untuk X dan y jika df None

    print(f"Memulai preprocessing data untuk klasifikasi (Target: {target_column_name})...")
    
    # 1. Validasi kolom target
    if target_column_name not in df.columns:
        print(f"Error: Kolom target '{target_column_name}' tidak ditemukan.")
        return None, None
    
    # 2. Pisahkan target variabel y dan Label Encode
    y_target_series = df[target_column_name].copy()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_target_series)
    
    print(f"Target '{target_column_name}' telah di-LabelEncode. Shape y_encoded: {y_encoded.shape}")
    
    # 3. Siapkan fitur X (hapus target asli dan ID)
    X_features = df.drop(columns=[target_column_name])
    if 'Customer ID' in X_features.columns:
        X_features = X_features.drop('Customer ID', axis=1)
        print("Kolom 'Customer ID' telah dihapus dari fitur X.")
    else:
        print("Kolom 'Customer ID' tidak ditemukan di fitur X, tidak ada yang dihapus.")
    print(f"Shape X_features (setelah drop target & ID): {X_features.shape}")

    # 4. Identifikasi Fitur Numerik dan Kategorikal dari X_features
    numerical_features = X_features.select_dtypes(include=np.number).columns.tolist()
    categorical_features_for_ohe = X_features.select_dtypes(include='object').columns.tolist()
    
    print(f"Fitur Numerik yang akan di-scale ({len(numerical_features)}): {numerical_features}")
    print(f"Fitur Kategorikal yang akan di-OHE ({len(categorical_features_for_ohe)}): {categorical_features_for_ohe}")

    # 5. Membuat Preprocessor
    # StandardScaler untuk numerik, OneHotEncoder untuk kategorikal (yang tersisa di X_features)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features_for_ohe)
        ],
        remainder='passthrough' # Seharusnya tidak ada sisa jika semua tipe data ditangani
    )

    # 6. Terapkan Preprocessing pada X_features
    print(f"Menerapkan ColumnTransformer pada X_features dengan shape: {X_features.shape}")
    try:
        X_transformed_array = preprocessor.fit_transform(X_features)
        
        # Mendapatkan nama kolom setelah transformasi
        ohe_feature_names = []
        if categorical_features_for_ohe: # Hanya jika ada fitur kategorikal untuk OHE
             ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features_for_ohe)
        
        final_feature_names = numerical_features + list(ohe_feature_names) 
        
        # Cek apakah jumlah nama fitur hasil transformasi sesuai dengan jumlah kolom array
        if X_transformed_array.shape[1] != len(final_feature_names):
            print("PERINGATAN: Jumlah nama fitur tidak cocok dengan jumlah kolom hasil transformasi!")
            print(f"Kolom array: {X_transformed_array.shape[1]}, Nama fitur terkonstruksi: {len(final_feature_names)}")
            
        X_transformed_df = pd.DataFrame(X_transformed_array, columns=final_feature_names, index=X_features.index)
        
        print("Preprocessing fitur X berhasil diterapkan.")
        print(f"Dimensi X_transformed_df setelah preprocessing: {X_transformed_df.shape}")
        return X_transformed_df, y_encoded # Mengembalikan X yang sudah diproses dan y yang sudah di-LabelEncode
        
    except Exception as e:
        print(f"Error saat preprocessing fitur X: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_combined_data(X_df, y_encoded_array, target_name, output_file_path):
    """
    Menggabungkan X_df dan y_encoded_array (sebagai Series), lalu menyimpan ke file CSV.
    """
    if X_df is None or y_encoded_array is None:
        print("Tidak ada data X atau y untuk disimpan karena error pada tahap sebelumnya.")
        return

    try:
        # Jadikan y_encoded_array sebagai Pandas Series dengan nama dan indeks yang sesuai
        y_series = pd.Series(y_encoded_array, name=target_name, index=X_df.index)
        
        # Menggabungkan fitur yang sudah diproses dengan target yang sudah di-LabelEncode
        final_df_to_save = pd.concat([X_df, y_series], axis=1)
        print(f"Menggabungkan X_transformed dan y_encoded. Shape akhir: {final_df_to_save.shape}")

        # Membuat direktori output jika belum ada
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Direktori '{output_dir}' telah dibuat di {output_dir}")
            
        final_df_to_save.to_csv(output_file_path, index=False)
        print(f"Dataset yang sudah diproses (fitur X dan target y_encoded) berhasil disimpan di: {output_file_path}")
    except Exception as e:
        print(f"Gagal menyimpan file gabungan: {e}")



if __name__ == "__main__":
    print('--- Memulai otomatisasi preprocessing data untuk Klasifikasi ---')

    # Memuat data mentah
    data_raw = load_data(RAW_DATA_INPUT_PATH)

    # Preprocessing data (mengembalikan X_processed dan y_encoded)
    # Target default adalah 'Category'
    X_processed, y_encoded = preprocess_data_for_classification(data_raw, target_column_name='Category')

    # Menyimpan data gabungan (X_processed + y_encoded)
    if X_processed is not None and y_encoded is not None:
        save_combined_data(X_processed, y_encoded, 'Encoded_Category', PROCESSED_DATA_OUTPUT_FILE)
    else:
        print("Preprocessing gagal, tidak ada data untuk disimpan.")

    print('--- Otomatisasi preprocessing data untuk Klasifikasi selesai ---')
