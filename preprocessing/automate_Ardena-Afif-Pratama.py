import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    
def preprocess_data(df):
    """
    Melakukan preprocessing pada DataFrame.
    """
    if df is None:
        return None

    print("Memulai preprocessing data...")
    df_processed = df.copy()

    # 1. Menghapus Kolom Tidak Relevan
    if 'Customer ID' in df_processed.columns:
        df_processed = df_processed.drop('Customer ID', axis=1)
        print("Kolom 'Customer ID' telah dihapus.")
    else:
        print("Kolom 'Customer ID' tidak ditemukan, tidak ada yang dihapus.")

    # 2. Identifikasi Fitur Numerik dan Kategorikal
    numerical_features = df_processed.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df_processed.select_dtypes(include='object').columns.tolist()
    
    print(f"Fitur Numerik ({len(numerical_features)}): {numerical_features}")
    print(f"Fitur Kategorikal ({len(categorical_features)}): {categorical_features}")

    # 3. Membuat Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # 4. Terapkan Preprocessing
    try:
        processed_data_array = preprocessor.fit_transform(df_processed)
        
        # Mendapatkan nama kolom setelah transformasi
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        transformed_feature_names = numerical_features + list(ohe_feature_names)
        
        df_transformed = pd.DataFrame(processed_data_array, columns=transformed_feature_names, index=df_processed.index)
        
        print("Preprocessing berhasil diterapkan.")
        print(f"Dimensi data setelah preprocessing: {df_transformed.shape}")
        return df_transformed
        
    except Exception as e:
        print(f"Error saat preprocessing: {e}")
        return None

def save_data(df, output_file_path):
    """
    Menyimpan DataFrame ke file CSV.
    """
    if df is None:
        print("Tidak ada data untuk disimpan karena error pada tahap sebelumnya.")
        return

    try:
        # Membuat direktori output jika belum ada
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Direktori '{output_dir}' telah dibuat.")
            
        df.to_csv(output_file_path, index=False)
        print(f"Dataset yang sudah diproses berhasil disimpan di: {output_file_path}")
    except Exception as e:
        print(f"Gagal menyimpan file: {e}")


if __name__ == "__main__":
    print('Memulai otomatisasi preprocessing data...')

    # Memuat data
    data_raw = load_data(RAW_DATA_INPUT_PATH)

    # Preprocessing data
    data_preprocessed = preprocess_data(data_raw)

    # Menyimpan data
    save_data(data_preprocessed, PROCESSED_DATA_OUTPUT_FILE)

    print('Otomatisasi preprocessing data selesai.')