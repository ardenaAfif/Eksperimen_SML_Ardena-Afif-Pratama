import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
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
    
def preprocess_data_for_regression(df):
    """
    Melakukan preprocessing pada DataFrame untuk tugas regresi (target: Purchase Amount (USD)).
    """
    if df is None:
        return None, None # Mengembalikan None untuk X dan y jika df None

    print("Memulai preprocessing data untuk regresi...")
    
    # 1. Pisahkan target variabel y terlebih dahulu
    if 'Purchase Amount (USD)' not in df.columns:
        print("Error: Kolom target 'Purchase Amount (USD)' tidak ditemukan.")
        return None, None
    y_target = df['Purchase Amount (USD)'].copy()
    X_features = df.drop('Purchase Amount (USD)', axis=1)
    print(f"Target 'Purchase Amount (USD)' telah dipisahkan. Shape y: {y_target.shape}")
    print(f"Shape X awal (sebelum drop Customer ID): {X_features.shape}")

    # 2. Menghapus Kolom Tidak Relevan dari X_features
    if 'Customer ID' in X_features.columns:
        X_features = X_features.drop('Customer ID', axis=1)
        print("Kolom 'Customer ID' telah dihapus dari fitur X.")
    else:
        print("Kolom 'Customer ID' tidak ditemukan di fitur X, tidak ada yang dihapus.")
    
    # 3. Daftar Fitur Kategorikal (sesuai contoh Anda)
    user_categorical_features = [
        'Gender', 'Item Purchased', 'Category', 'Location', 'Size', 'Color', 
        'Season', 'Subscription Status', 'Payment Method', 'Shipping Type', 
        'Discount Applied', 'Promo Code Used', 'Preferred Payment Method', 
        'Frequency of Purchases'
    ]
    
    # Filter daftar ini untuk memastikan hanya kolom yang ada di X_features yang digunakan
    valid_categorical_features = [col for col in user_categorical_features if col in X_features.columns]
    print(f"Fitur Kategorikal yang valid untuk di-encode ({len(valid_categorical_features)}): {valid_categorical_features}")

    # Fitur yang tidak ada di valid_categorical_features akan menjadi 'passthrough'
    
    # 4. Membuat Preprocessor (OneHotEncoder untuk kategorikal, sisanya passthrough)
    preprocessor = ColumnTransformer(
        transformers=[
            ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), valid_categorical_features)
        ],
        remainder='passthrough' # Fitur numerik dan kategorikal lain (jika ada) akan dilewatkan
    )

    # 5. Terapkan Preprocessing pada X_features
    print(f"Menerapkan ColumnTransformer pada X_features dengan shape: {X_features.shape}")
    try:
        X_transformed_array = preprocessor.fit_transform(X_features)
        
        # Mendapatkan nama kolom setelah transformasi
        ohe_feature_names = preprocessor.named_transformers_['one_hot'].get_feature_names_out(valid_categorical_features)
        
        # Mendapatkan nama kolom yang dilewatkan (passthrough)
        # Urutan mereka dipertahankan oleh ColumnTransformer setelah kolom yang ditransformasi.
        passthrough_feature_names = [
            col for col in X_features.columns if col not in valid_categorical_features
        ]
        
        # Nama kolom gabungan (OHE dulu, baru passthrough)
        final_feature_names = list(ohe_feature_names) + passthrough_feature_names
        
        X_transformed_df = pd.DataFrame(X_transformed_array, columns=final_feature_names, index=X_features.index)
        
        print("Preprocessing fitur X berhasil diterapkan.")
        print(f"Dimensi X_transformed_df setelah preprocessing: {X_transformed_df.shape}")
        return X_transformed_df, y_target
        
    except Exception as e:
        print(f"Error saat preprocessing fitur X: {e}")
        import traceback
        traceback.print_exc() # Cetak traceback untuk debug lebih detail
        return None, None

def save_data(X_df, y_series, output_file_path):
    """
    Menggabungkan X_df dan y_series, lalu menyimpan ke file CSV.
    """
    if X_df is None or y_series is None:
        print("Tidak ada data X atau y untuk disimpan karena error pada tahap sebelumnya.")
        return

    try:
        # Menggabungkan fitur yang sudah diproses dengan target
        final_df_to_save = pd.concat([X_df, y_series.rename('Purchase Amount (USD)')], axis=1)
        print(f"Menggabungkan X_transformed dan y. Shape akhir: {final_df_to_save.shape}")

        # Membuat direktori output jika belum ada
        output_dir = os.path.dirname(output_file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Direktori '{output_dir}' telah dibuat di {output_dir}")
            
        final_df_to_save.to_csv(output_file_path, index=False)
        print(f"Dataset yang sudah diproses (fitur X dan target y) berhasil disimpan di: {output_file_path}")
    except Exception as e:
        print(f"Gagal menyimpan file gabungan: {e}")


if __name__ == "__main__":
    print('Memulai otomatisasi preprocessing data...')

    # Memuat data
    data_raw = load_data(RAW_DATA_INPUT_PATH)

    # Preprocessing data
    X_processed, y_target = preprocess_data_for_regression(data_raw)

    if X_processed is not None and y_target is not None:
        save_data(X_processed, y_target, PROCESSED_DATA_OUTPUT_FILE)
    else:
        print("Preprocessing gagal, tidak ada data untuk disimpan.")

    print('--- Otomatisasi preprocessing data untuk Regresi selesai ---')
