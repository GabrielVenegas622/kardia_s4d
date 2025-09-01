import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import wfdb
import h5py
import os

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Modifica esta línea para que apunte a la carpeta donde descomprimiste PTB-XL.
PATH_A_PTBXL = r'/home/gaara/mnt/USM/2025-02/INF221/repos/data/physionet.org/files/ptb-xl/1.0.1'
# --------------------

def preprocesar_ptbxl(path_a_datos, sampling_rate=100):
    """
    Esta función lee los datos crudos del dataset PTB-XL, procesa las señales y las etiquetas,
    y guarda los resultados en los archivos x.hdf5 y y.csv que necesita Train.py.
    """
    print("--- Iniciando pre-procesamiento de PTB-XL ---")

    try:
        # 1. Cargar el archivo CSV principal que contiene toda la metadata
        print(f"Leyendo metadatos de: {os.path.join(path_a_datos, 'ptbxl_database.csv')}")
        ptbxl_data = pd.read_csv(os.path.join(path_a_datos, 'ptbxl_database.csv'), index_col='ecg_id')

        # 2. Cargar las señales de ECG
        print(f"Cargando señales... Esto puede tardar unos minutos.")
        # La función 'rdrecord' de wfdb lee los archivos .hea y .dat
        signals = []
        for ecg_id in tqdm(ptbxl_data.index, desc="Cargando ECGs"):
            # Construimos el nombre del archivo. Usamos os.path.join para compatibilidad.
            # Los archivos están en carpetas 'records100/00000/' etc.
            filename = os.path.join(path_a_datos, ptbxl_data.loc[ecg_id, 'filename_lr'] if sampling_rate == 100 else ptbxl_data.loc[ecg_id, 'filename_hr'])
            signal, meta = wfdb.rdsamp(filename)
            signals.append(signal)
        
        # Convertimos la lista de señales a un único array de NumPy
        X = np.array(signals)
        print(f"Procesamiento de señales completo. Forma de X: {X.shape}")

        # 3. Procesar las etiquetas (diagnósticos)
        # El script Train.py espera 8 columnas. Usaremos 8 diagnósticos comunes.
        # Puedes adaptar esto si necesitas otras clases.
        y = ptbxl_data.copy()
        
        # Seleccionamos las columnas de diagnóstico y las convertimos a binario (0 o 1)

        #--------------------- Diagnóstico Normal ----------------------
        y['NORM'] = y.scp_codes.str.contains('NORM').astype(int)
        #---------------------------------------------------------------

        #-------------- Diagnóstico Infarto al Miocardio ---------------
        y['MI'] = y.scp_codes.str.contains('MI').astype(int)
        #---------------------------------------------------------------

        #-------------- Diagnóstico de lesión --------------------------
        y['INJ'] = y.scp_codes.str.contains('INJ').astype(int)
        #---------------------------------------------------------------

        #-------------- Diagnóstico de Isquemias -----------------------
        y['ISC'] = y.scp_codes.str.contains('ISC').astype(int)
        #---------------------------------------------------------------

        #---------------------Arritmias cardiácas ----------------------
        # Arritmias que se originan en el nodo sinusal o en las vías de conducción supraventriculares: [SARRH, SVARR]
        # Arritmias que se originan fuera de las vías normales de conducción, como los patrones rítmicos irregulares : [BIGU, TRIGU]
        arrh_labels = 'SARRH|SVARR|BIGU|TRIGU'
        y['ARRY'] = y.scp_codes.str.contains(arrh_labels, na=False).astype(int)
        #---------------------------------------------------------------

        #------------------ Diagnóstico cambios en ST/T ----------------
        sttc_labels = 'NDT|NST_|DIG|LNGQT|ISC_|ISCAL|ISCIN|ISCIL|ISCAS|ISCLA|ANEUR|EL|ISCAN'
        y['STTC'] = y.scp_codes.str.contains(sttc_labels, na=False).astype(int)
        #---------------------------------------------------------------

        #------------------ Diagnóstico trastorno de la conducción -----
        cd_labels = 'LAFB|IRBBB|1AVB|IVCD|CRBBB|CLBBB|LPFB|WPW|ILBBB|3AVB|2AVB'
        y['CD'] =  y.scp_codes.str.contains(cd_labels, na=False).astype(int)
        #---------------------------------------------------------------

        #------------------ Diagnóstico Hipertrofia --------------------
        hyp_labels = 'LVH|LAO/LAE|RVH|RAO/RAE|SEHYP'
        y['HYP'] = y.scp_codes.str.contains(hyp_labels, na=False).astype(int)
        #---------------------------------------------------------------
        
        # Nos quedamos solo con las 8 columnas de etiquetas
        y = y[['NORM', 'MI', 'INJ', 'ISC', 'ARRY', 'STTC', 'CD', 'HYP']]
        print(f"Procesamiento de etiquetas completo. Forma de y: {y.shape}")


        # 4. Guardar los archivos procesados
        print("Guardando archivos procesados 'x.hdf5' y 'y.csv'...")
        with h5py.File('x.hdf5', 'w') as f:
            f.create_dataset('tracings', data=X, compression='gzip')
        
        y.to_csv('y.csv', index=False)
        
        print("\n--- ¡Pre-procesamiento finalizado con éxito! ---")
        print("Ya puedes ejecutar Train.py")

    except FileNotFoundError:
        print("\n[ERROR] No se pudo encontrar la carpeta de PTB-XL. Por favor, verifica la variable 'PATH_A_PTBXL' en este script.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error inesperado: {e}")

if __name__ == '__main__':
    # Verificamos si la librería wfdb está instalada
    try:
        import wfdb
    except ImportError:
        print("[ERROR] La librería 'wfdb' no está instalada. Por favor, ejecute 'pip install wfdb' en su entorno.")
    else:
        preprocesar_ptbxl(PATH_A_PTBXL)
