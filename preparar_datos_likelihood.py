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
        y_dict = { 'NORM' : [] , 'MI' : [],
                  'INJ' : [], 'ISC' : [],
                  'ARRY' : [], 'STTC' : [],
                  'CD' : [], 'HYP' : [],
                  }
        
        key_labels = { 'NORM' : ['NORM'],
                      'MI' : ['IMI','ASMI','ILMI','AMI','ALMI','LMI','IPLMI','IPMI','PMI'],
                      'INJ' : ['INJAS','INJAL','INJIN','INJLA','INJIL'], 
                      'ISC' : ['ISC_','ISCAL','ISCIN','ISCIL','ISCAS','ISCLA','ISCAN'],
                      'ARRY' : ['SARRH', 'SVARR', 'BIGU', 'TRIGU'], 
                      'STTC' : ['NDT','NST_','DIG','LNGQT','ISC_','ISCAL','ISCIN','ISCIL','ISCAS','ISCLA','ANEUR','EL','ISCAN'],
                      'CD' : ['LAFB','IRBBB','1AVB','IVCD','CRBBB','CLBBB','LPFB','WPW','ILBBB','3AVB','2AVB'],
                      'HYP' : ['LVH','LAO/LAE','RVH','RAO/RAE','SEHYP'],
                }


        for row in tqdm(y.itertuples(), total=len(y), desc="Procesando filas"):
            tmp_dict = eval(row.scp_codes)
            
            # Diagnóstico normal
            y_dict['NORM'].append(next((tmp_dict.get(key)/100.0 for key in key_labels['NORM'] if key in tmp_dict), 0))
            # Diagnóstico MI
            y_dict['MI'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['MI'] if key in tmp_dict), 0))
            # Diagnóstico lesión
            y_dict['INJ'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['INJ'] if key in tmp_dict), 0))
            # Diagnóstico isquemia
            y_dict['ISC'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['ISC'] if key in tmp_dict), 0))
            # Diagnóstico arritmia 
            y_dict['ARRY'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['ARRY'] if key in tmp_dict), 0))
            # Diagnóstico cambios en el sttc 
            y_dict['STTC'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['STTC'] if key in tmp_dict), 0))
            # Diagnóstico síndrome de conducción
            y_dict['CD'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['CD'] if key in tmp_dict), 0))
            # Diagnóstico hypertrofia
            y_dict['HYP'].append(next((tmp_dict.get(key)/100.0  for key in key_labels['HYP'] if key in tmp_dict), 0))

        
        y_df = pd.DataFrame(y_dict)
        y_df.to_csv('y_output_gemini.csv', index=False)



        print(f"Procesamiento de etiquetas completo. Forma de y: {y.shape}")


        # 4. Guardar los archivos procesados
        print("Guardando archivos procesados 'x.hdf5' y 'y.csv'...")
        with h5py.File('x.hdf5', 'w') as f:
            f.create_dataset('tracings', data=X, compression='gzip')
        
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
