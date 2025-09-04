import subprocess
import os
import sys
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings('ignore')

# 1. Define los argumentos para cada llamada
script_name = 'inferencia.py'
model_path = '/home/gaara/mnt/USM/2025-02/INF221/repos/KardiaApp/kardia_s4d/s4_results/S4D/model.pt'
args = [ 'python', script_name,'--model_path', model_path, '--ecg_path' ]

# 2. Define la ruta al segundo script y el intérprete (python)

# 3. Itera sobre los argumentos y ejecuta el segundo script
sys.path.extend([os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))])
print("Buscando en:",sys.path)
#directorio = '../../repos/data/physionet.org/files/ptb-xl/1.0.1/records100'  # Reemplaza con la ruta de tu directorio
directorio = '/home/gaara/mnt/USM/2025-02/INF221/repos/data/physionet.org/files/ptb-xl/1.0.1/records100'

try:
    print("Intentando Listar")
    archivos_y_directorios = os.listdir(directorio)
    archivos_y_directorios.sort()
    i = 0
    for nombre in archivos_y_directorios:
        # if i % 100 == 0: print(nombre) 

        try:
            new_dir = directorio+'/'+nombre
            ecgs = os.listdir(new_dir)
            ecgs.sort()
            i = 0
            ls = ecgs[:-1]
            for name_file in tqdm(ls, desc=f"Procesando archivos ECG carpeta {nombre}"):
                ecg_name, ext = name_file.split('.')
                ecg_path  = new_dir+'/'+ecg_name
                if ext == 'hea':
                #     if i % 100 == 0:
                #         print(f"Llamando a {script_name} con argumentos:\n--model_path{model_path}   \n--ecg_path {ecg_path}")
                    try:
                        # Construye la lista de comands para subprocess.run
                        # El primer elemento es el intérprete, el segundo es el script
                        # Luego, se agregan los argumentos 
                        comando = ['python', script_name, '--model_path', model_path, '--ecg_path', ecg_path]            
                        # Ejecuta el comando y espera a que termine
                        subprocess.run(comando, stdout=open(os.devnull, 'w'), check=True)
                    
                    except FileNotFoundError:
                        print(f"Error: No se encontró el script '{script_name}'.")
                    except subprocess.CalledProcessError as e:
                        print(f"El script falló con el código de salida {e.returncode}.")
                    except Exception as e:
                        print(f"Ocurrió un error inesperado: {e}")

        except FileNotFoundError:
            print(f"Error: El directorio '{directorio}' no fue encontrado.")
        except Exception as e:
            print(f"Ocurrió un error: {e}")

except FileNotFoundError:
    print(f"Error: El directorio '{directorio}' no fue encontrado.")
except Exception as e:
    print(f"Ocurrió un error: {e}")


