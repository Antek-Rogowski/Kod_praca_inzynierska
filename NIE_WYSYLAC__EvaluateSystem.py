import os
import glob
import subprocess
import pandas as pd
import numpy as np

# --- KONFIGURACJA ---
DATA_DIR = os.path.join('data', 'training_data')
RESULTS_DIR = "results" 
FAULT_TIME_FILE = os.path.join(DATA_DIR, "ftp75city2_fault_time.txt")

def load_ftp_fault_times(filepath):
    fault_times = {}
    if not os.path.exists(filepath):
        print(f"  [!] Ostrzeżenie: Nie znaleziono pliku {filepath}")
        return fault_times
        
    with open(filepath, 'r') as f:
        lines = f.readlines()[1:] 
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                filename = parts[0]
                time_val = float(parts[-1])
                fault_times[filename] = time_val
    return fault_times

def get_true_fault(filename):
    if "_f_pic" in filename: return "fpic"
    if "_f_pim" in filename: return "fpim"
    if "_f_waf" in filename: return "fwaf"
    if "_f_iml" in filename: return "fiml"
    if "_NF" in filename: return "NF"
    return "Unknown"

def evaluate():
    ftp_times = load_ftp_fault_times(FAULT_TIME_FILE)
    
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    waf_files = glob.glob(os.path.join(DATA_DIR, "*waf*.csv"))
    test_files = [f for f in all_files if not os.path.basename(f).startswith("output_")]
    
    results = []
    
    print(f"Rozpoczynam ewaluację {len(test_files)} plików...\n")
    
    for file_path in test_files:
        basename = os.path.basename(file_path)
        file_no_ext = os.path.splitext(basename)[0]
        
        print(f"Przetwarzanie: {basename}...")
        
        result = subprocess.run(["python", "RunDiagnoser.py", file_path], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"  [!] CRASH SKRYPTU RunDiagnoser.py na pliku {basename}!")
            print(f"  [!] Powód:\n{result.stderr}")
            continue
        
        output_file = os.path.join(RESULTS_DIR, f"output_{basename}")
        
        if not os.path.exists(output_file):
            print(f"  [!] BŁĄD: Brak pliku wyjściowego {output_file}. Sprawdź, czy RunDiagnoser na pewno go tam zapisał.")
            continue
            
        try:
            df = pd.read_csv(output_file, sep=None, engine='python')
        except Exception as e:
            print(f"  [!] Błąd odczytu {output_file}: {e}")
            continue
            
        true_fault = get_true_fault(basename)
        if true_fault == "NF":
            fault_time = float('inf') 
        elif "wltp" in basename:
            fault_time = 120.0
        elif "ftp75city2" in basename:
            fault_time = ftp_times.get(file_no_ext, 0.0)
        else:
            fault_time = 0.0
            
        pre_fault_samples = df[df['sample_time'] < fault_time]
        if len(pre_fault_samples) > 0:
            far = pre_fault_samples['detection'].mean()
        else:
            far = 0.0 
            
        if true_fault == "NF":
            tdr = 1.0 
            fia = 1.0 
            score = 1.0 - far
        else:
            post_fault_samples = df[df['sample_time'] >= fault_time]
            if len(post_fault_samples) > 0:
                tdr = post_fault_samples['detection'].mean()
            else:
                tdr = 0.0
                
            detected_post_fault = post_fault_samples[post_fault_samples['detection'] == 1]
            if len(detected_post_fault) > 0:
                rank_col = f"{true_fault}_rank"
                if rank_col in detected_post_fault.columns:
                    fia = detected_post_fault[rank_col].mean()
                else:
                    fia = 0.0
            else:
                fia = 0.0 
                
            score = ((1.0 - far) + tdr + fia) / 3.0
            
        results.append({
            'Plik': basename,
            'Typ': true_fault,
            'FAR': far,
            'TDR': tdr if true_fault != "NF" else np.nan,
            'FIA': fia if true_fault != "NF" else np.nan,
            'Punkty': score
        })
        
        print(f"  Wynik: {score*100:.1f}% | FAR: {far*100:.1f}%, TDR: {tdr*100:.1f}%, FIA: {fia*100:.1f}%")

    if not results:
        print("\n[!] Brak wyników do podsumowania. Sprawdź błędy powyżej.")
        return

    res_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("                 RAPORT KOŃCOWY")
    print("="*60)
    
    total_points = res_df['Punkty'].sum()
    max_points = len(res_df)
    faults_df = res_df[res_df['Typ'] != 'NF']
    
    print(f"Przetworzono plików: {max_points}")
    print(f"Średni False Alarm Rate (FAR):     {res_df['FAR'].mean()*100:.2f} % (im mniej tym lepiej)")
    print(f"Średnia Skuteczność Detekcji (TDR): {faults_df['TDR'].mean()*100:.2f} %")
    print(f"Średnia Precyzja Izolacji (FIA):    {faults_df['FIA'].mean()*100:.2f} %")
    print("-" * 60)
    print(f"ZDOBYŁEŚ PUNKTÓW: {total_points:.2f} na {max_points:.2f} możliwych!")
    print(f"Twój Total Score: { (total_points / max_points)*100:.2f} %")
    print("="*60)

if __name__ == "__main__":
    evaluate()