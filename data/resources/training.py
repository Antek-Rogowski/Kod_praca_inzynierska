import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib
import glob
import copy
import os

# ==============================================================================
# Common functions and classes
# ==============================================================================

class SubNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SubNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.net(x)

class GreyBoxTrainer(nn.Module):
    def __init__(self, num_states, num_inputs, T_sample=0.05):
        super(GreyBoxTrainer, self).__init__()
        self.T = T_sample
        self.num_states = num_states
        self.num_inputs = num_inputs
        total_feat = self.num_states + self.num_inputs
        
        self.g_func = SubNetwork(total_feat, self.num_states)
        self.h_func = SubNetwork(total_feat, 1)

    def forward(self, u_seq, x_init):
        x_t = x_init
        preds = []
        for t in range(u_seq.size(1)):
            inputs = torch.cat((x_t, u_seq[:, t, :]), dim=1)
            preds.append(self.h_func(inputs))
            x_t = x_t + self.T * self.g_func(inputs)
        return torch.stack(preds, dim=1)

def train_greybox_model(model, dataloader, model_name, epochs=2000, patience=80, min_delta=1e-6):
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.97)
    criterion = nn.L1Loss() 
    
    best_loss = float('inf') 
    patience_counter = 0    
    best_model_weights = copy.deepcopy(model.state_dict()) 
    
    model.train()
    print(f"Trainig has been started: {model_name}")
    print(f"Max epochs: {epochs}, Patience: {patience}")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for u_seq, y_true_seq in dataloader:
            x_init = torch.zeros(u_seq.size(0), model.num_states)
            optimizer.zero_grad()
            y_pred_seq = model(u_seq, x_init)
            loss = criterion(y_pred_seq.view_as(y_true_seq), y_true_seq)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        
        if (epoch + 1) % 1 == 0:
            print(f"[{model_name}] Epoch {epoch+1}/{epochs} | MAE Loss: {avg_loss:.6f}")
            
        if best_loss - avg_loss > min_delta:
            best_loss = avg_loss          
            patience_counter = 0           
            best_model_weights = copy.deepcopy(model.state_dict()) 
        else:
            patience_counter += 1             
            
        if patience_counter >= patience:
            print(f"[!] EARLY STOPPING in epoch {epoch+1}. No improvement since {patience} epochs.")
            break 

    model.load_state_dict(best_model_weights)
    print(f"Training has been finished ({model_name}) | Best MAE: {best_loss:.6f}\n")
    return model

def create_sequences(inputs, targets, seq_len=600):
    xs, ys = [], []
    for i in range(0, len(inputs) - seq_len + 1, seq_len):
        xs.append(inputs[i : i + seq_len])
        ys.append(targets[i : i + seq_len])
    return np.array(xs), np.array(ys)

# ==============================================================================
# Data loading and preprocessing
# ==============================================================================

def load_and_prep_data(file_path):
    df = pd.read_csv(file_path)
    
    pim = df['intake_manifold_pressure']
    pic = df['Intercooler_pressure']
    df['delta_pressure'] = np.sqrt(np.abs(pim - pic))
    
    epsilon = 1e-6
    df['waf_x1'] = np.log(df['engine_speed'] + epsilon) * df['air_mass_flow']
    df['waf_x2'] = df['air_mass_flow']
    df['waf_x3'] = np.log(df['throttle_position'] + epsilon)
    
    return df

DATA_DIR = os.path.join('data', 'trainingdata')
os.makedirs('params', exist_ok=True)

HEALTHY_FILES = glob.glob(os.path.join(DATA_DIR, "*_NF*.csv"))
FAULTY_FILES_PIM = glob.glob(os.path.join(DATA_DIR, "*_f_pim*.csv"))
FAULTY_FILES_PIC = glob.glob(os.path.join(DATA_DIR, "*_f_pic*.csv"))
FAULTY_FILES_WAF = glob.glob(os.path.join(DATA_DIR, "*_f_waf*.csv"))
FAULTY_FILES_IML = glob.glob(os.path.join(DATA_DIR, "*_f_iml*.csv"))

healthy_dfs = [load_and_prep_data(f) for f in HEALTHY_FILES]
faulty_waf = [load_and_prep_data(f) for f in FAULTY_FILES_WAF]
faulty_iml = [load_and_prep_data(f) for f in FAULTY_FILES_IML]
faulty_pic = [load_and_prep_data(f) for f in FAULTY_FILES_PIC]
faulty_pim = [load_and_prep_data(f) for f in FAULTY_FILES_PIM]

# ==============================================================================
# Models config
# ==============================================================================

models_config = {
    'mso0': {
        'dfs': healthy_dfs + faulty_waf,
        'u_cols': ['Intercooler_pressure', 'intercooler_temperature', 'throttle_position', 'engine_speed'],
        'y_cols': ['intake_manifold_pressure'],
        'num_states': 1
    },
    'mso10': {
        'dfs': healthy_dfs + faulty_waf + faulty_iml,
        'u_cols': ['delta_pressure', 'air_mass_flow', 'throttle_position'],
        'y_cols': ['injected_fuel_mass'],
        'num_states': 1
    },
    'mso1': {
        'dfs': healthy_dfs + faulty_pic + faulty_waf,
        'u_cols': ['ambient_pressure', 'ambient_temperature', 'intercooler_temperature', 'throttle_position', 'engine_speed', 'injected_fuel_mass', 'wastegate_position'],
        'y_cols': ['intake_manifold_pressure'],
        'num_states': 5
    },
    'waf': {
        'dfs': healthy_dfs + faulty_pic + faulty_pim,
        'u_cols': ['waf_x1', 'waf_x2', 'waf_x3'],
        'y_cols': ['injected_fuel_mass'],
        'num_states': 1
    }
}

# ==============================================================================
# Main training
# ==============================================================================

for model_name, config in models_config.items():
    df_train = pd.concat(config['dfs'], ignore_index=True)
    
    scaler_u, scaler_y = MinMaxScaler(), MinMaxScaler()
    u_norm = scaler_u.fit_transform(df_train[config['u_cols']].values)
    y_norm = scaler_y.fit_transform(df_train[config['y_cols']].values)
    
    X_seq, Y_seq = create_sequences(u_norm, y_norm)
    loader = DataLoader(
        TensorDataset(torch.tensor(X_seq, dtype=torch.float32), torch.tensor(Y_seq, dtype=torch.float32)), 
        batch_size=16, shuffle=True
    )
    
    num_inputs = len(config['u_cols'])
    num_states = config['num_states']
    
    model = GreyBoxTrainer(num_states=num_states, num_inputs=num_inputs)
    model = train_greybox_model(model, loader, model_name=model_name, epochs=2000)
    
    torch.save(model.state_dict(), os.path.join('params', f'weights_{model_name}.pth'))
    joblib.dump(scaler_u, os.path.join('params', f'scaler_u_{model_name}.pkl'))
    joblib.dump(scaler_y, os.path.join('params', f'scaler_y_{model_name}.pkl'))
    
    print(f"[SUCCESS] Files saved for the {model_name.upper()} in folder 'params/'.\n")

print("=========================================================")
print("Everything done. All 4 models are ready.")
print("=========================================================")