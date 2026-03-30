import math
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import os
from DiagnosisSystemClass import DiagnosisSystemClass


class SubNetwork(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(SubNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x): return self.net(x)

class GreyBoxSystem(nn.Module):
    def __init__(self, num_states, num_inputs, T_sample=0.05):
        super(GreyBoxSystem, self).__init__()
        self.T = T_sample
        self.num_states = num_states
        total_features = num_states + num_inputs
        
        self.g_func = SubNetwork(total_features, num_states)
        self.h_func = SubNetwork(total_features, 1)

    def step(self, u_t, x_t):
        inputs = torch.cat((x_t, u_t), dim=1)
        y_hat_t = self.h_func(inputs)
        x_next = x_t + self.T * self.g_func(inputs)
        return y_hat_t, x_next


class ExampleDiagnosisSystem(DiagnosisSystemClass):
    def __init__(self):
        super().__init__()
        
        self.u0_cols = ['Intercooler_pressure', 'intercooler_temperature', 'throttle_position', 'engine_speed']
        self.y0_cols = ['intake_manifold_pressure']
        
        self.u10_cols = ['delta_pressure', 'air_mass_flow', 'throttle_position']
        self.y10_cols = ['injected_fuel_mass']
        
        self.u1_cols = ['ambient_pressure', 'ambient_temperature', 'intercooler_temperature', 'throttle_position', 'engine_speed', 'injected_fuel_mass', 'wastegate_position']
        self.y1_cols = ['intake_manifold_pressure']

        self.ywaf_cols = ['injected_fuel_mass']

        self.th0 = 2801.1176         
        self.th1 = 1523.1193 
        self.th10 = 0.0002 
        self.thwaf = 0.00005

        self.warmup_steps = 700
        self._sample_count = 0      

    def Initialize(self):
        print("Grey-Box init. Reading weights and scalers (Fast Mode)...")
        
        self.model0 = GreyBoxSystem(num_states=1, num_inputs=4) 
        self.model0.load_state_dict(torch.load(os.path.join('data', 'resources', 'weights_mso0.pth')))
        self.model0.eval()
        scaler_u0 = joblib.load(os.path.join('data', 'resources', 'scaler_u_mso0.pkl'))
        scaler_y0 = joblib.load(os.path.join('data', 'resources', 'scaler_y_mso0.pkl'))
        self.x0 = torch.zeros(1, 1)
        self.e0_filt = 0.0
        
        self.model10 = GreyBoxSystem(num_states=1, num_inputs=3)
        self.model10.load_state_dict(torch.load(os.path.join('data', 'resources', 'weights_mso10.pth')))
        self.model10.eval()
        scaler_u10 = joblib.load(os.path.join('data', 'resources', 'scaler_u_mso10.pkl'))
        scaler_y10 = joblib.load(os.path.join('data', 'resources', 'scaler_y_mso10.pkl'))
        self.x10 = torch.zeros(1, 1)
        self.e10_filt = 0.0
        
        self.model1 = GreyBoxSystem(num_states=5, num_inputs=7)
        self.model1.load_state_dict(torch.load(os.path.join('data', 'resources', 'weights_mso1.pth')))
        self.model1.eval()
        scaler_u1 = joblib.load(os.path.join('data', 'resources', 'scaler_u_mso1.pkl'))
        scaler_y1 = joblib.load(os.path.join('data', 'resources', 'scaler_y_mso1.pkl'))
        self.x1 = torch.zeros(1, 5) 
        self.e1_filt = 0.0

        self.modelwaf = GreyBoxSystem(num_states=1, num_inputs=3)
        self.modelwaf.load_state_dict(torch.load(os.path.join('data', 'resources', 'weights_waf.pth')))
        self.modelwaf.eval()
        scaler_uwaf = joblib.load(os.path.join('data', 'resources', 'scaler_u_waf.pkl'))
        scaler_ywaf = joblib.load(os.path.join('data', 'resources', 'scaler_y_waf.pkl'))
        self.xwaf = torch.zeros(1, 1) 
        self.ewaf_filt = 0.0
        
        self.u0_s = torch.tensor(scaler_u0.scale_, dtype=torch.float32)
        self.u0_m = torch.tensor(scaler_u0.min_, dtype=torch.float32)
        self.y0_s = scaler_y0.scale_[0]
        self.y0_m = scaler_y0.min_[0]

        self.u10_s = torch.tensor(scaler_u10.scale_, dtype=torch.float32)
        self.u10_m = torch.tensor(scaler_u10.min_, dtype=torch.float32)
        self.y10_s = scaler_y10.scale_[0]
        self.y10_m = scaler_y10.min_[0]

        self.u1_s = torch.tensor(scaler_u1.scale_, dtype=torch.float32)
        self.u1_m = torch.tensor(scaler_u1.min_, dtype=torch.float32)
        self.y1_s = scaler_y1.scale_[0]
        self.y1_m = scaler_y1.min_[0]

        self.uwaf_s = torch.tensor(scaler_uwaf.scale_, dtype=torch.float32)
        self.uwaf_m = torch.tensor(scaler_uwaf.min_, dtype=torch.float32)
        self.ywaf_s = scaler_ywaf.scale_[0]
        self.ywaf_m = scaler_ywaf.min_[0]
        
        print("Models loaded. Ready to use.")

    def Input(self, sample):
        if not hasattr(self, '_cols_mapped'):
            cols = sample.columns.tolist()
            
            self._idx_u0 = [cols.index(c) for c in self.u0_cols]
            self._idx_y0 = cols.index(self.y0_cols[0])
            self._idx_pim = cols.index('intake_manifold_pressure')
            self._idx_pic = cols.index('Intercooler_pressure')
            self._idx_amf = cols.index('air_mass_flow')
            self._idx_thr = cols.index('throttle_position')
            self._idx_eng = cols.index('engine_speed')
            self._idx_y10 = cols.index(self.y10_cols[0])
            self._idx_u1 = [cols.index(c) for c in self.u1_cols]
            self._idx_y1 = cols.index(self.y1_cols[0])
            self._idx_ywaf = cols.index(self.ywaf_cols[0])
            
            def make_versor(vec):
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            
            self._signatures = np.array([
                make_versor(np.array([1, 1, 0, 0], dtype=float)),  # fpic
                make_versor(np.array([1, 1, 1, 0], dtype=float)),  # fpim
                make_versor(np.array([0, 0, 0, 1], dtype=float)),  # fwaf
                make_versor(np.array([1, 0, 1, 1], dtype=float)),  # fiml
            ])
            self._cols_mapped = True

        arr = sample.values[0]

        with torch.no_grad():
            
            u0_raw = torch.from_numpy(arr[self._idx_u0].astype(np.float32)).unsqueeze(0)
            u0_norm = u0_raw * self.u0_s + self.u0_m
            
            y0_hat_norm, self.x0 = self.model0.step(u0_norm, self.x0)
            y0_hat = (y0_hat_norm.item() - self.y0_m) / self.y0_s
            
            y0_true = arr[self._idx_y0]
            e0 = abs(y0_true - y0_hat) 
            
            pim = arr[self._idx_pim]
            pic = arr[self._idx_pic]
            amf = arr[self._idx_amf]
            thr = arr[self._idx_thr]
            
            delta_p = math.sqrt(abs(pim - pic))
            
            u10_raw = torch.tensor([[delta_p, amf, thr]], dtype=torch.float32)
            u10_norm = u10_raw * self.u10_s + self.u10_m
            
            y10_hat_norm, self.x10 = self.model10.step(u10_norm, self.x10)
            y10_hat = (y10_hat_norm.item() - self.y10_m) / self.y10_s
            
            y10_true = arr[self._idx_y10]
            e10 = abs(y10_true - y10_hat)                  
            
            u1_raw = torch.from_numpy(arr[self._idx_u1].astype(np.float32)).unsqueeze(0)
            u1_norm = u1_raw * self.u1_s + self.u1_m
            
            y1_hat_norm, self.x1 = self.model1.step(u1_norm, self.x1)
            y1_hat = (y1_hat_norm.item() - self.y1_m) / self.y1_s
            
            y1_true = arr[self._idx_y1]
            e1 = abs(y1_true - y1_hat)                                   

            eng_speed = arr[self._idx_eng]
            epsilon = 1e-6
            waf_x1 = math.log(eng_speed + epsilon) * amf
            waf_x2 = amf
            waf_x3 = math.log(thr + epsilon)
            
            uwaf_raw = torch.tensor([[waf_x1, waf_x2, waf_x3]], dtype=torch.float32)
            uwaf_norm = uwaf_raw * self.uwaf_s + self.uwaf_m
            
            ywaf_hat_norm, self.xwaf = self.modelwaf.step(uwaf_norm, self.xwaf)
            ywaf_hat = (ywaf_hat_norm.item() - self.ywaf_m) / self.ywaf_s
            
            ywaf_true = arr[self._idx_ywaf]
            ewaf = abs(ywaf_true - ywaf_hat)

            # --- WARMUP: przez pierwsze 50 próbek zerujemy residua ---
            self._sample_count += 1
            if self._sample_count <= self.warmup_steps:
                e0 = 0.0
                e10 = 0.0
                e1 = 0.0
                ewaf = 0.0

            self.e0_filt   = 0.001 * e0   + 0.999 * self.e0_filt
            self.e10_filt  = 0.001 * e10  + 0.999 * self.e10_filt
            self.e1_filt   = 0.001 * e1   + 0.999 * self.e1_filt
            self.ewaf_filt = 0.001 * ewaf + 0.999 * self.ewaf_filt  
        
        b0 = 1 if self.e0_filt > self.th0 else 0
        b10 = 1 if self.e10_filt > self.th10 else 0
        b1 = 1 if self.e1_filt > self.th1 else 0
        bwaf = 1 if self.ewaf_filt > self.thwaf else 0

        detection = [1] if (b0 or b10 or b1 or bwaf) else [0]
        isolation = np.zeros((1, 5)) 

        if detection[0] == 1: 
            obs_norm = math.sqrt(b0*b0 + b10*b10 + b1*b1 + bwaf*bwaf)
            if obs_norm > 0:
                observed_versor = np.array([b0/obs_norm, b10/obs_norm, b1/obs_norm, bwaf/obs_norm], dtype=float)
            else:
                observed_versor = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
            
            scores = np.dot(self._signatures, observed_versor)
            total_score = np.sum(scores)
            
            if total_score > 0:
                isolation[0, :4] = scores / total_score
            else:
                isolation[0, 4] = 1.0
                
            max_idx = np.argmax(isolation[0])  
            isolation = np.zeros((1, 5))       
            isolation[0, max_idx] = 1.0

            case1 = (b0 == 0 and b10 == 1 and b1 == 1 and bwaf == 1)
            case2 = (b0 == 1 and b10 == 1 and b1 == 1 and bwaf == 1)
            
            if case1 or case2:
                isolation = np.array([[0.0, 0.0, 0.0, 0.0, 1.0]])
                
        return detection, isolation