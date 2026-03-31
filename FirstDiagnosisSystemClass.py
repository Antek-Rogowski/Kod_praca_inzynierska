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

        self.u0_cols  = ['Intercooler_pressure', 'intercooler_temperature', 'throttle_position', 'engine_speed']
        self.y0_cols  = ['intake_manifold_pressure']

        self.u10_cols = ['delta_pressure', 'air_mass_flow', 'throttle_position']
        self.y10_cols = ['injected_fuel_mass']

        self.u1_cols  = ['ambient_pressure', 'ambient_temperature', 'intercooler_temperature',
                         'throttle_position', 'engine_speed', 'injected_fuel_mass', 'wastegate_position']
        self.y1_cols  = ['intake_manifold_pressure']

        self.ywaf_cols = ['injected_fuel_mass']

        self.th0    = 2801.1176
        self.th1    = 1523.1193
        self.th10   = 0.0002
        self.thwaf  = 0.00006

        self.warmup_steps = 700
        self._sample_count = 0

    def Initialize(self):
        print("Grey-Box init. Reading weights and scalers...")

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

        self.u0_s  = torch.tensor(scaler_u0.scale_,  dtype=torch.float32)
        self.u0_m  = torch.tensor(scaler_u0.min_,    dtype=torch.float32)
        self.y0_s  = scaler_y0.scale_[0]
        self.y0_m  = scaler_y0.min_[0]

        self.u10_s = torch.tensor(scaler_u10.scale_, dtype=torch.float32)
        self.u10_m = torch.tensor(scaler_u10.min_,   dtype=torch.float32)
        self.y10_s = scaler_y10.scale_[0]
        self.y10_m = scaler_y10.min_[0]

        self.u1_s  = torch.tensor(scaler_u1.scale_,  dtype=torch.float32)
        self.u1_m  = torch.tensor(scaler_u1.min_,    dtype=torch.float32)
        self.y1_s  = scaler_y1.scale_[0]
        self.y1_m  = scaler_y1.min_[0]

        self.uwaf_s = torch.tensor(scaler_uwaf.scale_, dtype=torch.float32)
        self.uwaf_m = torch.tensor(scaler_uwaf.min_,   dtype=torch.float32)
        self.ywaf_s = scaler_ywaf.scale_[0]
        self.ywaf_m = scaler_ywaf.min_[0]

        # --- Shared-memory numpy<->tensor bufory wejściowe ---
        self._u0_np   = np.zeros((1, 4), dtype=np.float32)
        self._u10_np  = np.zeros((1, 3), dtype=np.float32)
        self._u1_np   = np.zeros((1, 7), dtype=np.float32)
        self._uwaf_np = np.zeros((1, 3), dtype=np.float32)

        self._u0_t   = torch.from_numpy(self._u0_np)
        self._u10_t  = torch.from_numpy(self._u10_np)
        self._u1_t   = torch.from_numpy(self._u1_np)
        self._uwaf_t = torch.from_numpy(self._uwaf_np)

        self._u0_norm   = torch.zeros(1, 4)
        self._u10_norm  = torch.zeros(1, 3)
        self._u1_norm   = torch.zeros(1, 7)
        self._uwaf_norm = torch.zeros(1, 3)

        # ----------------------------------------------------------------
        # Pre-alokacja połączonych buforów [state | u_norm].
        # Eliminuje torch.cat wewnątrz step() — 4 alokacje × każda próbka.
        # Widoki (_x, _u) pozwalają wypełniać fragmenty bez kopiowania.
        # ----------------------------------------------------------------
        self._inp0   = torch.zeros(1, 1 + 4)
        self._inp10  = torch.zeros(1, 1 + 3)
        self._inp1   = torch.zeros(1, 5 + 7)
        self._inpwaf = torch.zeros(1, 1 + 3)

        self._inp0_x   = self._inp0[:,  :1]
        self._inp0_u   = self._inp0[:,  1:]
        self._inp10_x  = self._inp10[:, :1]
        self._inp10_u  = self._inp10[:, 1:]
        self._inp1_x   = self._inp1[:,  :5]
        self._inp1_u   = self._inp1[:,  5:]
        self._inpwaf_x = self._inpwaf[:, :1]
        self._inpwaf_u = self._inpwaf[:, 1:]

        # Stałe T potrzebne do in-place update stanu
        self._T0   = self.model0.T
        self._T10  = self.model10.T
        self._T1   = self.model1.T
        self._Twaf = self.modelwaf.T

        # Bezpośrednie referencje do sieci — eliminuje lookup atrybutu per krok
        self._g0   = self.model0.g_func
        self._h0   = self.model0.h_func
        self._g10  = self.model10.g_func
        self._h10  = self.model10.h_func
        self._g1   = self.model1.g_func
        self._h1   = self.model1.h_func
        self._gwaf = self.modelwaf.g_func
        self._hwaf = self.modelwaf.h_func

        print("Models loaded. Ready to use.")

    # Dekorator zamiast `with torch.inference_mode()` — zero narzutu
    # context managera per wywołanie Input()
    @torch.inference_mode()
    def Input(self, sample):
        if not hasattr(self, '_cols_mapped'):
            cols = sample.columns.tolist()

            self._idx_u0  = np.array([cols.index(c) for c in self.u0_cols])
            self._idx_y0  = cols.index(self.y0_cols[0])
            self._idx_pim = cols.index('intake_manifold_pressure')
            self._idx_pic = cols.index('Intercooler_pressure')
            self._idx_amf = cols.index('air_mass_flow')
            self._idx_thr = cols.index('throttle_position')
            self._idx_eng = cols.index('engine_speed')
            self._idx_y10 = cols.index(self.y10_cols[0])
            self._idx_u1  = np.array([cols.index(c) for c in self.u1_cols])
            self._idx_y1  = cols.index(self.y1_cols[0])
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

        # --- MSO0 ---
        np.copyto(self._u0_np[0], arr[self._idx_u0])
        torch.mul(self._u0_t, self.u0_s, out=self._u0_norm)
        self._u0_norm.add_(self.u0_m)
        # Wypełnij [state | u_norm] bez torch.cat
        self._inp0_x.copy_(self.x0)
        self._inp0_u.copy_(self._u0_norm)
        y0_hat_norm = self._h0(self._inp0)
        # In-place update stanu — eliminuje alokację tensora x_next
        self.x0.add_(self._g0(self._inp0), alpha=self._T0)
        y0_hat = (y0_hat_norm.item() - self.y0_m) / self.y0_s
        e0 = abs(arr[self._idx_y0] - y0_hat)

        # --- MSO10 ---
        pim = arr[self._idx_pim]
        pic = arr[self._idx_pic]
        amf = arr[self._idx_amf]
        thr = arr[self._idx_thr]
        delta_p = math.sqrt(abs(pim - pic))
        self._u10_np[0, 0] = delta_p
        self._u10_np[0, 1] = amf
        self._u10_np[0, 2] = thr
        torch.mul(self._u10_t, self.u10_s, out=self._u10_norm)
        self._u10_norm.add_(self.u10_m)
        self._inp10_x.copy_(self.x10)
        self._inp10_u.copy_(self._u10_norm)
        y10_hat_norm = self._h10(self._inp10)
        self.x10.add_(self._g10(self._inp10), alpha=self._T10)
        y10_hat = (y10_hat_norm.item() - self.y10_m) / self.y10_s
        e10 = abs(arr[self._idx_y10] - y10_hat)

        # --- MSO1 ---
        np.copyto(self._u1_np[0], arr[self._idx_u1])
        torch.mul(self._u1_t, self.u1_s, out=self._u1_norm)
        self._u1_norm.add_(self.u1_m)
        self._inp1_x.copy_(self.x1)
        self._inp1_u.copy_(self._u1_norm)
        y1_hat_norm = self._h1(self._inp1)
        self.x1.add_(self._g1(self._inp1), alpha=self._T1)
        y1_hat = (y1_hat_norm.item() - self.y1_m) / self.y1_s
        e1 = abs(arr[self._idx_y1] - y1_hat)

        # --- WAF ---
        eng_speed = arr[self._idx_eng]
        epsilon = 1e-6
        waf_x1 = math.log(eng_speed + epsilon) * amf
        waf_x3 = math.log(thr + epsilon)
        self._uwaf_np[0, 0] = waf_x1
        self._uwaf_np[0, 1] = amf
        self._uwaf_np[0, 2] = waf_x3
        torch.mul(self._uwaf_t, self.uwaf_s, out=self._uwaf_norm)
        self._uwaf_norm.add_(self.uwaf_m)
        self._inpwaf_x.copy_(self.xwaf)
        self._inpwaf_u.copy_(self._uwaf_norm)
        ywaf_hat_norm = self._hwaf(self._inpwaf)
        self.xwaf.add_(self._gwaf(self._inpwaf), alpha=self._Twaf)
        ywaf_hat = (ywaf_hat_norm.item() - self.ywaf_m) / self.ywaf_s
        ewaf = abs(arr[self._idx_ywaf] - ywaf_hat)

        # --- Warmup / filtracja ---
        self._sample_count += 1
        if self._sample_count <= self.warmup_steps:
            e0 = e10 = e1 = ewaf = 0.0

        self.e0_filt   = 0.001 * e0   + 0.999 * self.e0_filt
        self.e10_filt  = 0.001 * e10  + 0.999 * self.e10_filt
        self.e1_filt   = 0.001 * e1   + 0.999 * self.e1_filt
        self.ewaf_filt = 0.001 * ewaf + 0.999 * self.ewaf_filt

        b0   = 1 if self.e0_filt   > self.th0   else 0
        b10  = 1 if self.e10_filt  > self.th10  else 0
        b1   = 1 if self.e1_filt   > self.th1   else 0
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