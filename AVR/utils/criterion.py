import torch
import torch.nn as nn
import auraloss
import torch.nn.functional as F
import math

class Criterion(nn.Module):
    def __init__(self, cfg, cfg_render):
        super().__init__()

        self.spec_loss_weight = cfg['spec_loss_weight']
        self.amplitude_loss_weight = cfg['amplitude_loss_weight']
        self.angle_loss_weight = cfg['angle_loss_weight']
        self.time_loss_weight = cfg['time_loss_weight']
        self.energy_loss_weight = cfg['energy_loss_weight']
        self.multi_stft_weight = cfg['multistft_loss_weight']

        # DAS用パラメータ
        self.das_loss_weight = cfg.get('das_loss_weight', 0.0)
        self.beta = cfg.get('beta', 100.0)
        self.n_fft = cfg.get('das_n_fft', 512)

        # 360度分の角度 (ラジアン)
        self.angles_rad = torch.deg2rad(torch.arange(0.0, 360.0, 1.0))
        self.K = len(self.angles_rad)

        # rendering config
        self.fs = cfg_render['fs']
        self.sound_speed = cfg_render['speed']

        self.mse_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        self.mrft_loss = auraloss.freq.MultiResolutionSTFTLoss(w_lin_mag=1, fft_sizes=[512, 256, 128, 64], win_lengths=[300, 150, 75, 30], hop_sizes=[60, 30, 8, 4])
    
    def compute_beamforming_power(self, sig, rx_positions):
        """
        sig: 複素数 (B, M, F) or (M, F) or (B, F)
        rx_positions: (B, M, 2/3) or (M, 2/3) or (B, 2/3)
        """
        if rx_positions is None:
            raise ValueError("rx_positions is required for DAS loss")

        if sig.ndim == 2 and rx_positions.ndim == 2 and rx_positions.shape[0] == sig.shape[0] and rx_positions.shape[1] in (2, 3):
            sig = sig.unsqueeze(1)
            rx_positions = rx_positions.unsqueeze(1)
        else:
            if sig.ndim == 2:
                sig = sig.unsqueeze(0)
            if rx_positions.ndim == 2:
                rx_positions = rx_positions.unsqueeze(0)
        if rx_positions.ndim == 1:
            rx_positions = rx_positions.view(1, 1, -1)

        if sig.ndim == 3 and sig.shape[1] != rx_positions.shape[1]:
            if sig.shape[0] == rx_positions.shape[0] and rx_positions.shape[1] in (2, 3):
                sig = sig.unsqueeze(1)
            else:
                raise ValueError("sig and rx_positions shape mismatch for DAS")

        B, M, _ = sig.shape
        time_sig = torch.real(torch.fft.irfft(sig, dim=-1))
        n_fft = int(self.n_fft)

        freqs = torch.fft.rfftfreq(n_fft, 1 / self.fs).to(sig.device)
        X = torch.fft.rfft(time_sig, n=n_fft, dim=-1)  # (B, M, F)
        F = X.shape[-1]

        rx_xy = rx_positions[..., :2]
        rx_xy = rx_xy - rx_xy.mean(dim=1, keepdim=True)

        power_all = []
        for b in range(B):
            mic_pos = rx_xy[b]
            steering = torch.zeros(self.K, M, F, dtype=torch.cfloat, device=sig.device)
            for i, theta in enumerate(self.angles_rad.to(sig.device)):
                u = torch.tensor([torch.cos(theta), torch.sin(theta)], device=sig.device)
                delays = (mic_pos @ u) / self.sound_speed
                phase_shift = torch.exp(-1j * 2 * math.pi * delays[:, None] * freqs[None, :])
                steering[i] = phase_shift

            beam = torch.einsum('mf,kmf->kf', X[b], steering) / M
            beam_power = torch.abs(beam) ** 2
            beam_power_norm = beam_power / (torch.sum(beam_power, dim=0, keepdim=True) + 1e-8)
            power = torch.sum(beam_power_norm, dim=-1)  # (K,)
            power_all.append(power)
        return torch.stack(power_all, dim=0)

    def forward(self, pred_sig, ori_sig, rx_positions=None):

        pred_time = torch.real(torch.fft.irfft(pred_sig, dim=-1)) 
        ori_time = torch.real(torch.fft.irfft(ori_sig, dim=-1))

        pred_spec = torch.abs(torch.stft(pred_time, n_fft=256, return_complex=True))
        ori_spec = torch.abs(torch.stft(ori_time, n_fft=256, return_complex=True))

        pred_spec_energy = torch.sum(pred_spec ** 2, dim=1)
        ori_spec_energy = torch.sum(ori_spec ** 2, dim=1)

        predict_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(pred_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        predict_energy -= predict_energy[:,[0]]
        ori_energy = torch.log10(torch.flip(torch.cumsum(torch.flip(ori_spec_energy, [-1])**2, dim=-1), [-1]) + 1e-9)
        ori_energy -= ori_energy[:,[0]]

        real_loss = self.l1_loss(torch.real(pred_sig), torch.real(ori_sig))
        imag_loss  = self.l1_loss(torch.imag(pred_sig), torch.imag(ori_sig)) 
        spec_loss = (real_loss + imag_loss) * self.spec_loss_weight

        amplitude_loss = self.l1_loss(torch.abs(pred_sig), torch.abs(ori_sig)) * self.amplitude_loss_weight 

        angle_loss = (self.l1_loss(torch.cos(torch.angle(pred_sig)), torch.cos(torch.angle(ori_sig))) + \
                    self.l1_loss(torch.sin(torch.angle(pred_sig)), torch.sin(torch.angle(ori_sig)))) * self.angle_loss_weight
        
        time_loss = self.l1_loss(ori_time, pred_time) * self.time_loss_weight

        energy_loss = self.l1_loss(ori_energy, predict_energy) * self.energy_loss_weight

        multi_stft_loss = self.mrft_loss(ori_time.unsqueeze(1), pred_time.unsqueeze(1)) * self.multi_stft_weight

        # === DAS 損失 ===
        das_loss = torch.tensor(0.0, device=pred_sig.device)

        if self.das_loss_weight > 0:
            power_pred = self.compute_beamforming_power(pred_sig, rx_positions)  # (B, K)
            power_ori = self.compute_beamforming_power(ori_sig, rx_positions)    # (B, K)

            weights_pred = torch.softmax(self.beta * power_pred, dim=-1)
            weights_ori = torch.softmax(self.beta * power_ori, dim=-1)

            pred_angle_rad = torch.sum(weights_pred * self.angles_rad.to(pred_sig.device), dim=-1)
            true_angle_rad = torch.sum(weights_ori * self.angles_rad.to(pred_sig.device), dim=-1)

            das_loss_raw = self.l1_loss(torch.sin(pred_angle_rad), torch.sin(true_angle_rad)) + \
                self.l1_loss(torch.cos(pred_angle_rad), torch.cos(true_angle_rad))
            das_loss = das_loss_raw * self.das_loss_weight

        return (spec_loss, amplitude_loss, angle_loss, time_loss,
                energy_loss, multi_stft_loss, das_loss,
                ori_time, pred_time)
