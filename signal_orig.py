import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt  # alternative: moving average
import pywt  # for wavelet denoising (optional)

# ---------------------------
# 1. Build the dictionary
# ---------------------------
def build_dictionary(N, pattern):
    L = len(pattern)
    M = N - L + 1
    D = np.zeros((N, M))
    for i in range(M):
        D[i:i+L, i] = pattern
    return D

def generate_clean_signal(N, pattern, num_atoms=2, dc_range=(-2,2), amp_range=(0.5,1.5)):
    L = len(pattern)
    M = N - L + 1
    positions = np.random.choice(M, size=num_atoms, replace=False)
    amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=num_atoms)
    dc = np.random.uniform(dc_range[0], dc_range[1])
    x_clean = np.zeros(N)
    for pos, amp in zip(positions, amplitudes):
        x_clean[pos:pos+L] += amp * pattern
    x_clean += dc
    return x_clean, positions, amplitudes, dc

def tcg_denoise(y, D, lambda_thresh, max_iter=100, tol=1e-6):
    N = len(y)
    x = y.copy()
    ones = np.ones(N)
    for _ in range(max_iter):
        dc = np.mean(x)
        r = x - dc * ones
        alpha = D.T @ r
        alpha_new = np.where(np.abs(alpha) > lambda_thresh, alpha, 0)
        r_clean = D @ alpha_new
        x_new = r_clean + dc * ones
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

def moving_average(y, window=6):
    return np.convolve(y, np.ones(window)/window, mode='same')

def wavelet_denoise(y, wavelet='db4', level=4):
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    return pywt.waverec(coeffs_thresh, wavelet)

np.random.seed(42)
N = 60
pattern = np.array([1,4,2,8,5,7])
SNR_dB = 10
D = build_dictionary(N, pattern)
x_clean, positions, amplitudes, dc = generate_clean_signal(N, pattern, num_atoms=2)
signal_power = np.var(x_clean)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power) * np.random.randn(N)
y = x_clean + noise
lambda_thresh = 3 * np.sqrt(noise_power)
x_tcg = tcg_denoise(y, D, lambda_thresh)
x_ma = moving_average(y, window=6)
try:
    x_wavelet = wavelet_denoise(y)
except NameError:
    x_wavelet = None
    print("PyWavelets not installed; skipping wavelet denoising.")
mse_noisy = np.mean((y - x_clean)**2)
mse_tcg = np.mean((x_tcg - x_clean)**2)
mse_ma = np.mean((x_ma - x_clean)**2)
print(f"MSE (noisy): {mse_noisy:.4f}", flush=True)
print(f"MSE (TCG):   {mse_tcg:.4f}  Improvement: {mse_noisy/mse_tcg:.2f}x", flush=True)
print(f"MSE (MA):    {mse_ma:.4f}  Improvement: {mse_noisy/mse_ma:.2f}x", flush=True)
with open('signal_results.txt', 'w') as f:
    f.write(f"MSE (noisy): {mse_noisy:.6f}\n")
    f.write(f"MSE (TCG): {mse_tcg:.6f}  Improvement: {mse_noisy/mse_tcg:.6f}x\n")
    f.write(f"MSE (MA): {mse_ma:.6f}  Improvement: {mse_noisy/mse_ma:.6f}x\n")
plt.figure(figsize=(12,6))
plt.plot(x_clean, 'k-', label='Clean')
plt.plot(y, 'c-', alpha=0.5, label='Noisy')
plt.plot(x_tcg, 'r-', label='TCG Denoised')
plt.plot(x_ma, 'b--', label='Moving Average')
plt.legend()
plt.savefig('signal_output_from_orig.png', dpi=150, bbox_inches='tight')
plt.close()
