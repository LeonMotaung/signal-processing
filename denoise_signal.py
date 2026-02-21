import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt  # alternative: moving average
import pywt  # for wavelet denoising (optional)

# ---------------------------
# 1. Build the dictionary
# ---------------------------
def build_dictionary(N, pattern):
    """
    Create dictionary atoms: each column is the pattern placed at a different start index,
    zero elsewhere. Returns matrix D of shape (N, M) where M = N - len(pattern) + 1.
    """
    L = len(pattern)
    M = N - L + 1
    D = np.zeros((N, M))
    for i in range(M):
        D[i:i+L, i] = pattern
    return D

# ---------------------------
# 2. Generate synthetic clean signal
# ---------------------------
def generate_clean_signal(N, pattern, num_atoms=2, dc_range=(-2,2), amp_range=(0.5,1.5)):
    L = len(pattern)
    M = N - L + 1
    # Randomly select atom positions (without replacement for simplicity)
    positions = np.random.choice(M, size=num_atoms, replace=False)
    amplitudes = np.random.uniform(amp_range[0], amp_range[1], size=num_atoms)
    dc = np.random.uniform(dc_range[0], dc_range[1])
    
    x_clean = np.zeros(N)
    for pos, amp in zip(positions, amplitudes):
        x_clean[pos:pos+L] += amp * pattern
    x_clean += dc
    return x_clean, positions, amplitudes, dc

# ---------------------------
# 3. TCG‑Denoise (IHT)
# ---------------------------
def tcg_denoise(y, D, lambda_thresh, max_iter=100, tol=1e-6):
    N = len(y)
    x = y.copy()
    ones = np.ones(N)
    # normalize dictionary columns to improve numerical stability
    col_norms = np.linalg.norm(D, axis=0)
    safe_norms = col_norms.copy()
    safe_norms[safe_norms == 0] = 1.0
    Dn = D / safe_norms
    # step size (inverse Lipschitz approx) for gradient updates
    L = np.linalg.norm(Dn, ord=2)**2 + 1e-8
    mu = 0.9 / L
    M = D.shape[1]
    alpha = np.zeros(M)
    clip_val = 1e6
    for _ in range(max_iter):
        dc = np.mean(x)
        r = x - dc * ones
        # gradient step for sparse coefficients (IHT-style)
        grad = Dn.T @ (r - Dn @ alpha)
        alpha = alpha + mu * grad
        # hard threshold
        alpha = np.where(np.abs(alpha) > lambda_thresh, alpha, 0)
        # avoid runaway values
        alpha = np.clip(alpha, -clip_val, clip_val)
        r_clean = Dn @ alpha
        x_new = r_clean + dc * ones
        if not np.isfinite(x_new).all():
            break
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new
    return x

# ---------------------------
# 4. Baseline: moving average
# ---------------------------
def moving_average(y, window=6):
    return np.convolve(y, np.ones(window)/window, mode='same')

# ---------------------------
# 5. Baseline: wavelet denoising (requires pywt)
# ---------------------------
def wavelet_denoise(y, wavelet='db4', level=4):
    # soft thresholding with universal threshold
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # robust noise estimate
    threshold = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs_thresh = [coeffs[0]]  # keep approximation coefficients
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    return pywt.waverec(coeffs_thresh, wavelet)

# ---------------------------
# Main experiment
# ---------------------------
np.random.seed(42)  # for reproducibility

# Parameters
N = 60
pattern = np.array([1,4,2,8,5,7])
SNR_dB = 10  # desired signal-to-noise ratio

# Build dictionary
D = build_dictionary(N, pattern)

# Generate clean signal
x_clean, positions, amplitudes, dc = generate_clean_signal(N, pattern, num_atoms=2)

# Add noise to achieve desired SNR
signal_power = np.var(x_clean)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power) * np.random.randn(N)
y = x_clean + noise

# Denoise using TCG
lambda_thresh = 3 * np.sqrt(noise_power)  # 3 * noise std
x_tcg = tcg_denoise(y, D, lambda_thresh)

# Denoise using moving average
x_ma = moving_average(y, window=6)

# Denoise using wavelet (if pywt installed)
try:
    x_wavelet = wavelet_denoise(y)
except NameError:
    x_wavelet = None
    print("PyWavelets not installed; skipping wavelet denoising.")

# Compute MSE improvements
mse_noisy = np.mean((y - x_clean)**2)
mse_tcg = np.mean((x_tcg - x_clean)**2)
mse_ma = np.mean((x_ma - x_clean)**2)
# Use flush=True to ensure immediate stdout and also save results to a file
print(f"MSE (noisy): {mse_noisy:.4f}", flush=True)
print(f"MSE (TCG):   {mse_tcg:.4f}  Improvement: {mse_noisy/mse_tcg:.2f}x", flush=True)
print(f"MSE (MA):    {mse_ma:.4f}  Improvement: {mse_noisy/mse_ma:.2f}x", flush=True)
if x_wavelet is not None:
    mse_wavelet = np.mean((x_wavelet - x_clean)**2)
    print(f"MSE (Wavelet): {mse_wavelet:.4f}  Improvement: {mse_noisy/mse_wavelet:.2f}x", flush=True)

# Save results to a text file for easy inspection
with open('signal_results.txt', 'w') as f:
    f.write(f"MSE (noisy): {mse_noisy:.6f}\n")
    f.write(f"MSE (TCG): {mse_tcg:.6f}  Improvement: {mse_noisy/mse_tcg:.6f}x\n")
    f.write(f"MSE (MA): {mse_ma:.6f}  Improvement: {mse_noisy/mse_ma:.6f}x\n")
    if x_wavelet is not None:
        f.write(f"MSE (Wavelet): {mse_wavelet:.6f}  Improvement: {mse_noisy/mse_wavelet:.6f}x\n")

# Plot results
plt.figure(figsize=(12,6))
plt.plot(x_clean, 'k-', label='Clean')
plt.plot(y, 'c-', alpha=0.5, label='Noisy')
plt.plot(x_tcg, 'r-', label='TCG Denoised')
plt.plot(x_ma, 'b--', label='Moving Average')
if x_wavelet is not None:
    plt.plot(x_wavelet, 'g--', label='Wavelet')
plt.legend()
plt.title(f'Denoising comparison (SNR = {SNR_dB} dB)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
# Save the figure to a file so results are visible without a GUI
plt.savefig('signal_output.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved plot to signal_output.png and results to signal_results.txt', flush=True)
