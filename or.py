import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

# ---------------------------
# 1. Build normalized dictionary
# ---------------------------
def build_dictionary(N, pattern):
    """
    Create a dictionary of all shifts of the pattern.
    Each column is normalized to unit norm.
    Returns D (N x M) and the original norms.
    """
    L = len(pattern)
    M = N - L + 1
    D = np.zeros((N, M))
    for i in range(M):
        D[i:i+L, i] = pattern
    norms = np.linalg.norm(D, axis=0)
    norms[norms == 0] = 1
    D = D / norms
    return D, norms

# ---------------------------
# 2. Generate clean signal
# ---------------------------
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

# ---------------------------
# 3. TCG‑Denoise (correct IHT)
# ---------------------------
def tcg_denoise(y, D, lambda_thresh, mu=0.2, max_iter=200, tol=1e-6):
    """
    IHT with alternating DC estimation.
    D: normalized dictionary (N x M)
    lambda_thresh: hard threshold for coefficients
    """
    N, M = D.shape
    alpha = np.zeros(M)          # sparse coefficients
    beta = np.mean(y)             # initial DC estimate
    ones = np.ones(N)

    for it in range(max_iter):
        # Current reconstruction
        x = D @ alpha + beta * ones

        # Update DC (coordinate descent)
        beta_new = np.mean(y - D @ alpha)

        # Residual after removing DC
        r = y - beta_new * ones

        # Gradient step for sparse coding (minimize ||r - D alpha||^2)
        grad = D.T @ (r - D @ alpha)   # negative gradient of 0.5*||r-D alpha||^2
        alpha_new = alpha + mu * grad

        # Hard thresholding
        alpha_new = np.where(np.abs(alpha_new) > lambda_thresh, alpha_new, 0)

        # Check convergence
        delta = np.linalg.norm(alpha_new - alpha) + abs(beta_new - beta)
        if delta < tol:
            alpha = alpha_new
            beta = beta_new
            break

        alpha = alpha_new
        beta = beta_new

    # Final reconstruction
    x_denoised = D @ alpha + beta * ones
    return x_denoised

# ---------------------------
# 4. Baselines
# ---------------------------
def moving_average(y, window=6):
    return np.convolve(y, np.ones(window)/window, mode='same')

def wavelet_denoise(y, wavelet='db4', level=3):   # reduced level to avoid boundary warnings
    import pywt
    coeffs = pywt.wavedec(y, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(y)))
    coeffs_thresh = [coeffs[0]]
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    return pywt.waverec(coeffs_thresh, wavelet)

# ---------------------------
# 5. Monte Carlo experiment
# ---------------------------
np.random.seed(42)

# Parameters
N = 60
pattern = np.array([1, 4, 2, 8, 5, 7])
SNR_dB = 10
num_trials = 50   # increase for paper results

# Build dictionary
D, _ = build_dictionary(N, pattern)

# Storage for MSE
mse_noisy_list = []
mse_tcg_list = []
mse_ma_list = []
mse_wavelet_list = []

for trial in range(num_trials):
    # Generate clean signal
    x_clean, _, _, _ = generate_clean_signal(N, pattern, num_atoms=2)

    # Add noise
    signal_power = np.var(x_clean)
    noise_power = signal_power / (10**(SNR_dB/10))
    noise = np.sqrt(noise_power) * np.random.randn(N)
    y = x_clean + noise

    # TCG‑Denoise
    lambda_thresh = 3 * np.sqrt(noise_power)   # universal threshold
    x_tcg = tcg_denoise(y, D, lambda_thresh, mu=0.2)

    # Baselines
    x_ma = moving_average(y, window=6)
    try:
        x_wavelet = wavelet_denoise(y, level=3)   # level 3 to avoid boundary effects
    except Exception as e:
        x_wavelet = None

    # Compute MSE
    mse_noisy = np.mean((y - x_clean)**2)
    mse_tcg = np.mean((x_tcg - x_clean)**2)
    mse_ma = np.mean((x_ma - x_clean)**2)
    if x_wavelet is not None:
        mse_wavelet = np.mean((x_wavelet - x_clean)**2)

    # Store
    mse_noisy_list.append(mse_noisy)
    mse_tcg_list.append(mse_tcg)
    mse_ma_list.append(mse_ma)
    if x_wavelet is not None:
        mse_wavelet_list.append(mse_wavelet)

# Average results
print(f"Over {num_trials} trials at SNR = {SNR_dB} dB:")
print(f"Mean MSE (noisy)  : {np.mean(mse_noisy_list):.4f}")
print(f"Mean MSE (TCG)    : {np.mean(mse_tcg_list):.4f}  Improvement: {np.mean(mse_noisy_list)/np.mean(mse_tcg_list):.2f}x")
print(f"Mean MSE (MA)     : {np.mean(mse_ma_list):.4f}  Improvement: {np.mean(mse_noisy_list)/np.mean(mse_ma_list):.2f}x")
if mse_wavelet_list:
    print(f"Mean MSE (Wavelet): {np.mean(mse_wavelet_list):.4f}  Improvement: {np.mean(mse_noisy_list)/np.mean(mse_wavelet_list):.2f}x")

# Plot one example
trial_idx = 0
x_clean, _, _, _ = generate_clean_signal(N, pattern, num_atoms=2)
signal_power = np.var(x_clean)
noise_power = signal_power / (10**(SNR_dB/10))
noise = np.sqrt(noise_power) * np.random.randn(N)
y = x_clean + noise
x_tcg = tcg_denoise(y, D, 3*np.sqrt(noise_power), mu=0.2)
x_ma = moving_average(y, window=6)
x_wavelet = wavelet_denoise(y, level=3)

plt.figure(figsize=(12,6))
plt.plot(x_clean, 'k-', linewidth=2, label='Clean')
plt.plot(y, 'c-', alpha=0.5, label='Noisy')
plt.plot(x_tcg, 'r-', label='TCG Denoised')
plt.plot(x_ma, 'b--', label='Moving Average')
plt.plot(x_wavelet, 'g--', label='Wavelet')
plt.legend()
plt.title(f'Denoising comparison (SNR = {SNR_dB} dB)')
plt.xlabel('Sample')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()