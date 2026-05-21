import numpy as np
import matplotlib.pyplot as plt

e = np.linspace(0, 0.03, 1000)

gamma = 800
e0_a = 0.003
e0_b = 0.010

s_standard_sigmoid = 1 / (1 + np.exp(e))
s_shifted_sigmoid_e0a = 1 / (1 + np.exp(gamma * (e - e0_a)))
s_shifted_sigmoid_e0b = 1 / (1 + np.exp(gamma * (e - e0_b)))
s_exp_decay = np.exp(-69 * e)

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(e, s_standard_sigmoid,   label="Standard sigmoid: $s = 1/(1 + e^{e})$")
ax.plot(e, s_shifted_sigmoid_e0a, label=r"Shifted sigmoid $e_0=0.003$: $s = 1/(1 + e^{\gamma(e - e_0)})$")
ax.plot(e, s_shifted_sigmoid_e0b, label=r"Shifted sigmoid $e_0=0.010$: $s = 1/(1 + e^{\gamma(e - e_0)})$")
ax.plot(e, s_exp_decay,           label=r"Exponential decay: $s = e^{-69e}$")

ax.axvline(x=e0_a, color="gray", linestyle="--", linewidth=1, label=f"$e_0 = {e0_a}$")
ax.axvline(x=e0_b, color="black", linestyle="--", linewidth=1, label=f"$e_0 = {e0_b}$")
ax.axhline(y=0.5,  color="red",   linestyle=":",  linewidth=1, label="$s = 0.5$")

ax.set_xlabel("Temporal geometric error $e$ (Sampson distance)")
ax.set_ylabel("Temporal score $s_{\\mathrm{track}}$")
ax.set_xlim(0, 0.03)
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("temporal_score_curves.png", dpi=150)
plt.show()
