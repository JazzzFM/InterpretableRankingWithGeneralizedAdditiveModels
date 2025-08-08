import os
import matplotlib.pyplot as plt

def plot_partial_effect(outdir, feature, x, y, ci):
    os.makedirs(outdir, exist_ok=True)
    plt.figure()
    plt.plot(x, y, label="Efecto parcial")
    if ci and len(ci) == 2:
        plt.fill_between(x, ci[0], ci[1], alpha=0.2, label="IC 95%")
    plt.title(f"Efecto de {feature}")
    plt.xlabel(feature)
    plt.ylabel("Contribución (log-odds)")
    plt.legend()
    path = os.path.join(outdir, f"{feature}.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path
