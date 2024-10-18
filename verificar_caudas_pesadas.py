import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Exemplo de dataset desbalanceado
classe_majoritaria = np.random.normal(0, 1, 1000)  # Classe majoritária (normal)
classe_minoritaria = np.random.pareto(a=2, size=100) + 1  # Classe minoritária (cauda pesada - Pareto)

# Concatenar os dados
dados = np.concatenate([classe_majoritaria, classe_minoritaria])

# Curtose (para caudas pesadas, curtose será > 3)
curtose_majoritaria = stats.kurtosis(classe_majoritaria)
curtose_minoritaria = stats.kurtosis(classe_minoritaria)

print(f"Curtose da classe majoritária: {curtose_majoritaria}")
print(f"Curtose da classe minoritária: {curtose_minoritaria}")

# Q-Q Plot para a classe minoritária
stats.probplot(classe_minoritaria, dist="norm", plot=plt)
plt.title("Q-Q Plot para Classe Minoritária (Pareto)")
plt.show()
