# Markov-Chains-Denoising-Images

# Analiza odszumiania danych za pomocą modelu Isinga
Projekt został przygotowany w ramach przedmiotu **Algorytmiczne zastosowanie łańcuchów Markowa**.

Repozytorium bazuje na kodzie dostępnym tutaj:  
**[link](https://github.com/MatthewDaws/Python_bits/blob/master/monte_carlo/Ising%20Model.ipynb)**


## Autorzy:
Krzysztof Tkaczyk, Michał Zajączkowski


## Zawartość repozytorium

- **`ising.py`**  
  Zawiera implementację modelu Isinga z wprowadzonymi przez nas modyfikacjami, wykorzystywaną we wszystkich eksperymentach.

- **`fashion_mnist_denoising.ipynb`**  
  Notebook odpowiedzialny za przygotowanie danych: generowanie wersji zaszumionych i ich odszumianie przy użyciu zmodyfikowanego modelu Isinga. Dane te służą jako podstawa do dalszej analizy.

- **`fashion_mnist_noisiness.ipynb`**  
  Analiza wpływu poziomu zaszumienia na skuteczność odszumiania. Przeprowadzono eksperymenty dla różnych intensywności szumu na zbiorze Fashion MNIST.

- **`fashion_mnist_clumpiness.ipynb`**  
  Analiza wpływu kształtu i "zbitości" (clumpiness) obiektów na jakość odszumiania. Notebook bada zależności pomiędzy strukturą danych a skutecznością modelu.

- **`test_annealing.ipynb`**  
  Notebook prezentujący wyniki testów odszumiania z zastosowaniem **symulowanego wyżarzania (simulated annealing)** jako techniki optymalizacji. Zawiera porównanie wyników dla modelu z tą modyfikacją.

- **`chinese_letters_noisiness.ipynb`**  
Analiza wpływu poziomu zaszumienia oraz kształtu na skuteczność odszumiania z wykorzystaniem zbioru Chinese MNIST.

## Zbiory danych

- Chinese MNIST: https://www.kaggle.com/datasets/gpreda/chinese-mnist
- Fashion MNIST: https://www.kaggle.com/datasets/zalando-research/fashionmnist
