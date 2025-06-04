import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.special import expit

class IsingGrid:
    def __init__(self, height, width, extfield, invtemp):
        self.width, self.height, self.extfield, self.invtemp = height, width, extfield, invtemp
        self.grid = np.zeros([self.width, self.height], dtype=np.int8) + 1  # initialise grid to all 1s

    def plot(self):
        plt.imshow(self.grid, cmap=cm.gray, aspect="equal", interpolation="none", vmin=-1, vmax=1)

    def make_random(self):
        self.grid = (np.random.randint(0, 2, size=self.width * self.height).reshape(self.width,
                                                                                    self.height) * 2) - 1  # random -1 or 1

    def neighbours(self, x, y):
        """Return the coordinates of the four nearest neighbours of (x,y) in a periodic grid."""
        n = []
        if x == 0:
            n.append((self.width - 1, y))
        else:
            n.append((x - 1, y))
        if x == self.width - 1:
            n.append((0, y))
        else:
            n.append((x + 1, y))
        if y == 0:
            n.append((x, self.height - 1))
        else:
            n.append((x, y - 1))
        if y == self.height - 1:
            n.append((x, 0))
        else:
            n.append((x, y + 1))
        return n

    def local_energy(self, x, y):
        return self.extfield + sum(self.grid[xx, yy] for (xx, yy) in self.neighbours(x,
                                                                                     y))  # local energy of a single spin, extfield is H from the description above

    def total_energy(self):
        # Could maybe do some numpy games here, but periodic boundary conditions make this tricky.
        # This function is only ever useful for very small grids anyway.
        energy = - self.extfield * np.sum(self.grid)  # external field energy
        energy += - sum(self.grid[x, y] * sum(self.grid[xx, yy] for (xx, yy) in self.neighbours(x, y))
                        for x in range(self.width) for y in
                        range(self.height)) / 2  # energy from each pair of spins | lack of J parameter, 1 or 0
        return energy

    def total_energy_np(self):
        # External field energy (unchanged)
        energy = -self.extfield * np.sum(self.grid)

        # Periodic boundary conditions handled via np.roll
        energy += -np.sum(self.grid * np.roll(self.grid, 1, axis=0))  # Vertical interactions
        energy += -np.sum(self.grid * np.roll(self.grid, 1, axis=1))  # Horizontal interactions
        return energy

    def probability(self):
        return np.exp(- self.invtemp * self.total_energy_np())

    def gibbs_move(self, beta=None):
        """Wykonaj jeden ruch Gibbsa z opcjonalnym parametrem beta"""
        if beta is None:
            beta = self.invtemp
            
        n = np.random.randint(0, self.width * self.height)
        y = n // self.width
        x = n % self.width
        # p = 1 / (1 + np.exp(-2 * beta * self.local_energy(x,y)))
        p = expit(2 * beta * self.local_energy(x, y))  # bez ryzyka overflow
        if np.random.random() <= p:
            self.grid[x, y] = 1
        else:
            self.grid[x, y] = -1

    def from_number(self, n):
        """Convert an integer 0 <= n < 2**(width*height) into a grid."""
        N = self.width * self.height
        binstring = bin(n)[2:]
        binstring = "0" * (N - len(binstring)) + binstring
        self.grid = np.array([int(x) * 2 - 1 for x in binstring], dtype=np.int8).reshape(self.width, self.height)

    def to_number(self):
        """Convert grid into an integer."""
        flat = [self.grid[x, y] for x in range(self.width) for y in range(self.height)]
        return sum(2 ** n * (int(x) + 1) // 2 for n, x in enumerate(reversed(flat)))


class IsingGridVaryingField(IsingGrid):
    def __init__(self, height, width, extfield, invtemp):
        super().__init__(height, width, 0, invtemp)
        self.vextfield = extfield

    def local_energy(self, x, y):
        return self.vextfield[x, y] + sum(self.grid[xx, yy] for (xx, yy) in self.neighbours(x, y))

def get_annealing_schedule(beta_start, beta_end, total_steps, schedule_type='linear'):
    """
    Tworzy plan wyżarzania parametru beta.
    
    Parameters:
    - beta_start: początkowa wartość beta (niska temperatura = wysokie beta)
    - beta_end: końcowa wartość beta 
    - total_steps: całkowita liczba kroków
    - schedule_type: typ planu ('linear', 'exponential', 'power', 'cosine')
    
    Returns:
    - array z wartościami beta dla każdego kroku
    """
    steps = np.arange(total_steps)
    
    if schedule_type == 'linear':
        # Liniowy plan wyżarzania
        return beta_start + (beta_end - beta_start) * steps / (total_steps - 1)
    
    elif schedule_type == 'exponential':
        # Wykładniczy plan wyżarzania
        ratio = beta_end / beta_start
        return beta_start * (ratio ** (steps / (total_steps - 1)))
    
    elif schedule_type == 'power':
        # Plan potęgowy (powolne na początku, szybkie na końcu)
        alpha = 2.0  # można dostosować
        progress = steps / (total_steps - 1)
        return beta_start + (beta_end - beta_start) * (progress ** alpha)
    
    elif schedule_type == 'cosine':
        # Plan cosinusowy (płynne przejście)
        progress = steps / (total_steps - 1)
        return beta_start + (beta_end - beta_start) * (1 - np.cos(progress * np.pi)) / 2
    
    else:
        raise ValueError(f"Nieznany typ planu: {schedule_type}")

def IsingDeNoise(noisy, q, burnin=50000, loops=500000, 
                 use_annealing=False, beta_schedule='exponential'):
    """
    Denoisuje obraz używając modelu Isinga z opcjonalnym planem wyżarzania.
    
    Parameters:
    - noisy: zaszumiony obraz
    - q: prawdopodobieństwo poprawności piksela
    - burnin: liczba kroków burn-in
    - loops: liczba kroków próbkowania
    - use_annealing: czy używać planu wyżarzania
    - beta_schedule: typ planu wyżarzania
    """
    h = 0.5 * np.log(q / (1 - q))
    gg = IsingGridVaryingField(noisy.shape[0], noisy.shape[1], h * noisy, 2)
    gg.grid = np.array(noisy)

    if use_annealing:
        # Plan wyżarzania: zaczynamy od niskiej temperatury (wysokie beta)
        # i przechodzimy do docelowej temperatury
        beta_start = 0.1  # niska temperatura na początku
        beta_end = 2.0    # docelowa temperatura
        
        # Plan dla burn-in
        burnin_schedule = get_annealing_schedule(beta_start, beta_end, burnin, beta_schedule)
        
        # Burn-in z planem wyżarzania
        print(f"Burn-in z planem wyżarzania ({beta_schedule}): β = {beta_start:.2f} → {beta_end:.2f}")
        for i in range(burnin):
            gg.gibbs_move(beta=burnin_schedule[i])
    else:
        # Standardowy burn-in bez planu wyżarzania
        print("Burn-in bez planu wyżarzania")
        for _ in range(burnin):
            gg.gibbs_move()

    # Sample - tutaj używamy stałej temperatury (docelowej)
    print("Próbkowanie z stałą temperaturą")
    avg = np.zeros_like(noisy).astype(np.float64)
    for _ in range(loops):
        gg.gibbs_move(beta=2.0 if use_annealing else None)
        avg += gg.grid
    
    return avg / loops
