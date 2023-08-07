#!/usr/bin/env python3

from wordle import Player, GameManager
from wordlist import WordList
from information import patterns, Code
from scipy.stats import entropy
from matplotlib import pyplot as plt
import numpy as np
import copy
import csv

# Globals
POSSIBLE_TYPES = ['small_words', 'possible_words', 'words']
WORD_LIST_TYPE = POSSIBLE_TYPES[1]

class Solver(Player):
    def __init__(self):
        """
        Initialize the solver.
        """
        self.num_guesses = 0
        self.word_list = WordList()
        self.info = None

    def make_guess(self):
        """
        the make_guess function makes a guess.
        """
        return "salty"

    def update_knowledge(self, info):
        """
        update_knowledge updates the solver's knowledge with an `info`
        info is an element of the `Information` class. See `information.py`
        """
        self.word_list.refine(info)
        self.info = info
        return info
    
    def __str__(self):
        return "Solver"

class RandomSolver(Solver):
    def make_guess(self):
        """
        Randomly choose a possible word.
        """
        return self.word_list.get_random_word()
    def __str__():
        return "Random Solver"

class ParkerSolver(Solver):
    def __init__(self):
        Solver.__init__(self)
        self.parker_words = ["fjord", "gucks", "nymph", "vibex", "waltz"]

    def make_guess(self):
        """
        Guess five words that hit 25 unique letters.
        """
        if (not self.parker_words) or len(self.word_list) == 1:
            return self.word_list.get_random_word()
        return self.parker_words.pop(0)
    
    def __str__(self):
        return "Parker Solver"

class EntropySolver(Solver):
    def __init__(self):
        Solver.__init__(self)
        self.first_guess = True
        self.first_words = {
            'small_words': 'lares',
            'possible_words': 'raise',
            'common_letters': 'aerot'
        }
        self.word_list_type = WORD_LIST_TYPE
   
    def make_guess(self):
        """
        Guess the word that gives the greatest entropy.
        """
        if self.first_guess:
            self.first_guess = False
            try: return self.first_words[self.word_list_type]
            except: pass
        entropies = {word: self.find_entropy(word) for word in self.word_list}
        return max(entropies, key=entropies.get)

    def find_entropy(self, guess):
        """
        find_entropy returns entropy (bits) of a guess
        """
        # Get list of possible patterns
        p_patterns = patterns()
        possibilities = []; probabilities = []
        for i, p in enumerate(p_patterns):
            # count[i] is list of words that fit could pattern[i] for guess
            possibilities.append(self.word_list.matching(p, guess))
        counts = [len(possibility) for possibility in possibilities]
        probabilities = [count / sum(counts) for count in counts]
        return entropy(probabilities)
    
    def __str__(self):
        return "Parker Solver"

class CustomSolver(EntropySolver):
    def __init__(self):
        EntropySolver.__init__(self)
        self.letter_frequencies = {
            'possible_words': {'a': 0.0844521437851884, 'b': 0.024252923343438718, 'c': 0.0411433521004764, 'd': 0.03404071026418363, 'e': 0.10653962754439152, 'f': 0.019835426591598093, 'g': 0.026851450844521438, 'h': 0.033521004763967084, 'i': 0.058033780857514074, 'j': 0.0023386747509744478, 'k': 0.018189692507579038, 'l': 0.06201818969250758, 'm': 0.027371156344737982, 'n': 0.04963187527067995, 'o': 0.06522304027717626, 'p': 0.03161541792983976, 'q': 0.0025119099177132957, 'r': 0.07769597228237332, 's': 0.05786054569077523, 't': 0.06314421827631009, 'u': 0.04036379385015158, 'v': 0.013165872672152447, 'w': 0.016803811173668255, 'x': 0.0032048505846686876, 'y': 0.036725855348635775, 'z': 0.0034647033347769596}
        }

    def make_guess(self):
        """
        Ignore green letters when guessing next word, if more than 2 possible words left.
            If first guess: return hard-coded word (or just call entropy if not yet determined)
            Elif <= 2 poss. words left: guess randomly.
            Else:
                a) Find highest entropy word
                b) If no green letters: guess the word
                c) Get letter frequencies of word_list
                d) Get highest letter frequency, replace the greens
                e) Update the info and make the guess
        """
        # Check special base cases
        if self.first_guess:
            self.first_guess = False
            return self.first_words['common_letters']
        if len(self.word_list) <= 2:
            return self.word_list.get_random_word()
        
        # Check if entropy guess has any greens
        entropy_guess = EntropySolver.make_guess(self)
        # print(f"Entropy guess: {entropy_guess}")
        greens = self.find_greens()
        if not greens: return entropy_guess

        # Find replacements and add them to the word
        replacements = self.find_replacements(greens, entropy_guess)
        new_guess = list(entropy_guess)
        for i in greens: 
            if not replacements: break
            new_guess[i] = replacements.pop()
        # print(f"New guess: {new_guess}")
        return ''.join(new_guess)

    def find_greens(self):
        """Returns list of indices of the green `hits`."""
        greens = []
        for i, c in enumerate(self.info.pat):
            if c == Code.hit(): greens.append(i)
        # print(f"Greens: {greens}")
        return greens
    
    def find_replacements(self, greens, original_guess):
        letter_freq = self.find_letter_freq(self.word_list)
        replacements = []
        greens_copy = copy.deepcopy(greens)
        while len(greens_copy) > 0:
            # Find most common replacement letter(s). If out of letters, return current replacements
            letter = self.find_top_letter(letter_freq, original_guess)
            if not letter: return replacements
            replacements.append(letter)
            greens_copy.pop(); del letter_freq[letter]
        # print(f"Replacements: {replacements}")
        return replacements

    def find_letter_freq(self, word_list):
        """Calculate frequency of all letters."""
        letter_freq = {}; alphabet = 'abcdefghijklmnopqrstuvwxyz'
        for letter in alphabet:
            letter_freq[letter] = 0
        for word in word_list:
            for letter in word:
                letter_freq[letter] += 1
        return letter_freq
    
    def find_top_letter(self, letter_freq, original_guess):
        top_letter = None; max_freq = 0
        for k in letter_freq:
            if k in original_guess: continue
            if letter_freq[k] > max_freq:
                top_letter = k; max_freq = letter_freq[k]
        # print(f"Top letter: {top_letter}")
        return top_letter

    def __str__(self):
        return "Custom Solver"

class Tester():
    def __init__(self, n_trials):
        self.n_guesses = [] # contains number of guesses in each game
        self.trial_words = [] # specify same trial words
        self.n_trials = n_trials
        self.stats = {}
     
    def run_controlled_trials(self, solver_classes):
        trials = {}; word_list = WordList()
        for _ in range(self.n_trials):
            word = word_list.get_random_word()
            self.trial_words.append(word)
            for _, c in enumerate(solver_classes):
                solver = c()
                manager = GameManager(solver, word)
                n_guess = manager.play_game()
                try: trials[str(c)].append(n_guess)
                except KeyError: trials[str(c)] = [n_guess]
        return trials

    def run_random_trials(self, solver_class):
        for _ in range(self.n_trials):
            solver = solver_class()
            manager = GameManager(solver)
            n_guess = manager.play_game()
            self.n_guesses.append(n_guess)

    def update_stats(self, acc):
        self.stats['avg'] = round(sum(self.n_guesses) / len(self.n_guesses), acc)
        self.stats['max'] = round(max(self.n_guesses), acc)
        self.stats['min'] = round(min(self.n_guesses), acc)
        self.stats['std'] = round(np.std(self.n_guesses), acc)

    def plot(self):
        plt.hist(self.n_guesses, range(1, max(self.n_guesses) + 1))
        plt.title("Frequency vs. Guesses")
        plt.xlabel("Guess")
        plt.ylabel("Frequency")
        plt.show()
    
    def plot_multi(self, trials):
        bins = np.arange(0, 10)
        for solver in trials:
            plt.hist(trials[solver], bins, alpha=0.5, label=str(solver))
        plt.legend(loc='upper right')
        plt.savefig('results.png')
        plt.show()

    def __str__(self):
        stats = f"STATISTICS:"
        for key, value in self.stats.items():
            stats += f"\n{key}: {value}"
        return stats

def test(solver_class, plot=True, n_trials = 100):
    """Individually test performance of a solver class."""
    print("*" * 3, str(solver_class), "*" * 3)
    tester = Tester(n_trials)
    tester.run_random_trials(solver_class)
    tester.update_stats(2) # 2 dec pt accuracy
    print(tester)
    if plot: tester.plot()

def compare(solver_classes, n_trials=100, acc=3, filename='results.csv'):
    """Compare performances of separate solver classes"""
    print("\\" * 3, "SCIENTIFIC TRIALS", "/" * 3)
    tester = Tester(n_trials)
    trials = tester.run_controlled_trials(solver_classes)
    for solver in trials:
        # Print the statistics
        print(solver)
        print(f"""
              avg: {round(sum(trials[solver]) / len(trials[solver]), acc)}
              max: {round(max(trials[solver]), acc)}
              min: {round(min(trials[solver]), acc)}
              std: {round(np.std(trials[solver]), acc)}
              """)
        
        # Write the results to a CSV
        with open(filename, 'w') as csvfile:
            fieldnames = ['SOLVER', 'AVG', 'MAX', 'MIN', 'STD']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for solver in trials:
                writer.writerow({
                    'SOLVER': solver.__str__(),
                    'AVG': round(sum(trials[solver]) / len(trials[solver]), acc),
                    'MAX': round(max(trials[solver]), acc),
                    'MIN': round(min(trials[solver]), acc),
                    'STD': round(np.std(trials[solver]), acc)
                })
    tester.plot_multi(trials)

def main():
    # test(RandomSolver, plot=True, n_trials=1000)
    # test(ParkerSolver, plot=True, n_trials=1000)
    # test(EntropySolver, plot=True, n_trials=100)
    # test(CustomSolver, plot=True, n_trials=30)
    solvers = [RandomSolver, ParkerSolver, EntropySolver, CustomSolver]
    compare(solvers, n_trials=10**3, filename='results.csv')

if __name__ == "__main__": main()

