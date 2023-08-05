# wordle-ai

## Overview

Scientific comparison of Wordle solver variants. 

1. RandomSolver: guesses a random word from the remaining possibilities.
2. ParkerSolver: implements Matt Parker's solution--guesses five words that span 25 unique letters, then guesses randomly for sixth guess
3. EntropySolver: guesses word that offers highest entropy
4. CustomSolver: adapts EntropySolver but replaces known (green) letters with highest-frequency unused letters

## Testing

Ran 1,000 trials of randomly selected words from 'wordlists/possible_words.txt' (i.e., the list of all possible Wordle words). Computed and plotted the following statistics: 
* avg: the average number of guesses required to guess a word
* max: the maximum number of guesses required to guess a word
* min: the minimum number of guesses required to guess a word
* std: the standard deviation of the number of guesses required to guess a word

## Results

Statistics are documented in 'results.csv'. Graph of all the trial results for each solver:

![results](https://github.com/DarthBaguette/wordle-ai/assets/46388019/5be8a9ce-b338-44b8-80b3-7809ac360d1f)
