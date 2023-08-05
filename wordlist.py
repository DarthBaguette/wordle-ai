#!/usr/bin/env python3

from random import choice

class WordList():
    """
    The wordlist class is a wrapper for a list of words.
    It is not literally a list, though it has some list like features (see the __dunder__ methods)

    Importantly it provides three key facilities.
    For a given wordlist `words` you can
    - get a random word in the word list by writing words.get_random_word()
    - statefully remove the words inconsistent with the `info` you received from a guess
      by writing `words.refine(info)`
    - return a new literal list of words corresponding to a `pattern` and a `guess` by writing
      `words.matching(pattern, guess)`

    """

    def __init__(self, word_file = "wordlists/possible_words.txt", given_words = None):
        """construct a list of words by reading from `word_file`

        IMPORTANT. IF YOU ARE GETTING AN ERROR SAYING THAT THE ABOVE FILE IS NOT FOUND,
        PLEASE EDIT THE ABOVE STRING TO BE THE CORRECT FILE ON YOUR COMPUTER. 
        PLEASE ASK ERIC OR JUSTIN IF YOU HAVE ANY TROUBLE.
        
        The default wordfile given is `possible_words.txt`. This is the list of actual
        wordle words. `words.txt` is the list of all 5-letter words in the english language.
        `small_words` is a sample of 500 words that is useful for debugging.

        If `given_words` is None, read words from `word_file`, otherwise
        populate `self.words` with `given_words` If no `word_file` parameter is
        given, read from "possible_words.txt"

        """
        if given_words is None:
            self.words = []
            with open(word_file) as fp:
                self.words = fp.readlines()
            self.words = [w.strip() for w in self.words]
        else:
            self.words = given_words

    def get_random_word(self):
        """returns a random word from the set of words"""
        return choice(self.words)

    def __str__(self):
        return str(self.words)

    def __contains__(self, word):
        return word in self.words

    def __iter__(self):
        return self.words.__iter__()

    def __len__(self):
        return len(self.words)

    def refine(self, information):
        """updates the words to be consistent with the `information`"""
        words = []
        for word in self.words:
            if information.matches(word):
                words.append(word)
        self.words = words

    def matching(self, pattern, guess):
        """returns a list of words that could've produced `pattern` in response to `guess`"""
        return [word
                for word in self.words
                if pattern.matches(guess, word)]
