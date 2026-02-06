# Phonebook Search with String Matching (MPI-based)

## Problem Statement

This project implements a parallelized string matching solution to search for terms in a phonebook (or any given text file) using MPI (Message Passing Interface). It supports searching both **unordered** and **ordered** strings, with an additional Longest Common Subsequence (LCS) approach for matching. The goal is to find sequences in text that match a given search term.

### Key Features:
- **String Matching with LCS**: Uses the Longest Common Subsequence algorithm to find matches.
- **Parallelization with MPI**: Distributes the search work across multiple processes using MPI to improve performance.
- **Threshold Matching**: Allows you to set a threshold for the minimum length of a match.
- **File Input and Output**: Reads the phonebook (or any text file) and outputs the results to a file.
