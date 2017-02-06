Merging multiple segmentataions of blocks into one.

This repo has two parts

(1) python: given two neighbouring blocks, it produces a .txt file with a list of
            pairs of labels. The pair (l1, l2) means that segment l1 of the first
            block has to be merged with segment l2 of the second block.

(2) cpp: given .txt files from merging many different pairs of blocks in python,
         this code unifies these files using disjoint set data structure,
         and prints the result in a .txt file.
         With a corresponding option, it can relabel data files (SLOW).

