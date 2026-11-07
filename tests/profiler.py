import pstats
import cProfile

import pyximport
pyximport.install()

import examples.run_example as run_example

cProfile.runctx("run_example.example_AJ_finite()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()