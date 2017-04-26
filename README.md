# Orbit-Fitting
Repository for research code under Dr. Kevin Covey

KNOWN ISSUE: current multithreading implementation does NOT work on windows. OSX and Linux work just fine.
  Workaround: Set global variable 'threads' to 1 (this is slower).

functions.py holds all of the functions, i.e. most of the good stuff is in there. master.py is a script which has examples of the entire process of importing data, calling functions.py, and generating results.

Currently there are a lot of other scripts and notebooks which are there to help me develop the code, but are not necessarily useful for one's own implementation.

Tweaks and optimizations are ongoing, typically to functions.py, and master.py
