# Orbit-Fitting
Repository for research code under Dr. Kevin Covey

"RVs come in, Orbital elements go out, you can't explain that."

KNOWN ISSUE: current multithreading implementation does NOT work on windows, OSX and Linux work just fine.
  Set threads to 1 in ensemble sampler call to bypass this (this is slower).

functions.py holds all of the, functions, i.e. most of the good stuff is in there. master.py is a script which has examples of the entire process of importing data, calling functions.py, and generating parameters.

Currently there are a lot of other scripts and notebooks which are there to help me develop the code, but are not necessarily useful for one's own implementation.

Tweaks and optimizations are ongoing, typically to functions.py, and master.py
