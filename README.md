This is a simulator of vacuum cleaner robots in grid like environments, 
Some user defined algorithms are available, a reinforcement learning 
agent should available soon.

to install:
	pip install .	

to run
 	python vacuumclean

to run (as a module):
	pyhton -m main

When you simulate a robot over many episodes,
performance results are saved in "data/*.pkl" and plots in "plots/*.png", 
logs in "logs/*.log"


quick help:

	python main.py -v
	==> show available maps
	
	python vacuumclean 20 3 -cl
	==> run QL on map 20 (vacuum-12rooms-v0) with dirt no showing up again
	((re)train and test)

	python -m tools -t vacuum-12rooms-v0
	==> plot QL training perforance



tbc..
