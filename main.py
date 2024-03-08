#main.py

from tribe import create_tribes, initialize_tribes
from simulate import simulate

# Create tribes
initial_tribes = create_tribes(4)

# Initialize tribes
initialize_tribes(initial_tribes)

# Simulate
simulate(initial_tribes)
