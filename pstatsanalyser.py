import pstats
from pstats import SortKey

# Load the saved statistics
stats = pstats.Stats('serial_profile.prof')

# Sort stats (e.g., by cumulative time) and print the top 20
stats.sort_stats(SortKey.CUMULATIVE).print_stats(20)

# Or sort by total time within the function
# stats.sort_stats(SortKey.TIME).print_stats(20)