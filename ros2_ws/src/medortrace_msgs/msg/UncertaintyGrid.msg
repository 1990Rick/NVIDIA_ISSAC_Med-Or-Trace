# 2D map uncertainty used by active perception and the supervisor
# (medortrace.belief.occupancy.OccupancyBelief.uncertainty_field): column-occupancy entropy plus
# ghost-suspect (ambiguous) mass, in bits. Layout is that of nav_msgs/OccupancyGrid:
# row-major, x fastest, data[iy * info.width + ix], cell (0, 0) at info.origin.
std_msgs/Header header
nav_msgs/MapMetaData info
string layer                            # "entropy+ambiguous"
string units                            # "bits"
float32 max_value
float32[] data
