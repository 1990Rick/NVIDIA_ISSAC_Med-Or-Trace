# Safety supervisor state (medortrace.safety.supervisor), published every control tick.
# The velocity on /medortrace/cmd_vel has already been gated by this supervisor.
uint8 NOMINAL=0
uint8 CAUTION=1
uint8 STOP=2
uint8 RETREAT=3
uint8 HANDOVER=4

std_msgs/Header header
uint8 mode
string mode_name
string[] reasons                        # reasons of the latest classification
string category                         # collision_risk | uncertainty | contact | human_proximity | energy | operator | recovery
# monitored signals (see medortrace.safety.supervisor.Envelope for soft/hard limits)
float64 collision_prob                  # MPC predicted collision probability
float64 pred_clearance                  # MPC minimum predicted human clearance [m]
float64 human_clearance                 # tracker-measured human clearance [m]
float64 loc_std                         # localisation position std-dev [m]
float64 nis                             # windowed normalised innovation squared
float64 lidar_age                       # age of the last integrated lidar scan [s]
float64 path_entropy                    # mean map entropy along the path ahead [bits]
float64 contact_force                   # [N]
float64 battery_frac                    # [0, 1]
bool operator_request_open              # a HANDOVER is waiting for /medortrace/operator/ack
uint32 handover_requests                # HANDOVER requests since start
uint32 transitions                      # mode transitions since start
