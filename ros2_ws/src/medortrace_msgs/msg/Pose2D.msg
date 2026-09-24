# Planar pose in the frame given by the enclosing message's header (normally "map").
# Kept local to medortrace_msgs instead of the long-deprecated geometry_msgs/Pose2D.
float64 x
float64 y
float64 theta   # yaw [rad], wrapped to (-pi, pi]
