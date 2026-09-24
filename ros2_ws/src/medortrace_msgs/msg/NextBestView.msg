# Viewpoint chosen by active perception (medortrace.planning.nbv.ViewGoal). header.frame_id: map.
std_msgs/Header header
Pose2D pose                             # viewpoint (x, y, yaw)
nav_msgs/Path path                      # planned path from the robot to the viewpoint
string target_slot                      # slot the view is for ("" = exploration; "__dock__" = docking)
string probe_region                     # acoustic region to probe from there ("" = none)
float64 score
string[] breakdown_keys                 # score terms (eig, modal, unc, path, turn, risk, sterile, ...)
float64[] breakdown_values
