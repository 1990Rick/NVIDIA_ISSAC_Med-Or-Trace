# One 77 GHz radar detection (medortrace.common.msgs.RadarDetection), radar_link frame.
float32 range               # [m]
float32 azimuth             # [rad] positive to the left
float32 elevation           # [rad] positive up
float32 radial_velocity     # Doppler [m/s], positive = receding
float32 rcs_dbsm            # radar cross section [dBsm]
bool through_fabric         # the path crossed a drape / fabric (simulation annotation or driver flag)
string gt_object_id         # simulation ground truth for supervision only; "" on the physical robot
