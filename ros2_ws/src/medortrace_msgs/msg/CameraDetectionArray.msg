# Camera frame summary (medortrace.common.msgs.CameraFrame).
# header.stamp: acquisition time claimed by the camera clock (may be skewed - the stack's
# TimeSyncMonitor compares it with the receive time); header.frame_id: camera_link.
std_msgs/Header header
float32 fov_h               # horizontal field of view [rad]
float32 max_range           # detection range limit [m]
bool exposure_ok
string[] class_names        # logit order; empty = the stack's default order (sponge, clamp, needle_driver, specimen, implant_box)
CameraDetection[] detections
SurfaceObservation[] surfaces
