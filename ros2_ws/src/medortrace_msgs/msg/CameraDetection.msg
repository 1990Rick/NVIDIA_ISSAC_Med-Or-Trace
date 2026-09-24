# One detection of the learned camera front end (medortrace.common.msgs.CameraDetection).
# Angles/range are in the camera_link frame (x forward, z up); range comes from aligned depth.
string cls                  # argmax class name (one of CameraDetectionArray.class_names)
string item_id_hint         # decoded 2D tag id when the tag was readable, "" otherwise
float32 bearing             # [rad] positive to the left
float32 elevation           # [rad] positive up
float32 range               # [m]
float32[] logits            # raw (uncalibrated) class logits, ordered as CameraDetectionArray.class_names
float32 visible_fraction    # [0, 1] un-occluded fraction of the object's projected area
float32 glare               # [0, 1] specular highlight score
string gt_item_id           # simulation ground truth for supervision only; "" on the physical robot
