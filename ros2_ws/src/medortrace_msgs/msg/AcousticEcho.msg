# Echo of the active acoustic probe aimed at a named region (medortrace.common.msgs.AcousticEcho).
int8 GT_UNKNOWN=-1
int8 GT_FALSE=0
int8 GT_TRUE=1
string target_region        # region requested on /medortrace/acoustic/probe_target
float32 energy              # normalised echo energy in the range gate around the target
float32 delay_s             # round-trip delay [s]
bool path_occluded          # the direct path was blocked (diffracted / NLOS echo)
int8 gt_hard_reflector      # simulation ground truth (GT_* constants); GT_UNKNOWN on the physical robot
