# Bumper / manipulator contact (medortrace.common.msgs.ContactState).
std_msgs/Header header
bool in_contact
float32 force_n             # [N] measured (or estimated) contact force
string location             # e.g. "bumper", "gripper"
float64[] effort            # manipulator joint efforts [Nm]; empty when unavailable
