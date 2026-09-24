# Event reported by the OR workflow system (OR log, voice transcript, EHR integration).
# Mirrors medortrace.common.msgs.WorkflowEvent. header.stamp is the time the event happened
# (OR clock mapped to ROS time), which may be earlier than the time it was received.
string HANDOFF="handoff"
string PLACE="place"
string DISCARD="discard"
string COUNT="count"
string OPEN="open"
string SPECIMEN_OUT="specimen_out"

string SOURCE_OR_LOG="or_log"
string SOURCE_VOICE="voice"
string SOURCE_SIM="sim"

std_msgs/Header header
string type                 # one of the type constants above
string event_id             # unique per case; the autonomy node de-duplicates on it
string item_id              # "" when the event concerns no single item (e.g. COUNT)
string src                  # slot id, "" if unknown
string dst                  # slot id, "" if unknown
string reporter             # role that reported it (circulating_nurse, scrub_nurse, surgeon, ...)
float32 confidence          # reporter / ASR confidence in [0, 1]
string source               # SOURCE_* constant
string payload_json         # free-form JSON object, e.g. {"phase": "final_count"}
