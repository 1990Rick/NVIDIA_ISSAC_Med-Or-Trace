# All item custody beliefs at one instant. header.frame_id: map.
std_msgs/Header header
string[] slot_ids                       # support of every ItemBelief.probabilities vector
string[] slot_kinds                     # surface | container | under_drape | floor | hand | elsewhere
ItemBelief[] items
