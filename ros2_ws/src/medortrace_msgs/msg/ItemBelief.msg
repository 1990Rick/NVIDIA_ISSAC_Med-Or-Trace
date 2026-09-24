# Categorical custody belief of one critical item over the slot set
# (medortrace.belief.items.ItemBeliefState). Slot order is given by ItemBeliefArray.slot_ids.
string item_id
string cls
float32 criticality
float32[] probabilities                 # sums to 1, aligned with ItemBeliefArray.slot_ids
string map_slot                         # maximum a-posteriori slot
float32 map_probability
float32 entropy_bits
builtin_interfaces/Time last_seen       # last identity (tag) read; zero = never
