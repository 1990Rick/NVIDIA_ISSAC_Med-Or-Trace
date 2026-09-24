# Answer of the claim verifier (medortrace.provenance.verifier.VerdictRecord) with its evidence
# (medortrace.provenance.graph.ProvenanceGraph.explain). header.stamp: decision time.
uint8 VERIFIED=0
uint8 REFUTED=1
uint8 ABSTAIN=2

std_msgs/Header header
uint8 verdict                           # VERIFIED | REFUTED | ABSTAIN (ABSTAIN = ask a human)
string claim_id
string kind                             # location | handoff | count | query
string item_id
string slot_id                          # claimed slot
builtin_interfaces/Time t_ref           # time the claim refers to
float64 posterior                       # smoothed posterior P(item in slot_id at t_ref)
string reason
bool direct_evidence                    # direct sensor evidence about the claimed slot was gathered
string map_slot                         # most probable slot according to the smoothed posterior
string[] supporting_evidence_ids        # provenance node ids (wasDerivedFrom), strongest first
float64[] supporting_weights            # log-likelihood-ratio contributions (>= 0)
string[] contradicting_evidence_ids     # provenance node ids (contradicts), strongest first
float64[] contradicting_weights         # log-likelihood-ratio contributions (< 0)
string provenance_node_id               # "verdict:<claim_id>"
string provenance_hash                  # hash-chain entry of the verdict node (tamper evidence)
