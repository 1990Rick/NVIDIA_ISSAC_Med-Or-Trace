# One node appended to the tamper-evident provenance hash chain
# (medortrace.provenance.graph.Node). Replaying all events in `index` order and re-hashing
# sha256(json({id, kind, t, attrs, prev})) must reproduce `hash` (see ProvenanceGraph.verify_chain).
string KIND_EVIDENCE="evidence"
string KIND_BELIEF="belief"
string KIND_VERDICT="verdict"
string KIND_ACTIVITY="activity"
string KIND_AGENT="agent"

std_msgs/Header header                  # stamp: node time t (mission time + mission origin)
uint32 index                            # position in the chain (0 = robot agent node)
string node_id                          # e.g. camera:1234, wf:wf_003, verdict:c_wf_003
string kind                             # KIND_* constant
float64 t                               # node time in mission seconds, exactly as hashed
string sensor                           # evidence source (lidar, camera, radar, acoustic, workflow, operator, ...)
string attributed_to                    # prov:Agent (robot id, reporter role, operator)
string content_digest                   # sha256 prefix of the evidence content
string prev_hash
string hash
string json_attrs                       # canonical attrs as hashed (sorted keys)
