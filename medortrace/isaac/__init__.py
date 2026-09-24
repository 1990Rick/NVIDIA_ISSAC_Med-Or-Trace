"""NVIDIA Isaac Sim integration (import only inside Isaac Sim's Python).

Modules
    app            SimulationApp bootstrap + extension enabling
    compat         namespace shims (Isaac Sim 4.5+/5.x ``isaacsim.*`` vs legacy ``omni.isaac.*``)
    stage          open a MED-OR-TRACE USD scene, configure PhysX, spawn the rig
    sensors        RTX lidar / camera / radar / acoustic(experimental) + PhysX IMU/contact/effort
    robot          differential-drive + retrieval-arm articulation control
    staff          staff agents: kinematic capsules or animated characters
    replicator_randomizers  domain randomisation with causal locks + causal-label writer
    backend        :class:`IsaacBackend` implementing ``medortrace.sim.backend.SimBackend``
    ros2_bridge    OmniGraph ROS 2 bridge (clock, TF, sensors, cmd_vel)

Everything here is a thin adapter: the autonomy stack, the scene
specification, the workflow and the fault model are shared with the lite
backend (``medortrace.sim``), so a registry seed is the same experiment in
both simulators.
"""
