"""NVIDIA Isaac Sim integration.

Modules
    app            SimulationApp bootstrap + extension enabling (call before any omni import)
    compat         version shims (Isaac Sim 4.5+/5.x ``isaacsim.*`` vs legacy ``omni.isaac.*``,
                   stable vs experimental prim API, stage opening, ROS 2 node namespaces)
    sensors        RTX lidar / camera / radar, acoustic (PhysX echo model) + PhysX IMU/contact,
                   landmark surrogate; numpy-only conversion helpers (grid scan, pinhole rays,
                   gt-surrogate detector, non-visual material token checks)
    robot          differential-drive + retrieval-arm articulation control
    staff          staff agents: kinematic capsules or animated characters
    truth          ground-truth custody placement shared with the lite backend
    replicator_randomizers  domain randomisation with causal locks + causal-label writer
    synthetic      planning of matched-pair Replicator datasets (units, schedule, viewpoints, labels)
    backend        :class:`IsaacBackend` implementing ``medortrace.sim.backend.SimBackend``
    ros2_bridge    OmniGraph ROS 2 bridge (clock, TF, sensors, odometry, optional cmd_vel drive)
    detector       learned detector wrapper (torchvision, lazy import)

Import rules: ``compat``, ``app``, ``truth`` and ``synthetic`` import neither omni nor pxr at module
level; ``replicator_randomizers`` needs only ``pxr`` (usd-core outside Isaac Sim); everything that
needs Kit imports it inside functions, so every module compiles and imports without Isaac Sim.
Inside Isaac Sim, create the ``SimulationApp`` (``app.launch``) before importing modules that use
``pxr`` - there it is provided by Kit.

Everything here is a thin adapter: the autonomy stack, the scene
specification, the workflow and the fault model are shared with the lite
backend (``medortrace.sim``), so a registry seed is the same experiment in
both simulators.  Entry points live in ``scripts/isaac/`` (see its README).
"""
