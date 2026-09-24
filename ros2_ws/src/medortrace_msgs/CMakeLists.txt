cmake_minimum_required(VERSION 3.8)
project(medortrace_msgs)

# MED-OR-TRACE interfaces. Field-by-field mapping to the backend-neutral Python dataclasses
# (medortrace/common/msgs.py) is documented in docs/ros2_message_graph.md; the converters live in
# ros2_ws/src/medortrace_ros/medortrace_ros/convert.py.

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)

set(msg_files
  # --- common -----------------------------------------------------------------
  "msg/Pose2D.msg"
  # --- sensors / perception front-end outputs (inputs of the autonomy stack) --
  "msg/CameraDetection.msg"
  "msg/CameraDetectionArray.msg"
  "msg/SurfaceObservation.msg"
  "msg/RadarDetection.msg"
  "msg/RadarDetectionArray.msg"
  "msg/AcousticEcho.msg"
  "msg/AcousticFrame.msg"
  "msg/LandmarkObservation.msg"
  "msg/LandmarkObservationArray.msg"
  "msg/ContactState.msg"
  "msg/WorkflowEvent.msg"
  # --- autonomy-stack outputs -------------------------------------------------
  "msg/ItemBelief.msg"
  "msg/ItemBeliefArray.msg"
  "msg/ClaimVerdict.msg"
  "msg/ProvenanceEvent.msg"
  "msg/SafetyState.msg"
  "msg/NextBestView.msg"
  "msg/SceneGraph.msg"
  "msg/UncertaintyGrid.msg"
)

set(srv_files
  "srv/OperatorAck.srv"
  "srv/VerifyClaim.srv"
  "srv/ExplainVerdict.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs nav_msgs
)

ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
