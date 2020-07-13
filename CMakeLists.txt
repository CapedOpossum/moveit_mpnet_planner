cmake_minimum_required(VERSION 3.5)
project(moveit_mpnet_planner)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_ros REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(ompl REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pluginlib REQUIRED)
find_package(moveit_planners_ompl REQUIRED)
find_package(Torch REQUIRED)

add_library(moveit_mpnet_planner_plugin SHARED
    "src/moveit_mpnet_interface.cpp"
    "src/mpnet_planner_manager.cpp"
    "src/MPNet.cpp"
)

# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(moveit_mpnet_planner_plugin
    PRIVATE
        "MOVEIT_MPNET_PLANNER_BUILDING_LIBRARY"
)

target_include_directories(moveit_mpnet_planner_plugin PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(moveit_mpnet_planner_plugin "torch")

ament_target_dependencies(
    moveit_mpnet_planner_plugin
    OMPL
    "moveit_core"
    "moveit_ros_planning"
    "moveit_planners_ompl"
    "rclcpp"
    "tf2_ros"
    "pluginlib"
)

install(
  DIRECTORY include/
  DESTINATION include
)
install(
  TARGETS moveit_mpnet_planner_plugin
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_export_include_directories(
  "include"
)
ament_export_interfaces(
  export_${PROJECT_NAME}
)
ament_export_libraries(
  moveit_mpnet_planner_plugin
)
ament_export_dependencies(
    moveit_core
    moveit_ros_planning
    moveit_planners_ompl
)

pluginlib_export_plugin_description_file(
    moveit_core moveit_mpnet_planner_plugin_description.xml
)

ament_package()
