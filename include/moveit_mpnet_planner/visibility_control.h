#ifndef MOVEIT_MPNET_PLANNER__VISIBILITY_CONTROL_H_
#define MOVEIT_MPNET_PLANNER__VISIBILITY_CONTROL_H_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define MOVEIT_MPNET_PLANNER_EXPORT __attribute__ ((dllexport))
    #define MOVEIT_MPNET_PLANNER_IMPORT __attribute__ ((dllimport))
  #else
    #define MOVEIT_MPNET_PLANNER_EXPORT __declspec(dllexport)
    #define MOVEIT_MPNET_PLANNER_IMPORT __declspec(dllimport)
  #endif
  #ifdef MOVEIT_MPNET_PLANNER_BUILDING_LIBRARY
    #define MOVEIT_MPNET_PLANNER_PUBLIC MOVEIT_MPNET_PLANNER_EXPORT
  #else
    #define MOVEIT_MPNET_PLANNER_PUBLIC MOVEIT_MPNET_PLANNER_IMPORT
  #endif
  #define MOVEIT_MPNET_PLANNER_PUBLIC_TYPE MOVEIT_MPNET_PLANNER_PUBLIC
  #define MOVEIT_MPNET_PLANNER_LOCAL
#else
  #define MOVEIT_MPNET_PLANNER_EXPORT __attribute__ ((visibility("default")))
  #define MOVEIT_MPNET_PLANNER_IMPORT
  #if __GNUC__ >= 4
    #define MOVEIT_MPNET_PLANNER_PUBLIC __attribute__ ((visibility("default")))
    #define MOVEIT_MPNET_PLANNER_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define MOVEIT_MPNET_PLANNER_PUBLIC
    #define MOVEIT_MPNET_PLANNER_LOCAL
  #endif
  #define MOVEIT_MPNET_PLANNER_PUBLIC_TYPE
#endif

#endif  // MOVEIT_MPNET_PLANNER__VISIBILITY_CONTROL_H_
