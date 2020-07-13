#ifndef MOVEIT_MPNET_PLANNER__MOVEIT_MPNET_INTERFACE_HPP_
#define MOVEIT_MPNET_PLANNER__MOVEIT_MPNET_INTERFACE_HPP_

#include <string>

#include <rclcpp/rclcpp.hpp>

#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit/constraint_samplers/constraint_sampler_manager.h>
#include <moveit/constraint_sampler_manager_loader/constraint_sampler_manager_loader.h>
#include <moveit/ompl_interface/parameterization/model_based_state_space.h>
#include <moveit/ompl_interface/model_based_planning_context.h>
#include <moveit_mpnet_planner/visibility_control.h>

namespace moveit_mpnet_planner
{

class MOVEIT_MPNET_PLANNER_PUBLIC MoveitMpnetInterface
{
private:
    rclcpp::Node::SharedPtr m_node;
    std::string m_parameterNamespace;
    moveit::core::RobotModelConstPtr m_robotModel;
    planning_interface::PlannerConfigurationMap const& m_pConfig;
    constraint_samplers::ConstraintSamplerManagerPtr
        m_constraintSamplerManager;
    constraint_sampler_manager_loader::ConstraintSamplerManagerLoaderPtr
        m_constraintSamplerManagerLoader;

public:
    MoveitMpnetInterface(
        moveit::core::RobotModelConstPtr robotModel,
        planning_interface::PlannerConfigurationMap const& pConfig,
        rclcpp::Node::SharedPtr node,
        std::string const& parameterNamespace
    );

    virtual ~MoveitMpnetInterface() = default;

    ompl_interface::ModelBasedPlanningContextPtr getPlanningContext(
        const planning_scene::PlanningSceneConstPtr& planning_scene,
        const moveit_msgs::msg::MotionPlanRequest& req,
        moveit_msgs::msg::MoveItErrorCodes& error_code,
        const rclcpp::Node::SharedPtr& node,
        bool use_constraints_approximation
    ) const;

    ompl_interface::ConfiguredPlannerAllocator selectPlanner(
        const std::string& plannerType
    ) const;

    ompl::base::PlannerPtr allocatePlanner(
        ompl::base::SpaceInformationPtr const& si,
        std::string const& name,
        ompl_interface::ModelBasedPlanningContextSpecification const& spec
    ) const;
};

}  // namespace moveit_mpnet_planner

#endif  // MOVEIT_MPNET_PLANNER__MOVEIT_MPNET_INTERFACE_HPP_

// vim: set ts=4 sw=4 expandtab:
