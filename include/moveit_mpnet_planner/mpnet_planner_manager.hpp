#ifndef MOVEIT_MPNET_PLANNER_MPNET_PLANNER_MANAGER_HPP
#define MOVEIT_MPNET_PLANNER_MPNET_PLANNER_MANAGER_HPP

#include <string>
#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <moveit/planning_interface/planning_interface.h>
#include <moveit_mpnet_planner/visibility_control.h>
#include <moveit_mpnet_planner/moveit_mpnet_interface.hpp>


namespace moveit_mpnet_planner
{

class MOVEIT_MPNET_PLANNER_PUBLIC MpNetPlannerManager :
    public planning_interface::PlannerManager
{
private:
    using MoveitMpnetInterfacePtr = std::unique_ptr< MoveitMpnetInterface >;
    rclcpp::Node::SharedPtr m_node;
    std::string m_parameterNamespace;
    MoveitMpnetInterfacePtr m_mpNetIf;

public:
    MpNetPlannerManager();

    virtual ~MpNetPlannerManager();

    virtual bool initialize(
        moveit::core::RobotModelConstPtr const& model,
        rclcpp::Node::SharedPtr const& node,
        std::string const& parameterNamespace
    ) override;

    virtual bool canServiceRequest(
        moveit_msgs::msg::MotionPlanRequest const& req
    ) const override;

    virtual std::string getDescription() const override;

    virtual void getPlanningAlgorithms(
        std::vector< std::string >& algsRet
    ) const override;

    virtual planning_interface::PlanningContextPtr
    getPlanningContext(
        planning_scene::PlanningSceneConstPtr const& planningScene,
        planning_interface::MotionPlanRequest const& req,
        moveit_msgs::msg::MoveItErrorCodes& errorCode
    ) const override;
};

} // end namespace moveit_mpnet_planner

#endif /* !MOVEIT_MPNET_PLANNER_MPNET_PLANNER_MANAGER_HPP */

// vim: set ts=4 sw=4 expandtab:
