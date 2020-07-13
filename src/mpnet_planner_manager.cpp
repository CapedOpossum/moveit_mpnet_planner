#include <sstream>
#include <moveit_mpnet_planner/mpnet_planner_manager.hpp>
#include <class_loader/class_loader.hpp>

// CODE ATTRIBUTION NOTICE
// Several portions of the code in this file were adapted from the original
// code provided by the MPNet developers at:
// https://github.com/anthonysimeonov/baxter_mpnet_ompl_docker/blob/master/moveit_mpnet/planning_context_manager.cpp
// END CODE ATTRIBUTION NOTICE

using std::string;
using std::vector;
using std::stringstream;
using moveit::core::RobotModelConstPtr;
using rclcpp::Node;
using moveit_msgs::msg::MotionPlanRequest;
using planning_interface::PlannerConfigurationMap;
using planning_scene::PlanningSceneConstPtr;
using planning_interface::MotionPlanRequest;
using planning_interface::PlanningContextPtr;
using moveit_msgs::msg::MoveItErrorCodes;

namespace
{

static const rclcpp::Logger LOGGER = rclcpp::get_logger("moveit_mpnet_planner.MpNetPlannerManager");

} // end private namespace

namespace moveit_mpnet_planner
{

MpNetPlannerManager::MpNetPlannerManager()
{

}


MpNetPlannerManager::~MpNetPlannerManager()
{

}


bool
MpNetPlannerManager::initialize(
    RobotModelConstPtr const& model,
    Node::SharedPtr const& node,
    string const& parameterNamespace
)
{
    m_node = node;
    m_parameterNamespace = parameterNamespace;

    for (auto& groupName : model->getJointModelGroupNames())
    {
        planning_interface::PlannerConfigurationSettings groupConfig;
        groupConfig.group = groupName;
        groupConfig.name = groupName;
        groupConfig.config["type"] = "geometric::MPNet";
        config_settings_[groupName] = groupConfig;
    }

    stringstream groupNames;
    for (const auto& plannerConfig : this->getPlannerConfigurations())
    {
        if (groupNames.tellp() != 0)
        {
            groupNames << ",";
        }
        groupNames << plannerConfig.second.group;
    }

    RCLCPP_INFO(
        LOGGER,
        "Initialized with (%u) total configurations for groups: %s",
        config_settings_.size(),
        groupNames.str().c_str()
    );

    m_mpNetIf.reset(
        new MoveitMpnetInterface(
            model,
            config_settings_,
            node,
            parameterNamespace
        )
    );

    return true;
}


bool
MpNetPlannerManager::canServiceRequest(
    MotionPlanRequest const& req
) const
{
    return req.trajectory_constraints.constraints.empty();
}


std::string
MpNetPlannerManager::getDescription() const
{
    return "Motion Planning Networks";
}


void
MpNetPlannerManager::getPlanningAlgorithms(
    vector< string >& algsRet
) const
{
    algsRet.clear();
    algsRet.reserve(config_settings_.size());
    for(auto& aConfig : config_settings_)
    {
        algsRet.push_back(aConfig.first);
    }
}


PlanningContextPtr
MpNetPlannerManager::getPlanningContext(
    PlanningSceneConstPtr const& planningScene,
    MotionPlanRequest const& req,
    MoveItErrorCodes& errorCode
) const
{
    return m_mpNetIf->getPlanningContext(
        planningScene,
        req,
        errorCode,
        m_node,
        true
    );
}

} // end namespace moveit_mpnet_planner

CLASS_LOADER_REGISTER_CLASS(moveit_mpnet_planner::MpNetPlannerManager, planning_interface::PlannerManager)

// vim: set ts=4 sw=4 expandtab:
