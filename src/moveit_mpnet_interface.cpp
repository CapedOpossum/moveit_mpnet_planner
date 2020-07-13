#include <string>
#include <vector>
#include <functional>
#include <sstream>
#include <moveit/ompl_interface/parameterization/joint_space/joint_model_state_space_factory.h>
#include <moveit/ompl_interface/parameterization/work_space/pose_model_state_space_factory.h>
#include <moveit_mpnet_planner/moveit_mpnet_interface.hpp>
#include <moveit_mpnet_planner/MPNet.h>


using std::string;
using std::vector;
using std::stringstream;
using ompl::base::SpaceInformationPtr;
using ompl::base::PlannerPtr;
using ompl_interface::ConfiguredPlannerSelector;
using ompl_interface::JointModelStateSpaceFactory;
using ompl_interface::PoseModelStateSpaceFactory;
using ompl_interface::ModelBasedPlanningContextPtr;
using ompl_interface::ModelBasedPlanningContext;
using ompl_interface::ModelBasedStateSpaceSpecification;
using ompl_interface::ModelBasedPlanningContextSpecification;
using ompl_interface::ConfiguredPlannerAllocator;
using planning_scene::PlanningSceneConstPtr;
using moveit_msgs::msg::MotionPlanRequest;
using moveit_msgs::msg::MoveItErrorCodes;
using planning_interface::PlannerConfigurationMap;
using constraint_sampler_manager_loader::ConstraintSamplerManagerLoader;
using constraint_samplers::ConstraintSamplerManager;

// CODE ATTRIBUTION NOTICE
// A significant amount of the source code in this file is derived from the
// work provided by the MoveIt 2 developers at:
// https://github.com/ros-planning/moveit2/blob/master/moveit_planners/ompl/ompl_interface/src/ompl_interface.cpp
// END CODE ATTRIBUTION NOTICE

namespace
{

static const rclcpp::Logger LOGGER = rclcpp::get_logger(
    "moveit_mpnet_planner.mpnet_interface"
);

} // end private namespace

namespace moveit_mpnet_planner
{

MoveitMpnetInterface::MoveitMpnetInterface(
    moveit::core::RobotModelConstPtr robotModel,
    PlannerConfigurationMap const& pConfig,
    rclcpp::Node::SharedPtr node,
    std::string const& parameterNamespace
):
    m_node(node),
    m_parameterNamespace(parameterNamespace),
    m_robotModel(robotModel),
    m_pConfig(pConfig),
    m_constraintSamplerManager(new ConstraintSamplerManager())
{
    string modelPath = node->get_parameter(
        parameterNamespace + ".model_path"
    ).get_value< string >();
    vector< string > modelJointNames = m_node->get_parameter(
        m_parameterNamespace + ".model_joint_names"
    ).get_value< vector< string > >();
    stringstream jointNamesStr;
    for (string const& aName : modelJointNames)
    {
        if (jointNamesStr.tellp() != 0)
        {
            jointNamesStr << ",";
        }
        jointNamesStr << aName;
    }
    RCLCPP_INFO(
        LOGGER,
        "Initializing MPNet interface using model file \"%s\" and joint set [%s]",
        modelPath.c_str(),
        jointNamesStr.str().c_str()
    );
    m_constraintSamplerManagerLoader.reset(
        new ConstraintSamplerManagerLoader(m_node, m_constraintSamplerManager)
    );
}

ModelBasedPlanningContextPtr
MoveitMpnetInterface::getPlanningContext(
    const PlanningSceneConstPtr& planning_scene,
    const MotionPlanRequest& req,
    MoveItErrorCodes& error_code,
    const rclcpp::Node::SharedPtr& node,
    bool use_constraints_approximation
) const
{
    if (req.group_name.empty())
    {
        RCLCPP_ERROR(LOGGER, "No group specified to plan for");
        error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
        return ModelBasedPlanningContextPtr();
    }

    error_code.val = moveit_msgs::msg::MoveItErrorCodes::FAILURE;

    if (!planning_scene)
    {
        RCLCPP_ERROR(LOGGER, "No planning scene supplied as input");
        return ModelBasedPlanningContextPtr();
    }

    auto plannerConfigIter = m_pConfig.find(req.group_name);
    if (plannerConfigIter == m_pConfig.end())
    {
        RCLCPP_ERROR(
            LOGGER,
            "Cannot find planning configuration for group '%s'",
            req.group_name.c_str()
        );
        return ModelBasedPlanningContextPtr();
    }

    // Create the new planning context
    ModelBasedStateSpaceSpecification spaceSpec(
        m_robotModel,
        (*plannerConfigIter).second.group
    );
    ModelBasedPlanningContextSpecification contextSpec;
    contextSpec.config_ = (*plannerConfigIter).second.config;
    contextSpec.planner_selector_ = ConfiguredPlannerSelector(
        std::bind(
            &MoveitMpnetInterface::selectPlanner,
            this,
            std::placeholders::_1
        )
    );
    contextSpec.constraint_sampler_manager_ = m_constraintSamplerManager;

    // Select the appropriate state space based on the given problem
    JointModelStateSpaceFactory jmssFac;
    int jmssScore = jmssFac.canRepresentProblem(
        req.group_name,
        req,
        m_robotModel
    );
    PoseModelStateSpaceFactory pmssFac;
    int pmssScore = pmssFac.canRepresentProblem(
        req.group_name,
        req,
        m_robotModel
    );
    if (jmssScore > pmssScore)
    {
        contextSpec.state_space_ = jmssFac.getNewStateSpace(spaceSpec);
    }
    else
    {
        contextSpec.state_space_ = pmssFac.getNewStateSpace(spaceSpec);
    }

    // Choose the correct simple setup type to load
    contextSpec.ompl_simple_setup_.reset(
        new ompl::geometric::SimpleSetup(contextSpec.state_space_)
    );
    RCLCPP_INFO(LOGGER, "Creating new MPNet planning context");

    ModelBasedPlanningContextPtr context;
    context.reset(new ModelBasedPlanningContext((*plannerConfigIter).second.name, contextSpec));
    context->useStateValidityCache(true);
    context->setMaximumPlanningThreads(4);
    context->setMaximumGoalSamples(10);
    context->setMaximumStateSamplingAttempts(4);
    context->setMaximumGoalSamplingAttempts(1000);
    context->setMinimumWaypointCount(2);

    context->setSpecificationConfig((*plannerConfigIter).second.config);

    context->clear();

    robot_state::RobotStatePtr startState = planning_scene->getCurrentStateUpdated(req.start_state);

    // Setup the context
    context->setPlanningScene(planning_scene);
    context->setMotionPlanRequest(req);
    context->setCompleteInitialState(*startState);

    context->setPlanningVolume(req.workspace_parameters);
    if (!context->setPathConstraints(req.path_constraints, &error_code))
      return ModelBasedPlanningContextPtr();

    if (!context->setGoalConstraints(req.goal_constraints, req.path_constraints, &error_code))
      return ModelBasedPlanningContextPtr();

    try
    {
      context->configure(node, use_constraints_approximation);
      RCLCPP_DEBUG(LOGGER, "%s: New planning context is set.", context->getName().c_str());
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
    }
    catch (ompl::Exception& ex)
    {
      RCLCPP_ERROR(LOGGER, "OMPL encountered an error: %s", ex.what());
      context.reset();
    }

    return context;
}

ConfiguredPlannerAllocator
MoveitMpnetInterface::selectPlanner(
    string const& plannerType
) const
{
    if (plannerType != "geometric::MPNet")
    {
        RCLCPP_ERROR(
            LOGGER,
            "Unknown planner \"%s\"",
            plannerType.c_str()
        );
        return ConfiguredPlannerAllocator();
    }

    return ConfiguredPlannerAllocator(
        std::bind(
            &MoveitMpnetInterface::allocatePlanner,
            this,
            std::placeholders::_1,
            std::placeholders::_2,
            std::placeholders::_3
        )
    );
}

PlannerPtr
MoveitMpnetInterface::allocatePlanner(
    SpaceInformationPtr const& si,
    string const& name,
    ModelBasedPlanningContextSpecification const& spec
) const
{
    string modelPath = m_node->get_parameter(
        m_parameterNamespace + ".model_path"
    ).get_value< string >();

    vector< string > modelJointNames = m_node->get_parameter(
        m_parameterNamespace + ".model_joint_names"
    ).get_value< vector< string > >();

    RCLCPP_INFO(
        LOGGER,
        "Allocating MPNet planner with model \"%s\".",
        modelPath.c_str()
    );

    PlannerPtr result(
        new MPNet(
            si,
            modelPath,
            modelJointNames
        )
    );
    if (!name.empty())
    {
        result->setName(name);
    }
    result->params().setParams(spec.config_, true);
    // result->setup();

    return result;
}

}  // namespace moveit_mpnet_planner
