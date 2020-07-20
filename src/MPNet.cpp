/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2011, Rice University
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Rice University nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/* Authors: Alejandro Perez, Sertac Karaman, Ryan Luna, Luis G. Torres, Ioan Sucan, Javier V Gomez, Jonathan Gammell */

/* MPNet Authors: Ahmed Qureshi, Anthony Simeonov */
/* BASED ON OMPL RRT* SOURCE CODE, BEAR WITH ME */

#include <algorithm>
#include <boost/math/constants/constants.hpp>
#include <limits>
#include <vector>
#include <ompl/base/Goal.h>
#include <ompl/base/goals/GoalSampleableRegion.h>
#include <ompl/base/goals/GoalState.h>
#include <ompl/base/objectives/PathLengthOptimizationObjective.h>
#include <ompl/base/samplers/InformedStateSampler.h>
#include <ompl/base/samplers/informed/RejectionInfSampler.h>
#include <ompl/base/samplers/informed/OrderedInfSampler.h>
#include <ompl/tools/config/SelfConfig.h>
#include <ompl/util/GeometricEquations.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/goals/GoalStates.h>
#include <fstream>
#include <sstream>
#include <moveit_mpnet_planner/MPNet.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/ompl_interface/parameterization/model_based_state_space.h>


using std::stringstream;
using std::vector;
using std::string;
using ompl::base::SpaceInformationPtr;
using ompl::base::PlannerStatus;
using ompl::geometric::PathGeometric;
using ompl::unitNBallMeasure;
using ompl::NearestNeighbors;
using robot_state::RobotState;
using ompl_interface::ModelBasedStateSpace;

namespace moveit_mpnet_planner
{

void
MPNet::mlpToMoveIt(
    torch::Tensor mlp_state_tensor,
    ompl::base::State *moveit_state
)
{
    ModelBasedStateSpace const* mbss = si_->getStateSpace()->as< ModelBasedStateSpace >();
    RobotState destStateMoveit(mbss->getRobotModel());
    size_t dim = m_modelJointNames.size();

    auto mlpStateVec = mlp_state_tensor.accessor< float, 2 >();

    for (size_t idx = 0u; idx < dim; idx++)
    {
        destStateMoveit.setVariablePosition(m_modelJointNames[idx], mlpStateVec[0][idx]);
        mbss->copyToOMPLState(moveit_state, destStateMoveit);
    }
}

torch::Tensor
MPNet::getStartGoalTensor(
    ompl::base::State *start_state,
    const ompl::base::State *goal_state,
    bool start_first
)
{
    int dim = m_modelJointNames.size();
    // convert to torch tensor by getting data from states
    std::vector< float > goal_vec;
    std::vector< float > start_vec;

    ModelBasedStateSpace const* mbss = si_->getStateSpace()->as< ModelBasedStateSpace >();
    RobotState startStateMoveit(mbss->getRobotModel());
    RobotState goalStateMoveit(mbss->getRobotModel());
    mbss->copyToRobotState(startStateMoveit, start_state);
    mbss->copyToRobotState(goalStateMoveit, goal_state);

    for (int idx = 0; idx < dim; idx++)
    {
        goal_vec.push_back(
            goalStateMoveit.getVariablePosition(m_modelJointNames[idx])
        );
        start_vec.push_back(
            startStateMoveit.getVariablePosition(m_modelJointNames[idx])
        );
    }

    torch::Tensor start_tensor = torch::from_blob(
        start_vec.data(),
        { 1, dim }
    );
    torch::Tensor goal_tensor = torch::from_blob(
        goal_vec.data(),
        { 1, dim }
    );
    
    torch::Tensor sg_cat;

    if (start_first)
    {
        sg_cat = torch::cat({ start_tensor, goal_tensor }, 1);
    }
    else {
        sg_cat = torch::cat({ goal_tensor, start_tensor }, 1);
    }

    return sg_cat;
}

torch::Tensor
MPNet::getStartStartTensor(
    ompl::base::State *start_state,
    ompl::base::State *goal_state,
    bool start_first
)
{
    int dim = m_modelJointNames.size();
    //convert to torch tensor by getting data from states
    std::vector< float > goal_vec;
    std::vector< float > start_vec;

    ModelBasedStateSpace const* mbss = si_->getStateSpace()->as< ModelBasedStateSpace >();
    RobotState startStateMoveit(mbss->getRobotModel());
    RobotState goalStateMoveit(mbss->getRobotModel());
    mbss->copyToRobotState(startStateMoveit, start_state);
    mbss->copyToRobotState(goalStateMoveit, goal_state);

    for (int idx = 0; idx < dim; idx++)
    {
        goal_vec.push_back(
            goalStateMoveit.getVariablePosition(m_modelJointNames[idx])
        );
        start_vec.push_back(
            startStateMoveit.getVariablePosition(m_modelJointNames[idx])
        );
    }

    torch::Tensor start_tensor = torch::from_blob(
        start_vec.data(),
        { 1, dim }
    );
    torch::Tensor goal_tensor = torch::from_blob(
        goal_vec.data(),
        { 1, dim }
    );
    
    torch::Tensor sg_cat;
    if (start_first)
    {
        sg_cat = torch::cat({ start_tensor, goal_tensor }, 1);
    }
    else
    {
        sg_cat = torch::cat({ goal_tensor, start_tensor }, 1);
    }

    return sg_cat;
}

std::vector< ompl::base::State* >
MPNet::lazyStateContraction(std::vector< ompl::base::State* > path)
{
    for (size_t i = 0; i < (path.size() - 1); i++)
    {
        for (size_t j = (path.size() - 1); j > i + 1; j--)
        {
            bool ind = false;
            ind = si_->checkMotion(path[i], path[j]);

            #ifdef DEBUG
                std::cout << "i: " << i << ", j: " << j << " ind: " << ind << "\n";
                std::cout << "path length: " << path.size() << "\n";
            #endif

            if (ind)
            {

                #ifdef DEBUG
                    std::cout << "calling LSC again... \n";
                #endif

                std::vector< ompl::base::State* > pc;
                for (size_t k = 0; k < i+1; k++)
                {
                    pc.push_back(path[k]);
                }
                for (size_t k = j; k < path.size(); k++)
                {
                    pc.push_back(path[k]);
                }

                return lazyStateContraction(pc);
            }
        }
    }

    return path;
}

bool MPNet::feasibilityCheck(std::vector<ompl::base::State *> path)
{
    for (size_t i = 0; i < (path.size() - 1); i++){
        if (!si_->checkMotion(path[i], path[i+1])){
            return false;
        }
    }

    return true;
}

std::vector<ompl::base::State *>
MPNet::replanPath(
    std::vector< ompl::base::State* > path,
    ompl::base::State *goal,
    torch::jit::Module& MLP
)
{
    size_t dim = m_modelJointNames.size();

    std::vector< ompl::base::State* > new_path;

    std::vector< ompl::base::State* > return_path;
    if (path.size() < 1)
    {
        return return_path;
    }
    for (size_t i = 0; i < path.size() - 1; i++)
    {
        if (si_->isValid(path[i]))
        {
            new_path.push_back(path[i]);
        }
    }

    new_path.push_back(goal);

    for (size_t i = 0; i < new_path.size() - 1; i++)
    {
        if (si_->checkMotion(new_path[i], new_path[i+1]))
        {
            return_path.push_back(new_path[i]);
            return_path.push_back(new_path[i+1]);
        }
        else
        {
            int itr = 0;
            bool target_reached = false;

            torch::Tensor sg_new = getStartStartTensor(
                new_path[i],
                new_path[i+1],
                true
            );
            torch::Tensor goal_tens = sg_new.narrow(1, dim, dim);
            torch::Tensor start_tens = sg_new.narrow(1, 0, dim);

            ompl::base::State *start1_state = si_->allocState();
            ompl::base::State *start2_state = si_->allocState();
            mlpToMoveIt(start_tens, start1_state);
            mlpToMoveIt(goal_tens, start2_state);

            std::vector< ompl::base::State * > pA;
            std::vector< ompl::base::State * > pB;

            pA.push_back(start1_state);
            pB.push_back(start2_state);

            std::vector< torch::jit::IValue > mlp_input_1;
            std::vector< torch::jit::IValue > mlp_input_2;

            torch::Tensor mlp_input_tensor;
            bool tree = 0;

            while (!target_reached && itr < 3000)
            {
                itr = itr + 1;

                if (tree == 0)
                {
                    // concat and fill input
                    mlp_input_tensor = torch::cat(
                        { start_tens, goal_tens },
                        1
                    );
                    mlp_input_1.push_back(mlp_input_tensor);

                    // forward pass and convert to OMPL state
                    start_tens = MLP.forward(mlp_input_1).toTensor();
                    ompl::base::State *new_state_1 = si_->allocState();
                    mlpToMoveIt(start_tens, new_state_1);

                    // append path
                    pA.push_back(new_state_1);

                    // clear input
                    mlp_input_1.clear();

                    // switch to goal
                    tree = 1;
                }
                else
                {
                    // concat and fill input
                    mlp_input_tensor = torch::cat(
                        { goal_tens, start_tens },
                        1
                    );
                    mlp_input_2.push_back(mlp_input_tensor);

                    // forward pass and convert to OMPL state
                    goal_tens = MLP.forward(mlp_input_2).toTensor();
                    ompl::base::State *new_state_2 = si_->allocState();
                    mlpToMoveIt(goal_tens, new_state_2);

                    // append path
                    pB.push_back(new_state_2);

                    // clear input
                    mlp_input_2.clear();

                    // switch to start
                    tree = 0;
                }

                target_reached = si_->checkMotion(pA.back(), pB.back());
            }

            if (!target_reached)
            {
                std::cout << "Failed to replan\n";
            }
            else
            {
                for (size_t i = 0; i < pA.size(); i ++)
                {
                    return_path.push_back(pA[i]);
                }
                for (int j = ((int)(pB.size()) - 1); j > -1; j--)
                {
                    return_path.push_back(pB[j]);
                }
            }
        }    
    }
    return return_path;
}

std::vector< ompl::base::State* >
MPNet::MPNetSolve()
{
    size_t dim = m_modelJointNames.size();

    // get start and goal tensors
    ompl::base::Goal *goal = pdef_->getGoal().get();

    const ompl::base::State *const_goal_state =
        goal->as< ompl::base::GoalStates >()->getState(0);

    ompl::base::State *goal_state = si_->getStateSpace()->cloneState(const_goal_state);
    ompl::base::State *start_state = pdef_->getStartState(0);

    // get start, goal in tensor form
    bool start_first = true;
    torch::Tensor sg = getStartGoalTensor(
        start_state,
        goal_state,
        start_first
    );
    torch::Tensor gs = getStartGoalTensor(
        start_state,
        goal_state,
        !start_first
    );

    torch::Tensor start_only = sg.narrow(1, 0, dim);
    torch::Tensor goal_only = sg.narrow(1, dim, dim);

    torch::Tensor start1 = sg.narrow(1, 0, dim); // path start
    torch::Tensor start2 = sg.narrow(1, dim, dim); // path goal

    torch::Tensor mlp_input_tensor;

    bool target_reached = false;
    int step = 0;
    bool tree = 0;

    float default_step = 0.01;
    float feas_step = 0.01;
    float step_size = default_step;
    si_->setStateValidityCheckingResolution(step_size);

    std::vector< torch::jit::IValue > mlp_input_1;
    std::vector< torch::jit::IValue > mlp_input_2;
    
    torch::jit::Module MLP = torch::jit::load(m_modelFileName);

    std::vector< ompl::base::State* > path1;
    path1.push_back(start_state);

    std::vector< ompl::base::State* > path2;
    path2.push_back(goal_state);

    while (!target_reached && step < 3000)
    {
        step = step + 1;

        if (tree == 0)
        {
            mlp_input_tensor = torch::cat({ start1, start2 }, 1);

            mlp_input_1.push_back(mlp_input_tensor);

            auto mlp_output = MLP.forward(mlp_input_1);

            start1 = mlp_output.toTensor();

            ompl::base::State *new_state_1 = si_->allocState();
            mlpToMoveIt(start1, new_state_1);

            // append path 
            path1.push_back(new_state_1);

            // clear input 
            mlp_input_1.clear();

            // switch to goal 
            tree = 1;
        }
        else
        {
            // concat and fill input 
            mlp_input_tensor = torch::cat({ start2, start1 }, 1);
            mlp_input_2.push_back(mlp_input_tensor);

            // forward pass and convert to OMPL state
            start2 = MLP.forward(mlp_input_2).toTensor();
            ompl::base::State *new_state_2 = si_->allocState();
            mlpToMoveIt(start2, new_state_2);

            // append path 
            path2.push_back(new_state_2);

            // clear input 
            mlp_input_2.clear();

            // switch to start
            tree = 0;
        }
        target_reached = si_->checkMotion(path1.back(), path2.back());
    }

    std::vector< ompl::base::State * > path;
    if (target_reached)
    {
        for (size_t i = 0; i < path1.size(); i++)
        {
            path.push_back(path1[i]);
        }

        for (int j = ((int)(path2.size()) - 1); j > -1; j--)
        {
            path.push_back(path2[j]);
        }
    }
    else
    {
        OMPL_WARN("%s: TARGET NOT REACHED", getName().c_str());
    }

    size_t pathLengths[2] = { path.size(), 0u };
    path = lazyStateContraction(path);
    pathLengths[1] = path.size();

    OMPL_INFORM("%s: Lazy states contraction took plan from (%u) states to (%u).", getName().c_str(), pathLengths[0], pathLengths[1]);

    si_->setStateValidityCheckingResolution(feas_step);

    bool is_feasible = feasibilityCheck(path);

    si_->setStateValidityCheckingResolution(step_size);

    if (is_feasible)
    {
        OMPL_INFORM("%s: MPNet successfully produced plan.", getName().c_str());
        return path;
    }
    else
    {
        int sp = 0;

        while (!is_feasible && sp < 10)
        {
            if (sp == 0)
            {
                step_size = default_step * 0.8;
            }
            else if (sp == 1)
            {
                step_size = default_step * 0.6;
            }
            else if (sp >= 2)
            {
                step_size = default_step * 0.4;
            }

            si_->setStateValidityCheckingResolution(step_size);

            sp = sp + 1;
            path = replanPath(
                path,
                goal_state,
                MLP
            );

            if (path.size() > 0)
            {
                path = lazyStateContraction(path);

                si_->setStateValidityCheckingResolution(feas_step);

                is_feasible = feasibilityCheck(path);

                if (is_feasible)
                {
                    OMPL_INFORM("%s: MPNet re-planning produced plan.", getName().c_str());
                    return path;
                }
            }
        }
        // we failed, return path with only start state 
        path.clear();
        // path.push_back(start_state);
        OMPL_WARN("%s: MPNet failed to produce a plan.", getName().c_str());
        return path; 
    }
}

MPNet::MPNet(
    SpaceInformationPtr const& si,
    string const& modelFileName,
    vector< string > const& modelJointNames
):
    ompl::base::Planner(si, "MPNet"),
    m_modelFileName(modelFileName),
    m_modelJointNames(modelJointNames)
{
    specs_.approximateSolutions = true;
    specs_.optimizingPaths = true;
    specs_.canReportIntermediateSolutions = true;

    Planner::declareParam<double>("range", this, &MPNet::setRange, &MPNet::getRange, "0.:1.:10000.");
    Planner::declareParam<double>("goal_bias", this, &MPNet::setGoalBias, &MPNet::getGoalBias, "0.:.05:1.");
    Planner::declareParam<double>("rewire_factor", this, &MPNet::setRewireFactor, &MPNet::getRewireFactor,
                                  "1.0:0.01:2.0");
    Planner::declareParam<bool>("use_k_nearest", this, &MPNet::setKNearest, &MPNet::getKNearest, "0,1");
    Planner::declareParam<bool>("delay_collision_checking", this, &MPNet::setDelayCC, &MPNet::getDelayCC, "0,1");
    Planner::declareParam<bool>("tree_pruning", this, &MPNet::setTreePruning, &MPNet::getTreePruning, "0,1");
    Planner::declareParam<double>("prune_threshold", this, &MPNet::setPruneThreshold, &MPNet::getPruneThreshold,
                                  "0.:.01:1.");
    Planner::declareParam<bool>("pruned_measure", this, &MPNet::setPrunedMeasure, &MPNet::getPrunedMeasure, "0,1");
    Planner::declareParam<bool>("informed_sampling", this, &MPNet::setInformedSampling, &MPNet::getInformedSampling,
                                "0,1");
    Planner::declareParam<bool>("sample_rejection", this, &MPNet::setSampleRejection, &MPNet::getSampleRejection,
                                "0,1");
    Planner::declareParam<bool>("new_state_rejection", this, &MPNet::setNewStateRejection,
                                &MPNet::getNewStateRejection, "0,1");
    Planner::declareParam<bool>("use_admissible_heuristic", this, &MPNet::setAdmissibleCostToCome,
                                &MPNet::getAdmissibleCostToCome, "0,1");
    Planner::declareParam<bool>("ordered_sampling", this, &MPNet::setOrderedSampling, &MPNet::getOrderedSampling,
                                "0,1");
    Planner::declareParam<unsigned int>("ordering_batch_size", this, &MPNet::setBatchSize, &MPNet::getBatchSize,
                                        "1:100:1000000");
    Planner::declareParam<bool>("focus_search", this, &MPNet::setFocusSearch, &MPNet::getFocusSearch, "0,1");
    Planner::declareParam<unsigned int>("number_sampling_attempts", this, &MPNet::setNumSamplingAttempts,
                                        &MPNet::getNumSamplingAttempts, "10:10:100000");

    addPlannerProgressProperty("iterations INTEGER", [this] { return numIterationsProperty(); });
    addPlannerProgressProperty("best cost REAL", [this] { return bestCostProperty(); });
}

MPNet::~MPNet()
{
    freeMemory();
}

void MPNet::setup()
{
    OMPL_INFORM("%s: There are (%u) joints in the model.", this->getName().c_str(), m_modelJointNames.size());
    Planner::setup();
    ompl::tools::SelfConfig sc(si_, getName());
    sc.configurePlannerRange(maxDistance_);
    if (!si_->getStateSpace()->hasSymmetricDistance() || !si_->getStateSpace()->hasSymmetricInterpolate())
    {
        OMPL_WARN("%s requires a state space with symmetric distance and symmetric interpolation.", getName().c_str());
    }

    if (!nn_)
        nn_.reset(ompl::tools::SelfConfig::getDefaultNearestNeighbors<Motion *>(this));
    nn_->setDistanceFunction([this](const Motion *a, const Motion *b) { return distanceFunction(a, b); });

    // Setup optimization objective
    //
    // If no optimization objective was specified, then default to
    // optimizing path length as computed by the distance() function
    // in the state space.
    if (pdef_)
    {
        if (pdef_->hasOptimizationObjective())
            opt_ = pdef_->getOptimizationObjective();
        else
        {
            OMPL_INFORM("%s: No optimization objective specified. Defaulting to optimizing path length for the allowed "
                        "planning time.",
                        getName().c_str());
            opt_ = std::make_shared<ompl::base::PathLengthOptimizationObjective>(si_);

            // Store the new objective in the problem def'n
            pdef_->setOptimizationObjective(opt_);
        }

        // Set the bestCost_ and prunedCost_ as infinite
        bestCost_ = opt_->infiniteCost();
        prunedCost_ = opt_->infiniteCost();
    }
    else
    {
        OMPL_INFORM("%s: problem definition is not set, deferring setup completion...", getName().c_str());
        setup_ = false;
    }

    // Get the measure of the entire space:
    prunedMeasure_ = si_->getSpaceMeasure();

    // Calculate some constants:
    calculateRewiringLowerBounds();
}

void MPNet::clear()
{
    setup_ = false;
    Planner::clear();
    sampler_.reset();
    infSampler_.reset();
    freeMemory();
    if (nn_)
        nn_->clear();

    bestGoalMotion_ = nullptr;
    goalMotions_.clear();
    startMotions_.clear();

    iterations_ = 0;
    bestCost_ = ompl::base::Cost(std::numeric_limits<double>::quiet_NaN());
    prunedCost_ = ompl::base::Cost(std::numeric_limits<double>::quiet_NaN());
    prunedMeasure_ = 0.0;

}


PlannerStatus
MPNet::solve(const ompl::base::PlannerTerminationCondition &ptc)
{
    this->checkValidity();
    std::vector<ompl::base::State *> mpnet_path;

    auto start = std::chrono::high_resolution_clock::now();

    mpnet_path = MPNetSolve();

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    bool is_feasible = false;
    if (mpnet_path.size() > 0)
    {
        is_feasible = true;
        OMPL_INFORM(
            "%s: MPNet took (%u)ms to find feasible path. "
            "Planned path with (%u) states.",
            getName().c_str(),
            duration.count(),
            mpnet_path.size()
        );
    }

    bool solved = false;
    bool approximate = false; //not sure what to do with this for MPNet

    if (is_feasible)
    {
        /* set the solution path */
        auto solution_path(std::make_shared<PathGeometric>(si_));
        for (size_t i = 0; i < mpnet_path.size(); i++)
        {
            solution_path->append(mpnet_path[i]);
        }
        pdef_->addSolutionPath(solution_path, approximate, 0.0, getName());
        solved = true;
    }

    return PlannerStatus(solved, approximate);
}

void MPNet::getNeighbors(Motion *motion, std::vector<Motion *> &nbh) const
{
    auto cardDbl = static_cast<double>(nn_->size() + 1u);
    if (useKNearest_)
    {
        //- k-nearest RRT*
        unsigned int k = std::ceil(k_rrt_ * log(cardDbl));
        nn_->nearestK(motion, k, nbh);
    }
    else
    {
        double r = std::min(
            maxDistance_, r_rrt_ * std::pow(log(cardDbl) / cardDbl, 1 / static_cast<double>(si_->getStateDimension())));
        nn_->nearestR(motion, r, nbh);
    }
}

void MPNet::removeFromParent(Motion *m)
{
    for (auto it = m->parent->children.begin(); it != m->parent->children.end(); ++it)
    {
        if (*it == m)
        {
            m->parent->children.erase(it);
            break;
        }
    }
}

void MPNet::updateChildCosts(Motion *m)
{
    for (std::size_t i = 0; i < m->children.size(); ++i)
    {
        m->children[i]->cost = opt_->combineCosts(m->cost, m->children[i]->incCost);
        updateChildCosts(m->children[i]);
    }
}

void MPNet::freeMemory()
{
    if (nn_)
    {
        std::vector<Motion *> motions;
        nn_->list(motions);
        for (auto &motion : motions)
        {
            if (motion->state)
                si_->freeState(motion->state);
            delete motion;
        }
    }
}

void MPNet::getPlannerData(ompl::base::PlannerData &data) const
{
    Planner::getPlannerData(data);

    std::vector<Motion *> motions;
    if (nn_)
        nn_->list(motions);

    if (bestGoalMotion_)
        data.addGoalVertex(ompl::base::PlannerDataVertex(bestGoalMotion_->state));

    for (auto &motion : motions)
    {
        if (motion->parent == nullptr)
            data.addStartVertex(ompl::base::PlannerDataVertex(motion->state));
        else
            data.addEdge(ompl::base::PlannerDataVertex(motion->parent->state), ompl::base::PlannerDataVertex(motion->state));
    }
}

int MPNet::pruneTree(const ompl::base::Cost &pruneTreeCost)
{
    // Variable
    // The percent improvement (expressed as a [0,1] fraction) in cost
    double fracBetter;
    // The number pruned
    int numPruned = 0;

    if (opt_->isFinite(prunedCost_))
    {
        fracBetter = std::abs((pruneTreeCost.value() - prunedCost_.value()) / prunedCost_.value());
    }
    else
    {
        fracBetter = 1.0;
    }

    if (fracBetter > pruneThreshold_)
    {
        // We are only pruning motions if they, AND all descendents, have a estimated cost greater than pruneTreeCost
        // The easiest way to do this is to find leaves that should be pruned and ascend up their ancestry until a
        // motion is found that is kept.
        // To avoid making an intermediate copy of the NN structure, we process the tree by descending down from the
        // start(s).
        // In the first pass, all Motions with a cost below pruneTreeCost, or Motion's with children with costs below
        // pruneTreeCost are added to the replacement NN structure,
        // while all other Motions are stored as either a 'leaf' or 'chain' Motion. After all the leaves are
        // disconnected and deleted, we check
        // if any of the the chain Motions are now leaves, and repeat that process until done.
        // This avoids (1) copying the NN structure into an intermediate variable and (2) the use of the expensive
        // NN::remove() method.

        // Variable
        // The queue of Motions to process:
        std::queue<Motion *, std::deque<Motion *>> motionQueue;
        // The list of leaves to prune
        std::queue<Motion *, std::deque<Motion *>> leavesToPrune;
        // The list of chain vertices to recheck after pruning
        std::list<Motion *> chainsToRecheck;

        // Clear the NN structure:
        nn_->clear();

        // Put all the starts into the NN structure and their children into the queue:
        // We do this so that start states are never pruned.
        for (auto &startMotion : startMotions_)
        {
            // Add to the NN
            nn_->add(startMotion);

            // Add their children to the queue:
            addChildrenToList(&motionQueue, startMotion);
        }

        while (motionQueue.empty() == false)
        {
            // Test, can the current motion ever provide a better solution?
            if (keepCondition(motionQueue.front(), pruneTreeCost))
            {
                // Yes it can, so it definitely won't be pruned
                // Add it back into the NN structure
                nn_->add(motionQueue.front());

                // Add it's children to the queue
                addChildrenToList(&motionQueue, motionQueue.front());
            }
            else
            {
                // No it can't, but does it have children?
                if (motionQueue.front()->children.empty() == false)
                {
                    // Yes it does.
                    // We can minimize the number of intermediate chain motions if we check their children
                    // If any of them won't be pruned, then this motion won't either. This intuitively seems
                    // like a nice balance between following the descendents forever.

                    // Variable
                    // Whether the children are definitely to be kept.
                    bool keepAChild = false;

                    // Find if any child is definitely not being pruned.
                    for (unsigned int i = 0u; keepAChild == false && i < motionQueue.front()->children.size(); ++i)
                    {
                        // Test if the child can ever provide a better solution
                        keepAChild = keepCondition(motionQueue.front()->children.at(i), pruneTreeCost);
                    }

                    // Are we *definitely* keeping any of the children?
                    if (keepAChild)
                    {
                        // Yes, we are, so we are not pruning this motion
                        // Add it back into the NN structure.
                        nn_->add(motionQueue.front());
                    }
                    else
                    {
                        // No, we aren't. This doesn't mean we won't though
                        // Move this Motion to the temporary list
                        chainsToRecheck.push_back(motionQueue.front());
                    }

                    // Either way. add it's children to the queue
                    addChildrenToList(&motionQueue, motionQueue.front());
                }
                else
                {
                    // No, so we will be pruning this motion:
                    leavesToPrune.push(motionQueue.front());
                }
            }

            // Pop the iterator, std::list::erase returns the next iterator
            motionQueue.pop();
        }

        // We now have a list of Motions to definitely remove, and a list of Motions to recheck
        // Iteratively check the two lists until there is nothing to to remove
        while (leavesToPrune.empty() == false)
        {
            // First empty the current leaves-to-prune
            while (leavesToPrune.empty() == false)
            {
                // If this leaf is a goal, remove it from the goal set
                if (leavesToPrune.front()->inGoal == true)
                {
                    // Warn if pruning the _best_ goal
                    if (leavesToPrune.front() == bestGoalMotion_)
                    {
                        OMPL_ERROR("%s: Pruning the best goal.", getName().c_str());
                    }
                    // Remove it
                    goalMotions_.erase(std::remove(goalMotions_.begin(), goalMotions_.end(), leavesToPrune.front()),
                                       goalMotions_.end());
                }

                // Remove the leaf from its parent
                removeFromParent(leavesToPrune.front());

                // Erase the actual motion
                // First free the state
                si_->freeState(leavesToPrune.front()->state);

                // then delete the pointer
                delete leavesToPrune.front();

                // And finally remove it from the list, erase returns the next iterator
                leavesToPrune.pop();

                // Update our counter
                ++numPruned;
            }

            // Now, we need to go through the list of chain vertices and see if any are now leaves
            auto mIter = chainsToRecheck.begin();
            while (mIter != chainsToRecheck.end())
            {
                // Is the Motion a leaf?
                if ((*mIter)->children.empty() == true)
                {
                    // It is, add to the removal queue
                    leavesToPrune.push(*mIter);

                    // Remove from this queue, getting the next
                    mIter = chainsToRecheck.erase(mIter);
                }
                else
                {
                    // Is isn't, skip to the next
                    ++mIter;
                }
            }
        }

        // Now finally add back any vertices left in chainsToReheck.
        // These are chain vertices that have descendents that we want to keep
        for (const auto &r : chainsToRecheck)
            // Add the motion back to the NN struct:
            nn_->add(r);

        // All done pruning.
        // Update the cost at which we've pruned:
        prunedCost_ = pruneTreeCost;

        // And if we're using the pruned measure, the measure to which we've pruned
        if (usePrunedMeasure_)
        {
            prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);

            if (useKNearest_ == false)
            {
                calculateRewiringLowerBounds();
            }
        }
        // No else, prunedMeasure_ is the si_ measure by default.
    }

    return numPruned;
}

void MPNet::addChildrenToList(std::queue<Motion *, std::deque<Motion *>> *motionList, Motion *motion)
{
    for (auto &child : motion->children)
    {
        motionList->push(child);
    }
}

bool MPNet::keepCondition(const Motion *motion, const ompl::base::Cost &threshold) const
{
    // We keep if the cost-to-come-heuristic of motion is <= threshold, by checking
    // if !(threshold < heuristic), as if b is not better than a, then a is better than, or equal to, b
    if (bestGoalMotion_ && motion == bestGoalMotion_)
    {
        // If the threshold is the theoretical minimum, the bestGoalMotion_ will sometimes fail the test due to floating point precision. Avoid that.
        return true;
    }

    return !opt_->isCostBetterThan(threshold, solutionHeuristic(motion));
}

ompl::base::Cost MPNet::solutionHeuristic(const Motion *motion) const
{
    ompl::base::Cost costToCome;
    if (useAdmissibleCostToCome_)
    {
        // Start with infinite cost
        costToCome = opt_->infiniteCost();

        // Find the min from each start
        for (auto &startMotion : startMotions_)
        {
            costToCome = opt_->betterCost(
                costToCome, opt_->motionCost(startMotion->state,
                                             motion->state));  // lower-bounding cost from the start to the state
        }
    }
    else
    {
        costToCome = motion->cost;  // current cost from the state to the goal
    }

    const ompl::base::Cost costToGo =
        opt_->costToGo(motion->state, pdef_->getGoal().get());  // lower-bounding cost from the state to the goal
    return opt_->combineCosts(costToCome, costToGo);            // add the two costs
}

void MPNet::setTreePruning(const bool prune)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // If we just disabled tree pruning, but we wee using prunedMeasure, we need to disable that as it required myself
    if (prune == false && getPrunedMeasure() == true)
    {
        setPrunedMeasure(false);
    }

    // Store
    useTreePruning_ = prune;
}

void MPNet::setPrunedMeasure(bool informedMeasure)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // This option only works with informed sampling
    if (informedMeasure == true && (useInformedSampling_ == false || useTreePruning_ == false))
    {
        OMPL_ERROR("%s: InformedMeasure requires InformedSampling and TreePruning.", getName().c_str());
    }

    // Check if we're changed and update parameters if we have:
    if (informedMeasure != usePrunedMeasure_)
    {
        // Store the setting
        usePrunedMeasure_ = informedMeasure;

        // Update the prunedMeasure_ appropriately, if it has been configured.
        if (setup_ == true)
        {
            if (usePrunedMeasure_)
            {
                prunedMeasure_ = infSampler_->getInformedMeasure(prunedCost_);
            }
            else
            {
                prunedMeasure_ = si_->getSpaceMeasure();
            }
        }

        // And either way, update the rewiring radius if necessary
        if (useKNearest_ == false)
        {
            calculateRewiringLowerBounds();
        }
    }
}

void MPNet::setInformedSampling(bool informedSampling)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // This option is mutually exclusive with setSampleRejection, assert that:
    if (informedSampling == true && useRejectionSampling_ == true)
    {
        OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
    }

    // If we just disabled tree pruning, but we are using prunedMeasure, we need to disable that as it required myself
    if (informedSampling == false && getPrunedMeasure() == true)
    {
        setPrunedMeasure(false);
    }

    // Check if we're changing the setting of informed sampling. If we are, we will need to create a new sampler, which
    // we only want to do if one is already allocated.
    if (informedSampling != useInformedSampling_)
    {
        // If we're disabled informedSampling, and prunedMeasure is enabled, we need to disable that
        if (informedSampling == false && usePrunedMeasure_ == true)
        {
            setPrunedMeasure(false);
        }

        // Store the value
        useInformedSampling_ = informedSampling;

        // If we currently have a sampler, we need to make a new one
        if (sampler_ || infSampler_)
        {
            // Reset the samplers
            sampler_.reset();
            infSampler_.reset();

            // Create the sampler
            allocSampler();
        }
    }
}

void MPNet::setSampleRejection(const bool reject)
{
    if (static_cast<bool>(opt_) == true)
    {
        if (opt_->hasCostToGoHeuristic() == false)
        {
            OMPL_INFORM("%s: No cost-to-go heuristic set. Informed techniques will not work well.", getName().c_str());
        }
    }

    // This option is mutually exclusive with setInformedSampling, assert that:
    if (reject == true && useInformedSampling_ == true)
    {
        OMPL_ERROR("%s: InformedSampling and SampleRejection are mutually exclusive options.", getName().c_str());
    }

    // Check if we're changing the setting of rejection sampling. If we are, we will need to create a new sampler, which
    // we only want to do if one is already allocated.
    if (reject != useRejectionSampling_)
    {
        // Store the setting
        useRejectionSampling_ = reject;

        // If we currently have a sampler, we need to make a new one
        if (sampler_ || infSampler_)
        {
            // Reset the samplers
            sampler_.reset();
            infSampler_.reset();

            // Create the sampler
            allocSampler();
        }
    }
}

void MPNet::setOrderedSampling(bool orderSamples)
{
    // Make sure we're using some type of informed sampling
    if (useInformedSampling_ == false && useRejectionSampling_ == false)
    {
        OMPL_ERROR("%s: OrderedSampling requires either informed sampling or rejection sampling.", getName().c_str());
    }

    // Check if we're changing the setting. If we are, we will need to create a new sampler, which we only want to do if
    // one is already allocated.
    if (orderSamples != useOrderedSampling_)
    {
        // Store the setting
        useOrderedSampling_ = orderSamples;

        // If we currently have a sampler, we need to make a new one
        if (sampler_ || infSampler_)
        {
            // Reset the samplers
            sampler_.reset();
            infSampler_.reset();

            // Create the sampler
            allocSampler();
        }
    }
}

void MPNet::allocSampler()
{
    // Allocate the appropriate type of sampler.
    if (useInformedSampling_)
    {
        // We are using informed sampling, this can end-up reverting to rejection sampling in some cases
        OMPL_INFORM("%s: Using informed sampling.", getName().c_str());
        infSampler_ = opt_->allocInformedStateSampler(pdef_, numSampleAttempts_);
    }
    else if (useRejectionSampling_)
    {
        // We are explicitly using rejection sampling.
        OMPL_INFORM("%s: Using rejection sampling.", getName().c_str());
        infSampler_ = std::make_shared<ompl::base::RejectionInfSampler>(pdef_, numSampleAttempts_);
    }
    else
    {
        // We are using a regular sampler
        sampler_ = si_->allocStateSampler();
    }

    // Wrap into a sorted sampler
    if (useOrderedSampling_ == true)
    {
        infSampler_ = std::make_shared<ompl::base::OrderedInfSampler>(infSampler_, batchSize_);
    }
    // No else
}

bool MPNet::sampleUniform(ompl::base::State *statePtr)
{
    // Use the appropriate sampler
    if (useInformedSampling_ || useRejectionSampling_)
    {
        // Attempt the focused sampler and return the result.
        // If bestCost is changing a lot by small amounts, this could
        // be prunedCost_ to reduce the number of times the informed sampling
        // transforms are recalculated.
        return infSampler_->sampleUniform(statePtr, bestCost_);
    }
    else
    {
        // Simply return a state from the regular sampler
        sampler_->sampleUniform(statePtr);

        // Always true
        return true;
    }
}

void MPNet::calculateRewiringLowerBounds()
{
    const auto dimDbl = static_cast<double>(si_->getStateDimension());

    // k_rrt > 2^(d + 1) * e * (1 + 1 / d).  K-nearest RRT*
    k_rrt_ = rewireFactor_ * (std::pow(2, dimDbl + 1) * boost::math::constants::e<double>() * (1.0 + 1.0 / dimDbl));

    // r_rrt > (2*(1+1/d))^(1/d)*(measure/ballvolume)^(1/d)
    // If we're not using the informed measure, prunedMeasure_ will be set to si_->getSpaceMeasure();
    r_rrt_ =
        rewireFactor_ *
        std::pow(2 * (1.0 + 1.0 / dimDbl) * (prunedMeasure_ / unitNBallMeasure(si_->getStateDimension())), 1.0 / dimDbl);
}

} // end namespace moveit_mpnet_planner

// vim: set ts=4 sw=4 expandtab:
