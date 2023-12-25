#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include<cmath>

using namespace madrona;
using namespace madrona::math;

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10
#define TAU 0.02
#define X_THRESHOLD 2.4

#define MA_PI 3.141592653589793238463

#define THETA_THRESHOLD_RADIANS (12 * 2 * MA_PI / 360)

namespace Cartpole {

    
  void Sim::registerTypes(ECSRegistry &registry, const Config &)
  {
    base::registerTypes(registry);

    // registry.registerSingleton<WorldReset>();

    registry.registerComponent<WorldReset>();
    registry.registerComponent<Action>();
    registry.registerComponent<State>();
    registry.registerComponent<Reward>();

    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    // registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, WorldReset>((uint32_t)ExportID::WorldReset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, State>((uint32_t)ExportID::State);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, WorldID>((uint32_t)ExportID::WorldID);
  }

  static void resetWorld(Engine &ctx)
  {
    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx = episode_mgr.curEpisode.fetch_add_relaxed(1);
    ctx.data().rng = RNG::make(episode_idx);

    const math::Vector2 bounds { -0.05f, 0.05f };
    float bounds_diff = bounds.y - bounds.x;

    Entity agent = ctx.data().agents[0];
    
    ctx.get<State>(agent) = {
      bounds.x + ctx.data().rng.rand() * bounds_diff,
      bounds.x + ctx.data().rng.rand() * bounds_diff,
      bounds.x + ctx.data().rng.rand() * bounds_diff,
      bounds.x + ctx.data().rng.rand() * bounds_diff
    };
  }

  inline void actionSystem(Engine &, Action &action, State &state, Reward &reward)
  {
    float force = (action.choice == 1 ? FORCE_MAG : -FORCE_MAG);
    float costheta = cosf(state.theta);
    float sintheta = sinf(state.theta);

    float temp = (force + POLEMASS_LENGTH * state.theta_dot * state.theta_dot * sintheta) / TOTAL_MASS;
    float thetaacc = (GRAVITY * sintheta - costheta * temp) / (LENGTH * (4.0 / 3.0 - MASSPOLE * costheta * costheta / TOTAL_MASS));
    float xacc = temp - POLEMASS_LENGTH * thetaacc * costheta / TOTAL_MASS;

    state.x = state.x + TAU * state.x_dot;
    state.x_dot = state.x_dot + TAU * xacc;
    state.theta = state.theta + TAU * state.theta_dot;
    state.theta_dot = state.theta_dot + TAU * thetaacc;

    reward.rew = 1.f; // just need to stay alive
  }

  inline void checkDone(Engine &ctx, WorldReset &reset, State &state)
  {
    float x = state.x;
    float theta = state.theta;

    reset.resetNow = x < -X_THRESHOLD || x > X_THRESHOLD || theta < -THETA_THRESHOLD_RADIANS || theta > THETA_THRESHOLD_RADIANS;

    if (reset.resetNow) {
      resetWorld(ctx);
    }
  }

    

  void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
  {
    // auto reset_sys =
    //     builder.addToGraph<ParallelForNode<Engine, resetSystem, WorldReset>>({});

    // auto sort_sys =
    //     builder.addToGraph<SortArchetypeNode<Agent, WorldID>>({reset_sys});

    // auto clear_tmp_alloc =
    //     builder.addToGraph<ResetTmpAllocNode>({sort_sys});
    
    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
							 Action, State, Reward>>({});

    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, checkDone, WorldReset, State>>({action_sys});

    (void)terminate_sys;
    // (void) action_sys;

    // printf("Setup done\n");
  }


  Sim::Sim(Engine &ctx, const Config&, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
  {
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    agents = (Entity *)rawAlloc(sizeof(Entity));

    agents[0] = ctx.makeEntity<Agent>();

    ctx.get<Action>(agents[0]).choice = 0;
    ctx.get<State>(agents[0]).x = 0.f;
    ctx.get<State>(agents[0]).theta = 0.f;
    ctx.get<State>(agents[0]).x_dot = 0.f;
    ctx.get<State>(agents[0]).theta_dot = 0.f;
    ctx.get<Reward>(agents[0]).rew = 0.f;
    
    // Initial reset
    resetWorld(ctx);
    // ctx.getSingleton<WorldReset>().resetNow = false;
    ctx.get<WorldReset>(agents[0]).resetNow = false;
  }

  MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
