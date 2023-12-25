#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include<cmath>

using namespace madrona;
using namespace madrona::math;

#define BUFFER 2
#define NUM_SPACES 5
#define NUM_MOVES 4
#define TIME 3
#define SCALE 0.2

namespace Balance {

    
  void Sim::registerTypes(ECSRegistry &registry, const Config &)
  {
    base::registerTypes(registry);

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<WorldTime>();
    
    registry.registerComponent<Action>();
    registry.registerComponent<Observation>();
    registry.registerComponent<Location>();
    registry.registerComponent<AgentID>();
    registry.registerComponent<ActionMask>();
    registry.registerComponent<ActiveAgent>();
    registry.registerComponent<Reward>();

    registry.registerArchetype<Agent>();

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>((uint32_t)ExportID::WorldReset);
    registry.exportColumn<Agent, ActiveAgent>((uint32_t)ExportID::ActiveAgent);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Observation>((uint32_t)ExportID::Observation);
    registry.exportColumn<Agent, ActionMask>((uint32_t)ExportID::ActionMask);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, WorldID>((uint32_t)ExportID::WorldID);
    registry.exportColumn<Agent, AgentID>((uint32_t)ExportID::AgentID);
  }

  static void resetWorld(Engine &ctx)
  {
    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx = episode_mgr.curEpisode.fetch_add_relaxed(1);

    ctx.data().rng = RNG::make(episode_idx);

    ctx.singleton<WorldTime>().time = TIME - 1;

    for (int i = 0; i < 2; i++) {
      Entity agent = ctx.data().agents[i];
    
      ctx.get<Location>(agent) = {
	static_cast<int32_t>(NUM_SPACES * ctx.data().rng.rand())
      };

      for (int t = 0; t < 2 * TIME; t++) {
	ctx.get<Observation>(agent).x[t] = 0;
      }
      ctx.get<Observation>(agent).time = ctx.singleton<WorldTime>().time;
    }

    int32_t locs[2] = {
      ctx.get<Location>(ctx.data().agents[0]).x,
      ctx.get<Location>(ctx.data().agents[1]).x
    };

    for (int i = 0; i < 2; i++) {

      Entity agent = ctx.data().agents[i];

      ctx.get<Observation>(agent).x[0] = locs[i] + BUFFER;
      ctx.get<Observation>(agent).x[TIME] = locs[1-i] + BUFFER;
    }
  }

  inline void actionSystem(Engine &, Action &action, Location &loc)
  {
    switch (action.choice) {
    case 0:
      loc.x += -2;
      break;
    case 1:
      loc.x += -1;
      break;
    case 2:
      loc.x += 1;
      break;
    case 3:
      loc.x += 2;
      break;
    }
  }

  inline void timeSystem(Engine &, WorldTime &worldTime)
  {
    worldTime.time -= 1;
  }

  inline void observationSystem(Engine &ctx, Location &loc, AgentID &id, Observation &obs)
  {
    int32_t time = ctx.singleton<WorldTime>().time;
    int32_t loc1 = ctx.get<Location>(ctx.data().agents[1 - id.id]).x;

    for (int i = TIME * 2; i > 0; i--) {
      obs.x[i] = obs.x[i-1];
    }

    obs.x[TIME] = loc1 + BUFFER;
    obs.x[0] = loc.x + BUFFER;
    obs.time = time;
  }
    
  inline void checkDone(Engine &ctx, WorldReset &reset)
  {
    WorldTime &worldTime = ctx.singleton<WorldTime>();
    reset.resetNow = false;
    int32_t locs[2] = {
      ctx.get<Location>(ctx.data().agents[0]).x,
      ctx.get<Location>(ctx.data().agents[1]).x
    };
    // int32_t loc1 = ctx.get<Location>(ctx.data().agents[1]).x;

    float reward = (locs[0] == locs[1] ? 1.0 : -abs(locs[0] - locs[1]) * SCALE);

    for (int i = 0; i < 2; i++) {
      int32_t x = locs[i];

      if (x < 0 || x >= NUM_SPACES) {
	reset.resetNow = true;
	reward = -NUM_SPACES * (worldTime.time + 1) * SCALE;
      }
    }

    for (int i = 0; i < 2; i++) {
      Entity agent = ctx.data().agents[i];

      ctx.get<Reward>(agent).rew = reward;
    }

    if (worldTime.time == 0) {
      reset.resetNow = true;
    }
    

    if (reset.resetNow) {
      resetWorld(ctx);
    }
  }

    

  void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
  {   
    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
							 Action, Location>>({});

    auto time_sys = builder.addToGraph<ParallelForNode<Engine, timeSystem, WorldTime>>({action_sys});

    auto update_obs = builder.addToGraph<ParallelForNode<Engine, observationSystem,
                                                         Location, AgentID, Observation>>({time_sys});

    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, checkDone, WorldReset>>({update_obs});

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
    agents = (Entity *)rawAlloc(2 * sizeof(Entity));

    for (int i = 0; i < 2; i++) {
      agents[i] = ctx.makeEntity<Agent>();

      ctx.get<Action>(agents[i]).choice = 0;
      for (int t = 0; t < 2 * TIME; t++) {
	ctx.get<Observation>(agents[i]).x[t] = 0;
      }
      ctx.get<Observation>(agents[i]).time = 0;
      ctx.get<Location>(agents[i]).x = 0;
      ctx.get<AgentID>(agents[i]).id = i;
      for (int t = 0; t < NUM_MOVES; t++) {
	ctx.get<ActionMask>(agents[i]).isValid[t] = true;
      }
      ctx.get<Reward>(agents[i]).rew = 0.f;
      ctx.get<ActiveAgent>(agents[i]).isActive = true;
    }
    // Initial reset
    resetWorld(ctx);
    ctx.singleton<WorldReset>().resetNow = false;
  }

  MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}
