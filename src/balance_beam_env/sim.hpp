#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

#define TIME 3
#define NUM_MOVES 4

namespace Balance {

  // struct RendererInitStub {};
  struct Config {};

  // 3D Position & Quaternion Rotation
  // These classes are defined in madrona/components.hpp

  class Engine;

  enum class ExportID : uint32_t {
    WorldReset,
    ActiveAgent,
    Action,
    Observation,
    ActionMask,
    Reward,
    WorldID,
    AgentID,
    NumExports,
  };

  struct WorldReset {
    int32_t resetNow;
  };

  struct WorldTime {
    int32_t time;
  };

  struct ActiveAgent {
    int32_t isActive;
  };

  struct Action {
    int32_t choice; // 4 discrete choices
  };

  struct Observation {
    int32_t x[2 * TIME];
    int32_t time;
  };

  struct Location {
    int32_t x;
  };

  struct AgentID {
    int32_t id;
  };
    
  struct ActionMask {
    int32_t isValid[NUM_MOVES];
  };

  struct Reward {
    float rew;
  };

  struct Agent : public madrona::Archetype<Action, Observation, Location, AgentID, ActionMask, ActiveAgent, Reward> {};

  struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry, const Config &cfg);

    static void setupTasks(madrona::TaskGraphBuilder &builder, const Config &cfg);

    Sim(Engine &ctx, const Config& cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    madrona::Entity *agents;
  };

  class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
  };

}
