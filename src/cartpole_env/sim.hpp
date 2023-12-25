#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace Cartpole {

  // struct RendererInitStub {};

  class Engine;

  enum class ExportID : uint32_t {
    WorldReset,
    Action,
    State,
    Reward,
    WorldID,
    NumExports,
  };

  struct WorldReset {
    int32_t resetNow;
  };

  struct Action {
    int32_t choice; // Binary Action
  };

  struct State {
    float x;
    float x_dot;
    float theta;
    float theta_dot;
  };

  struct Reward {
    float rew;
  };

  struct Agent : public madrona::Archetype<WorldReset, Action, State, Reward> {};

  struct Config {};

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
