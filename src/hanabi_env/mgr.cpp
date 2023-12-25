#include "mgr.hpp"
#include "sim.hpp"

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/mw_cpu.hpp>

#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::py;

namespace Hanabi {

  using CPUExecutor =
    TaskGraphExecutor<Engine, Sim, Config, WorldInit>;

  struct Manager::Impl {
    Config cfg;
    EpisodeManager *episodeMgr;

    static inline Impl * init(const Config &cfg);

    inline Impl(const Config &c, EpisodeManager *episode_mgr)
      : cfg(c),
	episodeMgr(episode_mgr)
    {}

    inline virtual ~Impl() {};
    virtual void run() = 0;
    virtual Tensor exportTensor(ExportID slot, Tensor::ElementType type,
                                Span<const int64_t> dims) = 0;
  };

  struct Manager::CPUImpl final : public Manager::Impl {
    CPUExecutor mwCPU;

    inline CPUImpl(const Config &cfg,
                   const Hanabi::Config &app_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits)
      : Impl(cfg, episode_mgr),
	mwCPU(ThreadPoolExecutor::Config {
	  .numWorlds = cfg.numWorlds,
	  // .renderWidth = 0,
	  // .renderHeight = 0,
	  .numExportedBuffers = (uint32_t)ExportID::NumExports,
	  // .cameraMode = render::CameraMode::None,
	  // .renderGPUID = 0,
	},
	  app_cfg,
	  world_inits)
    {}

    inline virtual ~CPUImpl() final
    {
      free(episodeMgr);
    }

    inline virtual void run() final { mwCPU.run(); }

    virtual inline Tensor exportTensor(ExportID slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
      void *dev_ptr = mwCPU.getExported((uint32_t)slot);
      return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
  };

#ifdef MADRONA_CUDA_SUPPORT
  struct Manager::GPUImpl final : public Manager::Impl {
    MWCudaExecutor mwGPU;

    inline GPUImpl(const Config &cfg,
                   const Hanabi::Config &app_cfg,
                   EpisodeManager *episode_mgr,
                   WorldInit *world_inits)
      : Impl(cfg, episode_mgr),
	mwGPU({
	  .worldInitPtr = world_inits,
	  .numWorldInitBytes = sizeof(WorldInit),
	  .userConfigPtr = (void *)&app_cfg,
	  .numUserConfigBytes = sizeof(Hanabi::Config),
	  .numWorldDataBytes = sizeof(Sim),
	  .worldDataAlignment = alignof(Sim),
	  .numWorlds = cfg.numWorlds,
	  .numExportedBuffers = (uint32_t)ExportID::NumExports,
	  .gpuID = (uint32_t)cfg.gpuID,
	  // .cameraMode = render::CameraMode::None,
	  // .renderWidth = 0,
	  // .renderHeight = 0,
	}, {
	  { HANABI_SRC_LIST },
	  { HANABI_COMPILE_FLAGS },
	  cfg.debugCompile ? CompileConfig::OptMode::Debug :
	  CompileConfig::OptMode::LTO
	})
    {}

    inline virtual ~GPUImpl() final
    {
      REQ_CUDA(cudaFree(episodeMgr));
    }

    inline virtual void run() final { mwGPU.run(); }
    virtual inline Tensor exportTensor(ExportID slot, Tensor::ElementType type,
                                       Span<const int64_t> dims) final
    {
      void *dev_ptr = mwGPU.getExported((uint32_t)slot);
      return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
  };
#endif

  static HeapArray<WorldInit> setupWorldInitData(int64_t num_worlds,
						 EpisodeManager *episode_mgr,
						 const Manager::Config &cfg)
  {
    HeapArray<WorldInit> world_inits(num_worlds);

    for (int64_t i = 0; i < num_worlds; i++) {
      world_inits[i] = WorldInit {
	episode_mgr,
	cfg.colors,
	cfg.ranks,
	cfg.players,
	cfg.max_information_tokens,
	cfg.max_life_tokens
      };
    }

    return world_inits;
  }

  Manager::Impl * Manager::Impl::init(const Manager::Config &cfg)
  {
    Hanabi::Config app_cfg {cfg.players};
    switch (cfg.execMode) {
    case ExecMode::CPU: {
      EpisodeManager *episode_mgr = new EpisodeManager { 0 };
      HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds, episode_mgr, cfg);
      return new CPUImpl(cfg, app_cfg, episode_mgr, world_inits.data());
    } break;
    case ExecMode::CUDA: {
#ifndef MADRONA_CUDA_SUPPORT
      FATAL("CUDA support not compiled in!");
#else
      EpisodeManager *episode_mgr = (EpisodeManager *)cu::allocGPU(sizeof(EpisodeManager));
      // Set the current episode count to 0
      REQ_CUDA(cudaMemset(episode_mgr, 0, sizeof(EpisodeManager)));
      HeapArray<WorldInit> world_inits = setupWorldInitData(cfg.numWorlds, episode_mgr, cfg);
      return new GPUImpl(cfg, app_cfg, episode_mgr, world_inits.data());
#endif
    } break;
    default: return nullptr;
    }
  }

  MADRONA_EXPORT Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
  {}

  MADRONA_EXPORT Manager::~Manager() {}

  MADRONA_EXPORT void Manager::step()
  {
    impl_->run();
  }

  MADRONA_EXPORT Tensor Manager::doneTensor() const
  {
    return impl_->exportTensor(ExportID::WorldReset, Tensor::ElementType::Int32,
			       {impl_->cfg.numWorlds});
  }

  MADRONA_EXPORT Tensor Manager::activeAgentTensor() const
  {
    return impl_->exportTensor(ExportID::ActiveAgent, Tensor::ElementType::Int32,
			       {2, impl_->cfg.numWorlds});
  }

  MADRONA_EXPORT Tensor Manager::actionTensor() const
  {
    return impl_->exportTensor(ExportID::Action, Tensor::ElementType::Int32,
			       {2, impl_->cfg.numWorlds, 1});
  }

  MADRONA_EXPORT Tensor Manager::observationTensor() const
  {
    return impl_->exportTensor(ExportID::Observation, Tensor::ElementType::Int8,
                               {N_PLAYERS, impl_->cfg.numWorlds, sizeof(Observation)});
  }

  MADRONA_EXPORT Tensor Manager::agentStateTensor() const
  {
    return impl_->exportTensor(ExportID::State, Tensor::ElementType::Int8,
                               {N_PLAYERS, impl_->cfg.numWorlds, sizeof(State)});
  }


  MADRONA_EXPORT Tensor Manager::actionMaskTensor() const
  {
    return impl_->exportTensor(ExportID::ActionMask, Tensor::ElementType::Int32,
			       {2, impl_->cfg.numWorlds, NUM_MOVES});
  }
    
  MADRONA_EXPORT Tensor Manager::rewardTensor() const
  {
    return impl_->exportTensor(ExportID::Reward, Tensor::ElementType::Float32,
			       {2, impl_->cfg.numWorlds});
  }

  MADRONA_EXPORT Tensor Manager::worldIDTensor() const
  {
    return impl_->exportTensor(ExportID::WorldID, Tensor::ElementType::Int32,
			       {2, impl_->cfg.numWorlds});
  }

  MADRONA_EXPORT Tensor Manager::agentIDTensor() const
  {
    return impl_->exportTensor(ExportID::AgentID, Tensor::ElementType::Int32,
			       {2, impl_->cfg.numWorlds});
  }

}
