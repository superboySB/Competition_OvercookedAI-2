#pragma once

#include <madrona/sync.hpp>

namespace Cartpole {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
  };

}
