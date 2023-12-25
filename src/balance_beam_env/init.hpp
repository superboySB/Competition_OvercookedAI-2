#pragma once

#include <madrona/sync.hpp>

namespace Balance {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
  };

}
