#pragma once

#include <madrona/sync.hpp>

namespace Overcooked {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
  };

}
