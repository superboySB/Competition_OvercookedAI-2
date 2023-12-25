#pragma once

#include <madrona/sync.hpp>

namespace Simplecooked {

  struct EpisodeManager {
    madrona::AtomicU32 curEpisode;
  };

  struct WorldInit {
    EpisodeManager *episodeMgr;
  };

}
