/*
    Proteus -- High-performance query processing on heterogeneous hardware.

                            Copyright (c) 2014
        Data Intensive Applications and Systems Laboratory (DIAS)
                École Polytechnique Fédérale de Lausanne

                            All Rights Reserved.

    Permission to use, copy, modify and distribute this software and
    its documentation is hereby granted, provided that both the
    copyright notice and this permission notice appear in all copies of
    the software, derivative works or modified versions, and any
    portions thereof, and that both notices appear in supporting
    documentation.

    This code is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. THE AUTHORS
    DISCLAIM ANY LIABILITY OF ANY KIND FOR ANY DAMAGES WHATSOEVER
    RESULTING FROM THE USE OF THIS SOFTWARE.
*/

#ifndef PROFILING_HPP_
#define PROFILING_HPP_

#if __has_include("ittnotify.h")
#include <ittnotify.h>
#else
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-macro-identifier"
#define __itt_event void *
#pragma clang diagnostic pop
#endif

#if __has_include("nvtx3/nvToolsExt.h")
#include <nvtx3/nvToolsExt.h>
#endif

/**
 * Utility wrappers around the runtime APIs of some profilers
 * Currently VTune & Cuda profilers
 */
namespace profiling {
void resume();
void pause();

class ProfileRegion;
class ProfileRegion_ProcessRange;
class ProfileRegionType {
 public:
#if __has_include("ittnotify.h")
  /**
   * Construct a new profiling region type
   * @param region_name name to associate with the region type, depending on the
   * profiler a ProfileRegion constructed with this ProfileRegionType will be
   * displayed with this string.
   * @details For at least VTune, there is an upperbound on the number of
   * ProfileRegionTypes (`__itt_event`) that can be created. Likely due to
   * __itt_event being type def'd as an int. This is not enforced by this class.
   */
  [[nodiscard]] explicit ProfileRegionType(const std::string &region_name)
      : m_region_name(region_name),
        m_event(__itt_event_create(m_region_name.c_str(),
                                   m_region_name.length())) {}
#else
  [[nodiscard]] explicit ProfileRegionType(const std::string &region_name)
      : m_region_name(region_name) {}
#endif

  const std::string m_region_name;

 private:

#if __has_include("ittnotify.h")
  const __itt_event m_event;
#endif
  friend ProfileRegion;
  friend ProfileRegion_ProcessRange;
};

/**
 * Mark a region in a thread
 *
 * Region begins when ProfileRegion is constructed and ends when destructed
 *
 */
class ProfileRegion {
 public:
#if __has_include("ittnotify.h")
  /**
   * Begin a new profiling region
   * @param profile_region_type previously constructed ProfileRegionType to
   * distinguish regions in a profiler
   * @details the lifetime of ProfileRegion does not need to be fully
   * encapsulated by the ProfileRegionType
   */
  [[nodiscard]] explicit ProfileRegion(
      const ProfileRegionType &profile_region_type)
      : m_event(profile_region_type.m_event) {
#else
  [[nodiscard]] explicit ProfileRegion(
      const ProfileRegionType &profile_region_type) {
#endif

#if __has_include("nvtx3/nvToolsExt.h")
    nvtxRangePushA(profile_region_type.m_region_name.c_str());
#endif

#if __has_include("ittnotify.h")
    __itt_event_start(m_event);
#endif
  }

  ~ProfileRegion() {
#if __has_include("nvtx3/nvToolsExt.h")
    nvtxRangePop();
#endif

#if __has_include("ittnotify.h")
    __itt_event_end(m_event);
#endif
  }

 private:
#if __has_include("ittnotify.h")
  const __itt_event m_event;
#endif
};

/**
 * Mark a region in a process
 *
 * Region begins when ProfileRegion is constructed and ends when destructed
 *
 */
class ProfileRegion_ProcessRange {
 public:
  nvtxRangeId_t m_nvtx_range_id;
#if __has_include("ittnotify.h")
  /**
   * Begin a new profiling region
   * @param profile_region_type previously constructed ProfileRegionType to
   * distinguish regions in a profiler
   * @details the lifetime of ProfileRegion does not need to be fully
   * encapsulated by the ProfileRegionType
   */
  [[nodiscard]] explicit ProfileRegion_ProcessRange(
      const ProfileRegionType &profile_region_type)
  {
#else
  [[nodiscard]] explicit ProfileRegion_ProcessRange(
      const ProfileRegionType &profile_region_type) {
#endif

#if __has_include("nvtx3/nvToolsExt.h")
    m_nvtx_range_id =
        nvtxRangeStartA(profile_region_type.m_region_name.c_str());
#endif

#if __has_include("ittnotify.h")
    __itt_event_start(profile_region_type.m_event);
#endif
  }

  ~ProfileRegion_ProcessRange()
  {
#if __has_include("nvtx3/nvToolsExt.h")
      nvtxRangeEnd(m_nvtx_range_id);
#endif

      // #if __has_include("ittnotify.h")
      //     __itt_event_end(m_event);
      // #endif
  }
};

/**
 * Mark a single point in time in a thread
 * TODO fix this and convert it to a function
 */
class ProfileMarkPoint {
 public:
  /**
   * Constructor does not mark a point. Points can be marked with @see
   * ProfileMarkPoint::mark
   */
#if __has_include("ittnotify.h")
  ProfileMarkPoint(std::string point_name)
      : m_point_name(std::move(point_name)),
        m_event(__itt_event_create(point_name.c_str(), point_name.length())) {}
#else
  ProfileMarkPoint(std::string point_name)
      : m_point_name(std::move(point_name)) {}
#endif

  /**
   * Mark a point in time. This can be called multiple times on the same
   * ProfileMarkPoint
   */
  inline void mark() {
#if __has_include("nvtx3/nvToolsExt.h")
    nvtxMarkA(m_point_name.c_str());
#endif

#if __has_include("ittnotify.h")
    __itt_event_start(m_event);
#endif
  }

 private:
  const std::string m_point_name;
#if __has_include("ittnotify.h")
  const __itt_event m_event;
#endif
};

}  // namespace profiling

#endif /* PROFILING_HPP_ */
