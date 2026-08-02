
#pragma once
// included by ponni_kokkos_utils.h

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

namespace ponni {

  inline LinearAllocator & device_pool() {
    static LinearAllocator pool;
    return pool;
  }

  inline void init_device_pool(std::size_t pool_size_bytes) {
    constexpr std::size_t block_size = 4096;
    LinearAllocator & pool = device_pool();
    if (pool.poolSize() != 0) Kokkos::abort("PONNI DeviceSpace pool is already initialized");
    pool = LinearAllocator(pool_size_bytes, block_size);
  }

  inline void finalize_device_pool() {
    Kokkos::fence("Morrison DeviceSpace finalization");
    device_pool().finalize();
  }


  class DeviceSpace {
    public:
    using memory_space    = ponni::DeviceSpace;
    using execution_space = Kokkos::DefaultExecutionSpace;
    using device_type     = Kokkos::Device<execution_space,memory_space>;
    using size_type       = Kokkos::DefaultExecutionSpace::memory_space::size_type;
    DeviceSpace()                    = default; // default constructible
    DeviceSpace(const DeviceSpace &) = default; // copy constructible
    ~DeviceSpace()                   = default; // destructible
    static const char * name() { return "ponni::DeviceSpace"; }
    template <class Ex> void * allocate(Ex const & /*ex*/ , size_t const sz) const {
      return device_pool().allocate(sz,"[Unlabeled]");
    }
    template <class Ex> void * allocate(Ex const & /*ex*/ , char const * label , size_t const sz , size_t const /*logical_sz*/=0) const {
      return device_pool().allocate(sz,label);
    }
    void * allocate(size_t const sz) const {
      return device_pool().allocate(sz,"[Unlabeled]");
    }
    void * allocate(char const * label , size_t const sz , size_t const /*logical_sz*/=0) const {
      return device_pool().allocate(sz,label);
    }
    void deallocate(void * const ptr , size_t sz) const {
      device_pool().free(ptr,"[Unlabeled]");
    }
    void deallocate(char const * label, void * const ptr , size_t const sz , size_t const /*logical_sz*/=0 ) const {
      device_pool().free(ptr,label);
    }
  };
}


#ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL
KOKKOS_IMPL_SHARED_ALLOCATION_SPECIALIZATION(ponni::DeviceSpace);
#else
KOKKOS_IMPL_HOST_INACCESSIBLE_SHARED_ALLOCATION_SPECIALIZATION(ponni::DeviceSpace);
#endif


namespace Kokkos {
  namespace Impl {
    // VerifyExecutionCanAccessMemorySpace appears to not be there anymore?

    #ifdef KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL

    template <> struct MemorySpaceAccess<HostSpace,ponni::DeviceSpace> {
      enum : bool { assignable = true };
      enum : bool { accessible = true };
      enum : bool { deepcopy   = true };
    };

    template <> struct MemorySpaceAccess<ponni::DeviceSpace,HostSpace> {
      enum : bool { assignable = true };
      enum : bool { accessible = true };
      enum : bool { deepcopy   = true };
    };

    #else

    #if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) || defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)

    template <> struct MemorySpaceAccess<HostSpace,ponni::DeviceSpace> {
      enum : bool { assignable = true };
      enum : bool { accessible = true };
      enum : bool { deepcopy   = true };
    };

    template <> struct MemorySpaceAccess<ponni::DeviceSpace,HostSpace> {
      enum : bool { assignable = true };
      enum : bool { accessible = true };
      enum : bool { deepcopy   = true };
    };

    #else

    template <> struct MemorySpaceAccess<HostSpace,ponni::DeviceSpace> {
      enum : bool { assignable = false };
      enum : bool { accessible = false };
      enum : bool { deepcopy   = true  };
    };

    template <> struct MemorySpaceAccess<ponni::DeviceSpace,HostSpace> {
      enum : bool { assignable = false };
      enum : bool { accessible = false };
      enum : bool { deepcopy   = true  };
    };

    template <> struct MemorySpaceAccess<Kokkos::DefaultExecutionSpace::memory_space,ponni::DeviceSpace> {
      enum : bool { assignable = true  };
      enum : bool { accessible = true  };
      enum : bool { deepcopy   = true  };
    };

    template <> struct MemorySpaceAccess<ponni::DeviceSpace,Kokkos::DefaultExecutionSpace::memory_space> {
      enum : bool { assignable = true  };
      enum : bool { accessible = true  };
      enum : bool { deepcopy   = true  };
    };

    #endif
    #endif

    template <typename ExecSpace>
    struct DeepCopy<Kokkos::HostSpace,ponni::DeviceSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::HostSpace,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::HostSpace,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };

    template <typename ExecSpace>
    struct DeepCopy<ponni::DeviceSpace,Kokkos::HostSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::HostSpace,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::HostSpace,ExecSpace>(exec,dst,src,n);
      }
    };

    template <typename ExecSpace>
    struct DeepCopy<ponni::DeviceSpace,ponni::DeviceSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };

    #if ! defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_SERIAL) && ! defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP) && ! defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
    template <typename ExecSpace>
    struct DeepCopy<ponni::DeviceSpace,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };

    template <typename ExecSpace>
    struct DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,ponni::DeviceSpace,ExecSpace> {
      DeepCopy(void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(dst,src,n);
      }
      DeepCopy(ExecSpace const & exec , void * dst , void const * src , size_t n) {
        DeepCopy<Kokkos::DefaultExecutionSpace::memory_space,Kokkos::DefaultExecutionSpace::memory_space,ExecSpace>(exec,dst,src,n);
      }
    };
    #endif

  }
}


