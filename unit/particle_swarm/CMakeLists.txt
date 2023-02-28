
add_executable(particle_swarm test_particle_swarm.cpp)
include($ENV{YAKL_HOME}/yakl_utils.cmake)
yakl_process_target(particle_swarm)
target_link_libraries(particle_swarm ponni)
if (DEFINED HDF5_LINK_FLAGS)
  target_link_libraries(particle_swarm "${HDF5_LINK_FLAGS}")
endif()

if (DEFINED MPI_COMMAND)
  separate_arguments(REAL_MPI_COMMAND UNIX_COMMAND $ENV{MPI_COMMAND})
  add_test(NAME particle_swarm_test COMMAND ${REAL_MPI_COMMAND} ./particle_swarm)
else()
  add_test(NAME particle_swarm_test COMMAND ./particle_swarm)
endif()

