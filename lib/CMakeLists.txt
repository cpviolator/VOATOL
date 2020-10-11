#set (VOATOL_OBJS linalgHelpers.cpp algoHelpers.cpp lapack.cpp)
set (VOATOL_OBJS algorithm_utils.cpp io.cpp blas.cpp sort.cpp)

add_library(voatol STATIC ${VOATOL_OBJS})
add_dependencies(voatol Eigen)
target_include_directories(voatol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)
include_directories(${EIGEN_INCLUDE_DIRS})

add_custom_command( TARGET voatol POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/include ${CMAKE_BINARY_DIR}/include)

if(ENABLE_OPENMP)
  target_link_libraries(voatol PUBLIC OpenMP::OpenMP_CXX)
endif()
