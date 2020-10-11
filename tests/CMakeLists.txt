add_executable (trlm_exe trlm.cpp)
target_link_libraries(trlm_exe voatol)
include_directories(trlm_exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN_INCLUDE_DIRS})

add_executable (blktrlm_exe blktrlm.cpp)
target_link_libraries(blktrlm_exe voatol)
include_directories(blktrlm_exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN_INCLUDE_DIRS})

add_executable (iram_exe iram.cpp)
target_link_libraries(iram_exe voatol)
include_directories(iram_exe PUBLIC ${PROJECT_SOURCE_DIR}/include)
include_directories(${EIGEN_INCLUDE_DIRS})
