# Add executable called "iram" that is built from the source files
# "iram.cpp". The extensions are automatically found.
add_executable (trlm_exe trlm.cpp)
add_executable (blktrlm_exe blktrlm.cpp)
add_executable (iram_exe iram.cpp)

# Link the executable to the voatol library. Since the library has
# public include directories we will use those link directories when building
# iram
target_link_libraries (libvoatol LINK_PUBLIC ../include)
