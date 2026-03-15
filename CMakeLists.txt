cmake_minimum_required(VERSION 3.10)
project(pcl_icp_registration)

# 找到 PCL（建议使用 PCL 1.8+）
find_package(PCL 1.14 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# add_executable(registration registe.cpp)

# target_link_libraries(registration
#    ${PCL_LIBRARIES}
# )

add_executable(icp_point_to_plane p2p_register.cpp)

target_link_libraries(icp_point_to_plane ${PCL_LIBRARIES})
