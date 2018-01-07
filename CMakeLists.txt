###############################################################
######                                                   ######
######			Author: Pablo R.S. (aka. Bardo91)        ######
######                                                   ######
###############################################################

cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project(grasp_arm_sim)

# Clasify sources according to folder structure.
# This creates a nice layout for the project in visual studio
# This macro is courtesy of http://www.cmake.org/pipermail/cmake/2013-November/056336.html
macro(GroupSources curdir)
	file(GLOB children RELATIVE ${PROJECT_SOURCE_DIR}/${curdir}
		${PROJECT_SOURCE_DIR}/${curdir}/*)
	foreach(child ${children})
		if(IS_DIRECTORY ${PROJECT_SOURCE_DIR}/${curdir}/${child})
			GroupSources(${curdir}/${child})
		else()
			string(REPLACE "/" "\\" groupname ${curdir})
			source_group(${groupname} FILES ${PROJECT_SOURCE_DIR}/${curdir}/${child})
		endif()
	endforeach()
endmacro()

GroupSources(src)

#########################################
###### Loading 3rd party libraries.######
#########################################
find_package(catkin REQUIRED COMPONENTS
			geometry_msgs
			rgbd_tools
			grasping_tools
			arm_controller)


catkin_package(
  CATKIN_DEPENDS roscpp std_msgs rgbd_tools grasping_tools arm_controller
)

find_package(PCL 1.8.0 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
#########################################
######			Executable		   ######
#########################################
IF(UNIX)
	add_definitions(-std=c++11)
ENDIF()

file(GLOB_RECURSE SRC_FILES "src/*.cpp" "src/*.h" "src/*.inl" )
GroupSources(src)

add_executable(${PROJECT_NAME} ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries (${PROJECT_NAME} ${PCL_LIBRARIES})

target_include_directories(${PROJECT_NAME} PUBLIC ${catkin_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} rgbd_tools grasping_tools arm_controller)

## TODO: how to add internal dep by adding the library (grasping tool in this case)
if(NOT DEFINED ${CMAKE_BUILD_TYPE})
	set(CMAKE_BUILD_TYPE Release)
endif(NOT EXIST ${CMAKE_BUILD_TYPE})

target_include_directories(${PROJECT_NAME} PUBLIC "../grasping_tools/dep/gpis")
target_link_libraries(${PROJECT_NAME} LINK_PUBLIC gpis)

target_include_directories(${PROJECT_NAME} PUBLIC "../grasping_tools/dep/qhull/src")
if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC qhullcpp_d)
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC qhullstatic_rd)
else()
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC qhullcpp)
    target_link_libraries(${PROJECT_NAME} LINK_PUBLIC qhullstatic_r)
endif()

target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${qhull_STATIC})
