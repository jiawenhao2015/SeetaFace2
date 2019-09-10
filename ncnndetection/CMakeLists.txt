

cmake_minimum_required(VERSION 2.8)

project(ncnndetection)

if(UNIX)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11  -fPIC")
endif()

add_definitions(-DSEETA_EXPORTS)

file(GLOB_RECURSE ROOT_SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    )


file(GLOB INCLUDE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/beauty/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/beauty/*.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/beauty/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/ncnn/include
    )

list(APPEND SRC_FILES ${ROOT_SRC_FILES})


set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
# add library
add_library(${PROJECT_NAME} ${SRC_FILES}  )
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/../3rdparty/ncnn/include

    )

message(${PROJECT_SOURCE_DIR}/../3rdparty/ncnn/include)
target_link_libraries(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/3rdparty/ncnn/lib/libncnn.a)

set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(${PROJECT_NAME} PROPERTIES VERSION ${BUILD_VERSION})
