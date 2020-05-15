cmake_minimum_required(VERSION 3.8)
project(semester_work)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(${PROJECT_NAME} main.cpp ini.cpp)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
