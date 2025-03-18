cmake_minimum_required(VERSION 4.0)
project(AIDEEPLEARNINGPROBLEM)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include the OpenCV header files
include_directories(${OpenCV_INCLUDE_DIRS})



# Add the executable for the project
add_executable(AIDEEPLEARNINGPROBLEM src/run.cpp)

# Link OpenCV libraries to the project
target_link_libraries(AIDEEPLEARNINGPROBLEM ${OpenCV_LIBS})
