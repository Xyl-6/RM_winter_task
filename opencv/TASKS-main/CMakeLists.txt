cmake_minimum_required(VERSION 3.10)
project( DisplayImage )
find_package( OpenCV REQUIRED )
add_executable( DisplayImage finaltask.cpp )
target_link_libraries( DisplayImage ${OpenCV_LIBS} )
