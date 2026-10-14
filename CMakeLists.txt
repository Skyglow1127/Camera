cmake_minimum_required(VERSION 3.16)
project(Camera LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# SDK 始终位于项目根目录下，避免依赖用户主目录的绝对路径。
set(ORBBEC_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/SDK")
set(ORBBEC_SDK_INCLUDE_DIR "${ORBBEC_SDK_DIR}/include")
set(ORBBEC_SDK_LIBRARY "${ORBBEC_SDK_DIR}/lib/libOrbbecSDK.so")

if(NOT EXISTS "${ORBBEC_SDK_INCLUDE_DIR}/libobsensor/ObSensor.hpp")
    message(FATAL_ERROR "Orbbec SDK headers were not found in ${ORBBEC_SDK_INCLUDE_DIR}")
endif()
if(NOT EXISTS "${ORBBEC_SDK_LIBRARY}")
    message(FATAL_ERROR "Orbbec SDK library was not found at ${ORBBEC_SDK_LIBRARY}")
endif()

add_library(OrbbecSDK SHARED IMPORTED)
set_target_properties(OrbbecSDK PROPERTIES
    IMPORTED_LOCATION "${ORBBEC_SDK_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORBBEC_SDK_INCLUDE_DIR}"
)

function(add_camera_executable target source)
    add_executable(${target} ${source})
    target_include_directories(${target} PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(${target} PRIVATE OrbbecSDK ${OpenCV_LIBS})
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "${ORBBEC_SDK_DIR}/lib"
        INSTALL_RPATH "$ORIGIN/../SDK/lib"
    )
endfunction()

add_camera_executable(streamer streamer.cpp)
add_camera_executable(detect detect.cpp)
add_library(vision_core vision_core.cpp vision_io.cpp handeye.cpp)
target_include_directories(vision_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_core PUBLIC ${OpenCV_LIBS})
target_link_libraries(detect PRIVATE vision_core Threads::Threads)
target_compile_options(vision_core PRIVATE -Wall -Wextra -Wpedantic)
target_compile_options(detect PRIVATE -Wall -Wextra -Wpedantic)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-linux-x64-1.31.0" CACHE PATH "ONNX Runtime CPU distribution")
if(EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    add_library(onnxruntime SHARED IMPORTED)
    set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so"
        INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
    add_library(segmentation segmentation.cpp)
    target_link_libraries(segmentation PUBLIC vision_core onnxruntime)
    target_compile_options(segmentation PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(detect PRIVATE segmentation)
    target_compile_definitions(detect PRIVATE HAVE_ONNXRUNTIME=1)
endif()
add_executable(handeye_calibrate handeye_calibrate.cpp)
target_link_libraries(handeye_calibrate PRIVATE vision_core)
target_compile_options(handeye_calibrate PRIVATE -Wall -Wextra -Wpedantic)
include(CTest)
if(BUILD_TESTING)
    add_executable(vision_tests tests/vision_tests.cpp)
    target_link_libraries(vision_tests PRIVATE vision_core)
    add_test(NAME vision_geometry_and_calibration COMMAND vision_tests)
endif()

# detect 通过 CMake 获取模型目录，项目移动后只需重新构建即可。
target_compile_definitions(detect PRIVATE
    CAMERA_ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)
