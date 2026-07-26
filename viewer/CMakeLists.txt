cmake_minimum_required(VERSION 3.15)
project(NeuroViz3D CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
add_executable(neuroviz3d
    src/main.cpp
    src/shader.cpp
    src/mesh.cpp
    src/camera.cpp
    src/network_loader.cpp
    external/glad/src/glad.c
    external/imgui/imgui.cpp
    external/imgui/imgui_demo.cpp
    external/imgui/imgui_draw.cpp
    external/imgui/imgui_tables.cpp
    external/imgui/imgui_widgets.cpp
    external/imgui/imgui_impl_glfw.cpp
    external/imgui/imgui_impl_opengl3.cpp
)
target_include_directories(neuroviz3d PRIVATE
    src
    external/glad/include
    external/imgui
)
target_link_libraries(neuroviz3d PRIVATE glfw glm::glm opengl32)
if(WIN32)
    set_target_properties(neuroviz3d PROPERTIES LINK_FLAGS "-mconsole -Wl,-subsystem,console")
endif()
