cmake_minimum_required(VERSION 3.20)
project(nncpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(MNIST REQUIRED CONFIG
    PATHS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mnist"
    NO_DEFAULT_PATH
)

set(SANITIZER "" CACHE STRING "Optional sanitizer: address or thread")
set_property(CACHE SANITIZER PROPERTY STRINGS "" address thread)

function(configure_nncpp_compile target)
    target_compile_options(${target} PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wconversion
        -Wsign-conversion
        -Wnull-dereference
        -Werror=uninitialized
        -Werror=return-type
        -fno-omit-frame-pointer
        $<$<STREQUAL:${SANITIZER},address>:-fsanitize=address>
        $<$<STREQUAL:${SANITIZER},thread>:-fsanitize=thread>
        $<$<OR:$<STREQUAL:${SANITIZER},address>,$<STREQUAL:${SANITIZER},thread>>:-O1>
    )
endfunction()

function(configure_nncpp_executable target)
    configure_nncpp_compile(${target})
    target_link_options(${target} PRIVATE
        $<$<STREQUAL:${SANITIZER},address>:-fsanitize=address>
        $<$<STREQUAL:${SANITIZER},thread>:-fsanitize=thread>
    )
endfunction()

add_library(nncpp_core src/network.cpp)
target_include_directories(nncpp_core PUBLIC include)
configure_nncpp_compile(nncpp_core)

add_executable(nncpp main.cpp)
target_link_libraries(nncpp PRIVATE nncpp_core)
target_include_directories(nncpp PRIVATE src ${MNIST_INCLUDE_DIR})
target_compile_definitions(nncpp PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")
configure_nncpp_executable(nncpp)

include(CTest)
if(BUILD_TESTING)
    add_executable(forward_pass_test tests/forward_pass_test.cpp)
    target_link_libraries(forward_pass_test PRIVATE nncpp_core)
    configure_nncpp_executable(forward_pass_test)
    add_test(NAME forward_pass COMMAND forward_pass_test)

    add_executable(backpropagation_test tests/backpropagation_test.cpp)
    target_link_libraries(backpropagation_test PRIVATE nncpp_core)
    configure_nncpp_executable(backpropagation_test)
    add_test(NAME backpropagation COMMAND backpropagation_test)

    add_executable(dense_layer_test tests/dense_layer_test.cpp)
    target_link_libraries(dense_layer_test PRIVATE nncpp_core)
    configure_nncpp_executable(dense_layer_test)
    add_test(NAME dense_layer COMMAND dense_layer_test)

    add_executable(conv2d_test tests/conv2d_test.cpp)
    target_link_libraries(conv2d_test PRIVATE nncpp_core)
    configure_nncpp_executable(conv2d_test)
    add_test(NAME conv2d COMMAND conv2d_test)

    add_executable(max_pool2d_test tests/max_pool2d_test.cpp)
    target_link_libraries(max_pool2d_test PRIVATE nncpp_core)
    configure_nncpp_executable(max_pool2d_test)
    add_test(NAME max_pool2d COMMAND max_pool2d_test)

    add_executable(layer_chain_test tests/layer_chain_test.cpp)
    target_link_libraries(layer_chain_test PRIVATE nncpp_core)
    configure_nncpp_executable(layer_chain_test)
    add_test(NAME layer_chain COMMAND layer_chain_test)

    add_executable(activation_loss_test tests/activation_loss_test.cpp)
    target_link_libraries(activation_loss_test PRIVATE nncpp_core)
    configure_nncpp_executable(activation_loss_test)
    add_test(NAME activation_loss COMMAND activation_loss_test)

    add_executable(mnist_training_test tests/mnist_training_test.cpp)
    target_link_libraries(mnist_training_test PRIVATE nncpp_core)
    target_include_directories(mnist_training_test PRIVATE src)
    configure_nncpp_executable(mnist_training_test)
    add_test(NAME mnist_training COMMAND mnist_training_test)
endif()
