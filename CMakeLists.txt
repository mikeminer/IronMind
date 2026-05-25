cmake_minimum_required(VERSION 3.20)
project(IronMind C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(ironmind-inspect native/ironmind_inspect.c)
target_compile_definitions(ironmind-inspect PRIVATE _CRT_SECURE_NO_WARNINGS)

if (MSVC)
  target_compile_options(ironmind-inspect PRIVATE /W4)
else()
  target_compile_options(ironmind-inspect PRIVATE -Wall -Wextra -Wpedantic)
endif()
