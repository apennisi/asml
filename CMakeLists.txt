cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_DEBUG} -Wdelete-non-virtual-dtor")

project( nei_gui )

find_package(Boost COMPONENTS filesystem regex)
find_package(OpenCV REQUIRED)
find_package(CGAL QUIET COMPONENTS Core )
find_package(Qt5Core REQUIRED)
find_package(Qt5Gui REQUIRED)
find_package(Qt5Widgets REQUIRED)

#INCLUDE(${QT_USE_FILE})
include( ${CGAL_USE_FILE} )
include( CGAL_CreateSingleSourceCGALProgram )

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_AUTOMOC ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(EXECUTABLE_OUTPUT_PATH ../bin)
file(GLOB_RECURSE objectannotationtool_src "src/*.cpp")

QT5_WRAP_UI(UI_INTERFACE 
			gui/mainwindow.ui 
)

include_directories(${Boost_INCLUDE_DIRS} ${CMAKE_BINARY_DIR} src)
add_executable( nei_gui ${objectannotationtool_src} ${UI_INTERFACE})
target_link_libraries( nei_gui ${Boost_LIBRARIES} ${OpenCV_LIBS} Qt5::Core
    Qt5::Gui
    Qt5::Widgets )
