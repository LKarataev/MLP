cmake_minimum_required(VERSION 3.5)

project(CPP7_MLP-0 VERSION 0.1 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS PrintSupport)

set(PROJECT_SOURCES
        src/main.cc
        src/view/mainwindow.cc
        src/view/mainwindow.h
        src/view/mainwindow.ui
        src/view/canvas/canvas.h
        src/view/canvas/canvas.cc
        src/view/canvas/qcustomplot.h
        src/view/canvas/qcustomplot.cpp
        src/model/interfaces/neuralnetwork.h
        src/model/interfaces/neuron.h
        src/model/utils/utils.h
        src/model/utils/utils.cc
        src/model/graphnetwork/graphneuron.h
        src/model/graphnetwork/graphneuron.cc
        src/model/graphnetwork/graphnetwork.h
        src/model/graphnetwork/graphnetwork.cc
        src/model/matrixnetwork/matrixneuron.h
        src/model/matrixnetwork/matrixneuron.cc
        src/model/matrixnetwork/matrixnetwork.h
        src/model/matrixnetwork/matrixnetwork.cc
        src/model/perceptron.h
        src/model/perceptron.cc
        src/model/storage/networks_storage.h
        src/model/storage/networks_storage.cc
        src/controller/controller.h
        src/controller/controller.cc
        src/resources/resources.qrc
)

if(${QT_VERSION_MAJOR} GREATER_EQUAL 6)
    qt_add_executable(CPP7_MLP-0
        MANUAL_FINALIZATION
        ${PROJECT_SOURCES}
    )
# Define target properties for Android with Qt 6 as:
#    set_property(TARGET CPP7_MLP-0 APPEND PROPERTY QT_ANDROID_PACKAGE_SOURCE_DIR
#                 ${CMAKE_CURRENT_SOURCE_DIR}/android)
# For more information, see https://doc.qt.io/qt-6/qt-add-executable.html#target-creation
else()
    if(ANDROID)
        add_library(CPP7_MLP-0 SHARED
            ${PROJECT_SOURCES}
        )
# Define properties for Android with Qt 5 after find_package() calls as:
#    set(ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android")
    else()
        add_executable(CPP7_MLP-0
            ${PROJECT_SOURCES}
        )
    endif()
endif()

target_link_libraries(CPP7_MLP-0 PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(CPP7_MLP-0 PRIVATE Qt${QT_VERSION_MAJOR}::PrintSupport)

set_target_properties(CPP7_MLP-0 PROPERTIES
    MACOSX_BUNDLE_GUI_IDENTIFIER my.example.com
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE
)

install(TARGETS CPP7_MLP-0
    BUNDLE DESTINATION .
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(CPP7_MLP-0)
endif()
