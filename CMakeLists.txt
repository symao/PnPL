cmake_minimum_required(VERSION 2.8.3)
project(PnPL)

set(CMAKE_CXX_FLAGS "-std=c++11")


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules/")
FIND_PACKAGE(G2O REQUIRED)
find_package(CSparse REQUIRED)
find_package(Cholmod REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS}
                    ${G2O_INCLUDE_DIR}
                    ${EIGEN3_INCLUDE_DIR}
                    ${CSPARSE_INCLUDE_DIR}
                    ${CHOLMOD_INCLUDE_DIR})

SET(G2O_LIBS g2o_cli g2o_ext_freeglut_minimal g2o_simulator g2o_solver_slam2d_linear g2o_types_icp g2o_types_slam2d g2o_core g2o_interface g2o_solver_csparse g2o_solver_structure_only g2o_types_sba g2o_types_slam3d g2o_csparse_extension g2o_opengl_helper g2o_solver_dense g2o_stuff g2o_types_sclam2d g2o_parser g2o_solver_pcg g2o_types_data g2o_types_sim3 cxsparse )


add_executable(demo_pnpl src/demo_pnpl.cpp src/pnpl.cpp)

target_link_libraries(demo_pnpl ${OpenCV_LIBRARIES} ${G2O_LIBS})

add_executable(demo_pnptrans src/demo_pnptrans.cpp src/pnp.cpp)

target_link_libraries(demo_pnptrans ${OpenCV_LIBRARIES} ${G2O_LIBS})
