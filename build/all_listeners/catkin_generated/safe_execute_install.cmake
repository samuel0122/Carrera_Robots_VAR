execute_process(COMMAND "/home/samuel/Carrera_Robots_VAR/build/all_listeners/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/samuel/Carrera_Robots_VAR/build/all_listeners/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
