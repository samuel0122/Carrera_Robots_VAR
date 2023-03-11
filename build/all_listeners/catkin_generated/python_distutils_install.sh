#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/samuel/P1_Carrera_de_robots/src/all_listeners"

# ensure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/samuel/P1_Carrera_de_robots/install/lib/python3/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/samuel/P1_Carrera_de_robots/install/lib/python3/dist-packages:/home/samuel/P1_Carrera_de_robots/build/lib/python3/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/samuel/P1_Carrera_de_robots/build" \
    "/usr/bin/python3" \
    "/home/samuel/P1_Carrera_de_robots/src/all_listeners/setup.py" \
     \
    build --build-base "/home/samuel/P1_Carrera_de_robots/build/all_listeners" \
    install \
    --root="${DESTDIR-/}" \
    --install-layout=deb --prefix="/home/samuel/P1_Carrera_de_robots/install" --install-scripts="/home/samuel/P1_Carrera_de_robots/install/bin"
