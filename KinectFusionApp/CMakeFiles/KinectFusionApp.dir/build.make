# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp

# Include any dependencies generated for this target.
include KinectFusionApp/CMakeFiles/KinectFusionApp.dir/depend.make

# Include the progress variables for this target.
include KinectFusionApp/CMakeFiles/KinectFusionApp.dir/progress.make

# Include the compile flags for this target's objects.
include KinectFusionApp/CMakeFiles/KinectFusionApp.dir/flags.make

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/flags.make
KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o: KinectFusionApp/src/depth_camera.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o -c /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/src/depth_camera.cpp

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.i"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/src/depth_camera.cpp > CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.i

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.s"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/src/depth_camera.cpp -o CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.s

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.requires:

.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.requires

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.provides: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.requires
	$(MAKE) -f KinectFusionApp/CMakeFiles/KinectFusionApp.dir/build.make KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.provides.build
.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.provides

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.provides.build: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o


KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/flags.make
KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o: KinectFusionApp/src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KinectFusionApp.dir/src/main.cpp.o -c /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/src/main.cpp

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KinectFusionApp.dir/src/main.cpp.i"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/src/main.cpp > CMakeFiles/KinectFusionApp.dir/src/main.cpp.i

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KinectFusionApp.dir/src/main.cpp.s"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/src/main.cpp -o CMakeFiles/KinectFusionApp.dir/src/main.cpp.s

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.requires:

.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.requires

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.provides: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.requires
	$(MAKE) -f KinectFusionApp/CMakeFiles/KinectFusionApp.dir/build.make KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.provides.build
.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.provides

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.provides.build: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o


# Object files for target KinectFusionApp
KinectFusionApp_OBJECTS = \
"CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o" \
"CMakeFiles/KinectFusionApp.dir/src/main.cpp.o"

# External object files for target KinectFusionApp
KinectFusionApp_EXTERNAL_OBJECTS =

build/KinectFusionApp: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o
build/KinectFusionApp: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o
build/KinectFusionApp: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/build.make
build/KinectFusionApp: /usr/lib/x86_64-linux-gnu/librealsense2.so.2.19.1
build/KinectFusionApp: libKinectFusion.a
build/KinectFusionApp: /usr/local/lib/libopencv_cudabgsegm.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudaobjdetect.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudastereo.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_stitching.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudafeatures2d.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_superres.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudacodec.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_videostab.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudaoptflow.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudalegacy.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudawarping.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_aruco.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_bgsegm.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_bioinspired.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_ccalib.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_dpm.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_face.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_photo.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudaimgproc.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudafilters.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_freetype.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_fuzzy.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_hdf.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_hfs.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_img_hash.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_line_descriptor.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_optflow.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_reg.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_rgbd.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_saliency.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_stereo.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_structured_light.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_surface_matching.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_tracking.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_datasets.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_plot.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_text.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_dnn.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_xfeatures2d.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudaarithm.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_ml.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_shape.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_video.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_ximgproc.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_calib3d.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_features2d.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_flann.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_highgui.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_videoio.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_xobjdetect.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_imgcodecs.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_objdetect.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_xphoto.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_imgproc.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_core.so.3.4.1
build/KinectFusionApp: /usr/local/lib/libopencv_cudev.so.3.4.1
build/KinectFusionApp: /usr/local/cuda-9.0/lib64/libcudart_static.a
build/KinectFusionApp: /usr/lib/x86_64-linux-gnu/librt.so
build/KinectFusionApp: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable ../build/KinectFusionApp"
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KinectFusionApp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
KinectFusionApp/CMakeFiles/KinectFusionApp.dir/build: build/KinectFusionApp

.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/build

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/requires: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/depth_camera.cpp.o.requires
KinectFusionApp/CMakeFiles/KinectFusionApp.dir/requires: KinectFusionApp/CMakeFiles/KinectFusionApp.dir/src/main.cpp.o.requires

.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/requires

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/clean:
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp && $(CMAKE_COMMAND) -P CMakeFiles/KinectFusionApp.dir/cmake_clean.cmake
.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/clean

KinectFusionApp/CMakeFiles/KinectFusionApp.dir/depend:
	cd /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp /home/stefan/UNI/SS2019/RealSenseStuff/KinectFusionApp/KinectFusionApp/CMakeFiles/KinectFusionApp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : KinectFusionApp/CMakeFiles/KinectFusionApp.dir/depend

