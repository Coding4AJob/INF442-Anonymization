# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/Cellar/cmake/3.19.6/bin/cmake

# The command to remove a file.
RM = /usr/local/Cellar/cmake/3.19.6/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++"

# Include any dependencies generated for this target.
include CMakeFiles/LoadData.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/LoadData.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/LoadData.dir/flags.make

CMakeFiles/LoadData.dir/LoadData.cpp.o: CMakeFiles/LoadData.dir/flags.make
CMakeFiles/LoadData.dir/LoadData.cpp.o: LoadData.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/LoadData.dir/LoadData.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/LoadData.dir/LoadData.cpp.o -c "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++/LoadData.cpp"

CMakeFiles/LoadData.dir/LoadData.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/LoadData.dir/LoadData.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++/LoadData.cpp" > CMakeFiles/LoadData.dir/LoadData.cpp.i

CMakeFiles/LoadData.dir/LoadData.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/LoadData.dir/LoadData.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++/LoadData.cpp" -o CMakeFiles/LoadData.dir/LoadData.cpp.s

# Object files for target LoadData
LoadData_OBJECTS = \
"CMakeFiles/LoadData.dir/LoadData.cpp.o"

# External object files for target LoadData
LoadData_EXTERNAL_OBJECTS =

LoadData: CMakeFiles/LoadData.dir/LoadData.cpp.o
LoadData: CMakeFiles/LoadData.dir/build.make
LoadData: CMakeFiles/LoadData.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable LoadData"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/LoadData.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/LoadData.dir/build: LoadData

.PHONY : CMakeFiles/LoadData.dir/build

CMakeFiles/LoadData.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/LoadData.dir/cmake_clean.cmake
.PHONY : CMakeFiles/LoadData.dir/clean

CMakeFiles/LoadData.dir/depend:
	cd "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++" "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++" "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++" "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++" "/Users/frank/Documents/GitHub/INF442-Anonymization/5.1/Machine Learning Algo/C++/CMakeFiles/LoadData.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/LoadData.dir/depend

