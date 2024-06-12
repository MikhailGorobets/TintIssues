# Tint shader compilation issues

<a name="cloning"></a>
# Cloning the Repository

To get the repository, use the following command:

```
git clone https://github.com/MikhailGorobets/TintIssues.git
```

<a name="build_and_run"></a>
# Build and Run Instructions

<a name="build_and_run_win32"></a>
## Win32

Build prerequisites:

* Windows SDK 10.0.19041.0 or later
* C++ build tools

Use either CMake GUI or command line tool to generate build files. For example, to generate 
[Visual Studio 2019/2022](https://www.visualstudio.com/vs/community) 64-bit solution and project files in *build/Win64* folder, 
navigate to the engine's root folder and run the following command:

```
cmake -S . -B ./build/Win64 -G "Visual Studio 17 2022" -A x64
```
