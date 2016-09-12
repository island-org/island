island
============

island is a light-weight and low-level creative coding framework.

Getting Started
--------
Clone the repository with `git clone --recurse-submodules https://github.com/island-org/island.git` in order to pull in the 3rdparty repos.

You can also pull in the 3rdparty repos after cloning the repository with `git submodule update --init --recursive`.

To generate solutions, you would need [premake](http://premake.github.io/download.html).

On Windows, you can use the included `premake5.exe` with `premake5 vs2013`.

Examples
--------

### [01-processing-tree](https://github.com/vinjn/island/tree/master/examples/01-processing-tree)

Port of a Processing sample to island with most codes kept same.

![](https://github.com/vinjn/island/raw/master/examples/01-processing-tree/screenshot.png)

### [06-tilemap-editor](https://github.com/vinjn/island/tree/master/examples/06-tilemap-editor)

WIP intergration of `stb_tilemap_editor.h` and `sketch2d.h`, a 2d tilemap editor.

![](https://github.com/vinjn/island/raw/master/examples/06-tilemap-editor/screenshot.png)

### [09-cuda-shadertoy](https://github.com/vinjn/island/tree/master/examples/09-cuda-shadertoy)

[ShaderToy](http://shadertoy.com/) in CUDA language.

![](https://github.com/vinjn/island/raw/master/examples/09-cuda-shadertoy/screenshot.png)
