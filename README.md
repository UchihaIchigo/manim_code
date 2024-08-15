

pypi version MIT License Manim Subreddit Manim Discord docs

Manim is an engine for precise programmatic animations, designed for creating explanatory math videos.

Note, there are two versions of manim. This repository began as a personal project by the author of 3Blue1Brown for the purpose of animating those videos, with video-specific code available here. In 2020 a group of developers forked it into what is now the community edition, with a goal of being more stable, better tested, quicker to respond to community contributions, and all around friendlier to get started with. See this page for more details.

Installation
WARNING: These instructions are for ManimGL only. Trying to use these instructions to install ManimCommunity/manim or instructions there to install this version will cause problems. You should first decide which version you wish to install, then only follow the instructions for your desired version.

Note: To install manim directly through pip, please pay attention to the name of the installed package. This repository is ManimGL of 3b1b. The package name is manimgl instead of manim or manimlib. Please use pip install manimgl to install the version in this repository.

Manim runs on Python 3.7 or higher.

System requirements are FFmpeg, OpenGL and LaTeX (optional, if you want to use LaTeX). For Linux, Pango along with its development headers are required. See instruction here.

Directly
# Install manimgl
pip install manimgl

# Try it out
manimgl
For more options, take a look at the Using manim sections further below.

If you want to hack on manimlib itself, clone this repository and in that directory execute:

# Install manimgl
pip install -e .

# Try it out
manimgl example_scenes.py OpeningManimExample
# or
manim-render example_scenes.py OpeningManimExample
Directly (Windows)
Install FFmpeg.
Install a LaTeX distribution. MiKTeX is recommended.
Install the remaining Python packages.
git clone https://github.com/3b1b/manim.git
cd manim
pip install -e .
manimgl example_scenes.py OpeningManimExample
Mac OSX
Install FFmpeg, LaTeX in terminal using homebrew.

brew install ffmpeg mactex
Install latest version of manim using these command.

git clone https://github.com/3b1b/manim.git
cd manim
pip install -e .
manimgl example_scenes.py OpeningManimExample
Anaconda Install
Install LaTeX as above.
Create a conda environment using conda create -n manim python=3.8.
Activate the environment using conda activate manim.
Install manimgl using pip install -e ..
Using manim
Try running the following:

manimgl example_scenes.py OpeningManimExample
This should pop up a window playing a simple scene.

Some useful flags include:

-w to write the scene to a file
-o to write the scene to a file and open the result
-s to skip to the end and just show the final frame.
-so will save the final frame to an image and show it
-n <number> to skip ahead to the n'th animation of a scene.
-f to make the playback window fullscreen
Take a look at custom_config.yml for further configuration. To add your customization, you can either edit this file, or add another file by the same name "custom_config.yml" to whatever directory you are running manim from. For example this is the one for 3blue1brown videos. There you can specify where videos should be output to, where manim should look for image files and sounds you want to read in, and other defaults regarding style and video quality.

Look through the example scenes to get a sense of how it is used, and feel free to look through the code behind 3blue1brown videos for a much larger set of example. Note, however, that developments are often made to the library without considering backwards compatibility with those old videos. To run an old project with a guarantee that it will work, you will have to go back to the commit which completed that project.

Documentation
Documentation is in progress at 3b1b.github.io/manim. And there is also a Chinese version maintained by @manim-kindergarten: docs.manim.org.cn (in Chinese).

manim-kindergarten wrote and collected some useful extra classes and some codes of videos in manim_sandbox repo.

Contributing
Is always welcome. As mentioned above, the community edition has the most active ecosystem for contributions, with testing and continuous integration, but pull requests are welcome here too. Please explain the motivation for a given change and examples of its effect.

License
This project falls under the MIT license.
