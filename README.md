Running instructions

This repo uses all the same dependencies as the hb-song-analysis repo, use the instructions here to install them: https://github.com/tbergama/hb-song-analysis/blob/master/README.md

Once all dependencies are set, use the terminal to navigate to the directory parser.py is stored in (default directory name is /user/parser) and use the command, "python parser.py".

Currently, tester.py is a modified version of the visualize.py module found in the hb-song-analysis library. It contains two functions:
- spectrogram: generates the entire spectrogram of a .wav file
- sgraph: creates a snippit of the spectrogram using passed values from the .wave file's associated .txt file

Once the testing is done, we will move the sgraph function into parser.py, upload it to the hb-song-analysis repo, and call the spectrogram function from visualize.py via an import command.



