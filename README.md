# Easygen

EasyGen is a visual user interface to help set up simple neural network generation tasks.

There are a number of neural network frameworks (Tensorflow, TFLearn, Keras, PyTorch, etc.) that implement standard algorithms for generating text (e.g., recurrent neural networks, sequence2sequence) and images (e.g., generative adversarial networks). They require some familairity with coding. Beyond that, just running a simple experiment may require a complex set of steps to clean and prepare the data.

For example: to use a neural network to generate a history for a fictional country, I would:
1. Get superhero names from somewhere. If it is Wikipedia, we see a lot of names are followed by parentheticals.
2. Strip off parenthetical information.
3. Remove empty lines and strip off any whitespace at the end of lines.
4. Feed the resultant file into a character-rnn algorithm.
5. Run the resultant model and store the outputs to a file.

EasyGen allows one to quickly set up the data cleaning and neural network training pipeline using a graphical user interface and a number of self-contained "modules" that implement stardard data preparation routines. EasyGen differs from other neural network user interfaces in that it doesn't focus on the graphical instantiation of the neural network itself. Instead, it provides an easy to use way to instantiate some of the most common neural network algorithms used for generation. EasyGen focuses on the data preparation.

Here is how EasyGen can be used to run the superhero name generation example above:
![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/lstm.png "Screenshot of EasyGen setup to run an LSTM")
Each box is a chunk of standard code that can be selected from a library of options. Arrows indicate the data flow from module to module.
 
# Preliminaries

What is a neural network? A neural network is a machine learning algorithm very loosely inspired by the brain. I'm not a fan of the brain metaphor (see [introduction to neural networks without the brain metaphor](https://medium.com/@mark_riedl/introduction-to-neural-nets-without-the-brain-metaphor-874e7950bca0)).

What you need to know is that a neural network is an automated function approximation technique. If you remember back to grade school, your teacher might have asked you to solve a problem like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/function.png "A function")

That is, given the inputs and outputs, guess the function that turns the inputs into outputs. Probably you could figure out the exact function (`y = 2x + 1`). How did you do it? You might have guessed a function based on the first example, like `y = x + 2` and then checked to see if it worked on the next example. This guess won't work: `x=2` gives `y=4`, which is off by 1. So you probably updated your guess to `y = 2x + 1`. Now that example is right. What about the next example? Yup. What about the previous example? Yup.

That is more or less the process that a neural network goes through to approximate a function. Except it doesn't just work on algebra equations. It turns out that one can set up a graph of nodes and edges that connect inputs to outputs and allow information to flow through the box. If the weights on the edges are designed correctly, then the box will generate the right answer for any given input. 

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/neural-net.png "A conceptual diagram of a neural network")

Neural network training is a process of constructing a lot of examples of inputs and outputs and running them through the network graph. When the the network gives the wrong answer, some of the weights are fiddled with so that the answer is closer to the desired output. Then you run the examples through the network again (each time you do this is called an *epoch*) to make sure the weight fiddling didn't screw anything up. Do this enough times and the weights will converge on some numbers that make the network produce the right answer (or close to the right answer) more often than not.

Now that the network has been trained--that is, we have a really good guess about the function that might have generated the data--we can apply run the neural network on inputs we have never seen before and have some confidence that it does the right thing.

There are a lot of ways to set up the neural network based on what you want to do.

If your x's are values from radar sensors on a car and your y's are steering instructions that humans perform at the same time that the radar sensor information comes in, then one can train a neural network that approximates the "human driver" function.

![alt text](https://cdn-images-1.medium.com/max/2000/1*deKGPUvHCy9nbIw-J7QOoQ.png "A neural network for driving cars.")

(Example from [here](https://medium.com/@mark_riedl/introduction-to-neural-nets-without-the-brain-metaphor-874e7950bca0))

If your x's are words from sentences in one language and your y's are words from corresponding sentences in another language, then you can do language translation.

![](https://devblogs.nvidia.com/wp-content/uploads/2015/06/Figure2_NMT_system.png)

If your x's are a string of characters and your y's are the same string of characters but offset by one, then you can perform prediction. That is, given a string of `n` characters, predict the `n+1`th character. 

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/char-rnn.png "A recurrent neural network")

That last one sounds like a really strange thing to do. But if you start with some random characters and your neural network was trained on data that comprised superhero names, then the network try to predict the superhero name that would start with those random characters.

# Installation

1. Install Python 2.7

2. Clone or download this repository.

   `git clone https://github.com/markriedl/easygen.git`

3. Install Tensorflow 0.12

   `pip install -I tensorflow==0.12.1`

   If you are using a GPU, follow these [Instructions](https://www.tensorflow.org/versions/r0.12/get_started/os_setup)

4. Install TFLearn 0.2.1

   `pip install -I tflearn==0.2.1`

5. Install Beautiful Soup 4

   `pip install beautifulsoup4`

If you will be pulling data from Wikipedia, you must do the following:

6. Download an English language [Wikipedia dump](https://dumps.wikimedia.org/enwiki/). From this link you will find a file named something like "enwiki-20170401-pages-articles-multistream.xml.bz2". Make sure you download the .bz2 file that is not the index file.

7. Unzip the bz2 file to extract the .xml file.

8. Run a script to extract the information in the Wikipedia dump:
   
   `python wikiextractor/wikiextractor.py -o wiki --json --html --lists -s enwiki-...xml`

   Be sure to fill in the exact path and name to the XML file. This will create a directory called "wiki".


# Examples

1. Superhero name generator
   
   [![Superhero generator video](http://img.youtube.com/vi/o_7hnZXR51Q/0.jpg)](http://www.youtube.com/watch?v=o_7hnZXR51Q)

# Tutorial

## Using the Editor

First, open editor.html in a web browser. Typically you can just double-click on editor.html and it will open in your default browser.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step1.png "Empty editor")

You will see a row of numbers across the top of the browser window. Mouse over them. When you do so, they will highlight green empty boxes. These are *columns*. They are where we will put stuff. They will help us keep things organized, although it doesn't matter where things are put.

Right-click on the "1". You will see a menu appear. The menu items are *modules*. Modules are chunks of code that you can arrange into a program.

Pick *ReadTextFile*. You will see a box appear in the first column called "ReadTextFile1". You have indicated that you want this module in your program. Click on the plus sign in the corner of the module. Now you should see that the module has gotten bigger and has some more boxes inside. Like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step2.png "Editor with one module")

Modules have three types of components:

- White boxes: parameters that can be set. 
- Green boxes: accept inputs from other modules.
- Red boxes: send output to other modules.

*ReadTextFile* loads a text file so that the data inside can be used by other modules later. Click on the white "file" box. A pop-up window will ask you where the file is located that you want to use. Enter `datasets/superheroes` and press "ok".

Let's add another module. Right-click on the "2" column to get another menu of modules. This time, pick "RandomizeLines". Go ahead and click the plus on this new module to expand it.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step3.png "Editor with two modules")

"RandomizeLines" takes all the lines in some text data and randomizes the order of those lines. If we want to randomize all the lines in the superheroes data, we need to indicate that we want the data that we read in from the text file in the first module to be used by the second module. To do this, click on the red "output" box inside the "ReadTextFile1" module and drag an arrow over to the green "input" box inside the "RandomizeLines1" module. It should look like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step4.png "Connecting modules")

Once the data has been randomize, it needs to go somewhere. Right-click on the "3" column and pick the "WriteTextFile" menu option. Expand the new module.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step5.png "Three modules")

"WriteTextFile" takes some text data and writes it to a new file. Connect the output from "RandomizeLines1" to the input of "WriteTextFile". It should look like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step6.png "Three modules connected")

Click on the white "file" box in "WriteTextFile" and enter `randomized_superheroes` into the pop-up box.

Okay, now we are ready to run our program, which is covered in the next section.

By the way, you can drag modules from column to column. It doesn't matter where the modules are, as long as the arrows move the data in the right order.

If you make a mistake and need to start over, you can just refresh the browser.

## Running EasyGen Programs

Unfortunately, at this stage of development of EasyGen, I have not implemented an easy way to run programs designed in the editor. Sorry :-(

Push the "Done" button in the editor. You will see a pop-up box with a bunch of code-like jibberish. That is a text-version of the program you just designed. It is in a special format called a "JSON" file. But that isn't important.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step7.png "JSON text")

Open up a text editor program. On a Mac, I use a program called "TextEdit".

Cut and paste the JSON jibberish from the editor pop-up window into a blank document in your text editor. Save the text file as `superhero_program`. Be sure to save it in the EasyGen directory so it is easy to find later (note that your text editor may default to saving the file to `superhero_program.txt`. That is okay, just substitute that filename in subsequent steps in this tutorial.)

Now you are going to need to open a terminal window. Depending on your computer's operating system, this will be done in different ways. On my computer, the terminal screen looks like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step8.png "A terminal window")

We need to change the working directory of the terminal window to the directory where you installed EasyGen. On my computer it is in the "EasyGen" directory on my Desktop, so I would do this `cd Desktop/easygen/`. (The `cd` means change directory).

Now to finally run your program: type `python easygen.py superhero_program` into the terminal and press Enter.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step9.png "A terminal ready to run")

A bunch of text will be printed to the terminal. You can ignore that. Once the program is done, you can verify that it worked. A new file should have been created called "randomized_superheroes". Open it up and look at it. You can compare it to the original "superheroes" file in the "datasets" directory. They should be different.

## Train a Neural Network

You probably want to train and run a neural network.

Go back to the editor and close the pop-up window. We can just keep building on what we have done so far.

In column "4", add a "CharacterLSTM_Train" module. The CharacterLSTM_Train module teaches a neural network to predict the next character in text data. That is, when it sees certain characters in the superheroes text file, it will try to learn what character should go next. When we run the neural net later, it will try to construct new superhero names character by character. But first things first.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step10.png "LSTM module in editor")

CharacterLSTM_Train is significantly more complicated than the other modules we've seen so far. There are a lot of settings. they all have defaults, so you don't have to do anything. But from trial and error I happen to know that some settings can make the neural network work better.

- Click on the "history" box and change the value to `5`. The history is how much text to keep in memory when trying to predict the next character. That is, it will look at the most recent 5 characters when trying to guess what the 6th character should be. 

- Click on the "layers" box. You can keep this number as `2`. The layers parameter determines how deep the neural network will be. The more layers the more complex the patterns are that can be learned. But more layers means a bigger neural network that will take longer to train and require more data.

- Click on the "hidden_nodes" box and change the value to `64`. The hidden_nodes setting indicates how many nodes will be in the neural network. If layers is depth, think of hidden_nodes as width. A wider neural network may be able to learn more patterns but will take longer to train and require more data.

- Click on the "epochs" box and change teh value to `1000`. The epochs setting tells the neural network how many times to look at the data. The more times it looks at the data, generally the more accurate the neural network gets (but also the longer it takes). This dataset is small, so it is harder to find coherent patterns. I found that this dataset works better to have a lot of epochs.

Let's send our dataset into the neural network training module. Connect the red "output" box in "Randomizedline1" to the green "data" box inside "CharacterLSTM_Train".

It is okay that there are two arrows coming out of "RandomizeLines". The program will write the randomized data to file and also use the randomized data to train the neural network.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step11.png "Connecting the LSTM module")


Once the neural network is trained, it must be run. The trained neural network is output as the "model". When the model was created, it converted all the characters into numbers. Neural networks don't really understand anything other than numbers. The "dictionary" is a file that remembers which numbers are mapped to which characters. It's just one of those things.

In column "5", create a "CharacterLSTM_Run" module. Connect the red "model" in the training module to the green "model" in the "CharacterLSTM_Run" module. Do the same for the "dictionary". It should look like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step12.png "Connecting LSTM training to LSTM running")

Inside the "CharacterLSTM_Run" module, the "history", "layers", and "hidden_nodes" settings must all be the same as those in the "CharacterLSTM_Train" module. This is a pain in the butt. But there is a shortcut. If you drag from the white "history" box in "CharacterLSTM_Train" to the white "history" box in "CharacterLSTM_Run", the editor will set the values to be the same. Furthermore, it will remember that these values should always be the same, so that if you change one later, all will change to stay the same. This is indicated by dashed lines between the white boxes.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step13.png "Connecting settings")

Two more settings still need to be set in "CharacterLSTM_Run":

- Click on the "steps" box. The value of `600` is okay and stay the same. This is how many characters to generate when the neural network is run. The bigger the number, them more characters (and thus more names) you will get.

- Click on the "temperature" box. The default values is `0.5`, which can stay the same. The temperature setting will be a number between `0.0` and `1.0`. A higher temperature means the neural network will make more risky choices and the output will be a bit weirder. A lower temperature means the neural network will try to generate things that look more like the original data if possible.

There is one more input to "CharacterLSTM_Run" that we have to deal with. The "seed" is a random sequence of characters to get the neural network started. It works best when the seed is a random sequence pulled from the original data. We can use the "RandomSequence" module. Let's add that module to column "3" (although it can go anywhere you want I like to keep things flowing from a left-to-right fashion). You can have more than one module in a column. You can always drag the other modules into other columns if need to make room.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step14.png "Randomizing a sequence")

Connect the output of the "RandomizeLines" to the input of "RandomSequence". Connect the output of "RandomSequence" to the green "seed" of "CharacterLSTM_Run". It should look like this:

The "length" inside the "RandomSequence" module can be anything, although I like to set it to be the same as "history". You can use the dashed-line trick. 

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step15.png "Running the network is ready to go")

Finally, we need the generated data from the neural network to go somewhere. Add another "WriteTextFile" module to column 6. Connect the output of CharacterLSTM_Run to the input of WriteTextFile2 in column 6. Like this:

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step16.png "The complete program")

Change the "file" setting to `new_superheroes`.

You are all set. Follow the instructions above to save your program and run it in the terminal.




# Documentation