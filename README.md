# Easygen

[EasyGen runs on Google Colaboratory](https://drive.google.com/open?id=1XNiOuNtMnItl5CPGvRjEvj9C78nDuvXj).

EasyGen is a visual user interface to help set up simple neural network generation tasks.

There are a number of neural network frameworks (Tensorflow, TFLearn, Keras, PyTorch, etc.) that implement standard algorithms for generating text (e.g., recurrent neural networks, sequence2sequence) and images (e.g., generative adversarial networks). They require some familairity with coding. Beyond that, just running a simple experiment may require a complex set of steps to clean and prepare the data.

For example: to use a neural network to generate imaginary superhero origin stories, I would:

1. Get superhero origin stories from somewhere. Wikipedia is a good source, but there might not be any simple list of characters or collection of origin stories. I'd have to find every wikipedia article tagged with "superhero" as a category, find the part of each article that gives the origin story, strip that text from the articles. 

2. If it is Wikipedia, we are going to see a lot of html tags in the raw text of the articles, as well as empty lines (paragraph breaks), and special characters.

3. Remove html tags.

4. Remove special characters.

5. Remove empty lines.

6. Remove periods from abbreviations such as dr. and p.h.d. because that will mess up my ability to find sentence breaks.

7. Put every sentence on its own line.

8. Make sure each word is separated by a whitespace and separate conjunctions (e.g. "doctor's" becomes "doctor" "'s"). This usually helps the neural network learn.

9. Make all words lowercase. This usually helps the neural network learn.

10. Feed the resultant file into a character-rnn algorithm to train a neural network model.

11. Save the neural network model because I might want to use it again later.

12. Run the resultant model, seeded on a random valid sequence from the training data.

13. Write the output of running the model to a text file.

that's a lot of code to write, just to get the data ready for the neural network!

EasyGen allows one to quickly set up the data cleaning and neural network training pipeline using a graphical user interface and a number of self-contained "modules" that implement stardard data preparation routines. EasyGen differs from other neural network user interfaces in that it doesn't focus on the graphical instantiation of the neural network itself. Instead, it provides an easy to use way to instantiate some of the most common neural network algorithms used for generation. EasyGen focuses on the data preparation.

Here is how EasyGen can be used to run the superhero origin story generation example above:
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


# Running EasyGen

We provide a Jupyter notebook hosted on [Google Colaboratory](https://colab.research.google.com/), which provides access to GPUs.

1. Clone the [EasyGen notebook](https://drive.google.com/open?id=1XNiOuNtMnItl5CPGvRjEvj9C78nDuvXj) by following the link and selecting File -> Save a copy in Drive.

2. Turn on GPU support under Edit -> Notebook setting.

3. Run the cells in Sections 1. Some are optional if you know you aren't going to be using particular features.

4. Run the cell in Section 2. If you know there are any models or datasets that you won't be using you can skip them.

5. Run the cell in Section 3. This creates a blank area below the cell in which you can use the buttons to create your visual program. An example program is loaded automatically. You can clear it with the "clear" button below it. Afterwards you can create your own programs. Selecting "Make New Module" will cause the new module appears graphically above and can be dragged around. The inputs and outputs of different modules can be connected together by clicking on an output (red) and dragging to an input (green). Gray boxes are parameters that can be edited. Clicking on a gray box causes a text input field to appear at the bottom of the editing area, just above the "Make New Module" controls.

6. Save your program by entering a program name and pressing the "save" button.

7. Run your program by editing the program name in the cell in Section 4 and then running the cell.

# Example Programs

Example programs are in the easygen/examples directory.

| Program Name | Description |
| ------------ | ----------- | 
| make_new_colors | Given a json of paint names, train a recurrent neural network to generate new color names. This program demonstrates some of the text file pre-processing and post-processing modules that are necessary to prepare data for a neural network and to handle the results of a neural network. |
| make_superheroes | Crawls Wikipedia looking for superhero and supervillain names. Trains a neural network to generate new superhero and supervillain names. |
| make_new_curses | Crawl a webpage of English vulgarities and train a neural network. |
| star_trek_novels | Crawl lists of Star Trek books and Romance books, merge the data, and generate new Star Trek novel titles. This shows how to produce more interesting outputs by blending datasets. |
| make_cat_movie | Use a pretrained StyleGAN model trained on cat images to generate an animated video of cats morphing into each other. Demonstrates a simple use of StyleGAN. |
| 10xcats | Fine-tune a pretrained StyleGAN model trained on cats to create scary looking cats. Demonstrates StyleGAN fine-tuning and also how cutting training short can create blended art. |

# Tutorial

Let's walk through an example of creating superhero origin stories. Start by executing steps 1-4 above. Be sure to run the optional step of downloading Wikipedia. Run the cell in section 3 to start the GUI. Clear out the pre-loaded example. You should have a blank canvas.


1. Make a new *ReadWikipedia* module. To do this find the place where it says "Make New Module". Select *ReadWikipedia* and press "Add Module". You will see the module appear in the upper left. The *ReadWikipedia* module scans through Wikipedia articles looking for articles (and sections of articles) that match certain criteria, to be specified next.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step1-1.png "Make New Module")

You should see a graphical representation of your module in the blank area above. Green boxes indicate module inputs. Red boxes indicate module outputs. Gray boxes indicate parameters that you must specify.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step1-2.png "ReadWikipedia Module")


2. Parameterize the *ReadWikipedia* module. The gray boxes indicate parameters that you need to fill out. Click on *wiki_directory*. At the bottom, you will see a new text entry form appear that says "Read Wiki (1)" followed by "wiki_directory". Enter "wiki" (without quotes) into the text box and hit the "ok" button. The text entry form will disappear.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step2.png "Parameterizing ReadWikipedia")


3. Do the same for the *pattern* parameter in *ReadWikipedia*. This time enter "*:origin|origins|biography" in the text entry form that appears. This tells the module to accept all titles "*", then look for first-level headers that have the words "origin" or "origins" or "biography" in them. The ":" indicates the order of headers with the first header being an article title and the first-level headers specification directly after the colon. The "\|" indicates "or".

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step3.png "Parameterizing ReadWikipedia")

4. Further refine *ReadWikipedia* by filling in a specification for the *categories* parameter. Enter "superhero\|superheroes" into the text entry form.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step4.png "Parameterizing ReadWikipedia")

You have just specified how to get data out of Wikipedia. The *ReadWikipedia* module compiles this data into two outputs, indicated by red boxes. The first output is *out_file*, which contains all the text that was stripped from Wikipedia. Each chunk of text stripped out of a Wikipedia article will be placed into a text file, with a special indicator "<EOS>" on a line by itself to indicate the end of each sequence. 

The second output is *titles_file*, which contains the title of each Wikipedia article that contributed to the *out_file* data.

The text that is stripped from Wikipedia articles is pretty raw. It will contains paragraphs of text, empty lines between paragraphs, html tags, and special characters that we don't want in our neural network training data. We are going to need more modules.

5. Make a *RemoveEmptyLines* module. Select "RemoveEmptyLines" from the drop-down box underneath "Make New Module" and press the "ok" button. You will see another module show up just to the right of *ReadWikipedia*. This module takes a single input file and produces a single output file.



6. Connect *ReadWikipedia* outputs to *RemoveEmptyLines* inputs. To indicate that the output data from *ReadWikipeda* should be sent to the input of *RemoveEmptyLines* click and drag on the red *out_file* box under *ReadWikipedia*. You will see a line from the box to your cursor. Drag the line to the green *input* box under *RemoveEmptyLines* and release the mouse button.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step6.png "Connecting ReadWikipedia to RemoveEmptyLines")


7. Remove html tags from the data and do some other cleaning tasks. Make a *CleanText* module. Connect the output of *RemoveEmptyLines* to the input of *CleanText*. The *CleanText* module removes html tags, removes periods from common abbreviations such as "Mr.", and does a few other convenient things that I find myself doing a lot.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step7.png "CleanText")


8. Make words lowercase. Make a *MakeLowercase* module. Connect the output of *CleanText* to the input of *MakeLowercase*. By making the text lowercase, the neural network will not recognize uppercase and lowercase versions of the same word as distinct.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step8.png "MakeLowercase")


9. Make a *Wordify* module. Connect the output of *MakeLowercase* to the input of *Wordify*. The *Wordify* module makes sure all words are separated by at least one whitespace. It also splits "'s" from plural words so that it recognizes the roots regardless of pluralization. Punctuation is separated by a whitespace as well so that punctuation is recognized as distinct from words.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step9.png "Wordify")

10. Make the module to train the neural network. Make the *WordRNN_Train* module. This module assumes the data is made up of words. It produces a neural network module as an output and a "dictionary". A dictionary is an assignment of words to numbers because neural networks like numbers but not symbols, which is why we convert every unique word into a unique number. The default parameters should be fine. Connect the output of *Wordify* to the input of *WordRNN_Train*.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step10.png "WordRNN_Train")


11. Run the neural network model to create new text. Make a *WordRNN_Run* module. This module will generate text from the neural network once it has been trained. Connect the model output of *WordRNN_Train* to the model input of *WordRNN_Run*. Connect the dictionary output of *WordRNN_Train* to the dictionary input of *WordRNN_Run*.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step11.png "WordRNN_Run")

We have a problem: *WordRNN_Run* takes a third input, called the "seed". This is a piece of text that gets the generation process started off. The easiest way to create a seed is to use a *MakeString* module that lets you enter your own text string. Other way is to get a random sequence of words from the training data.

12. Create a *RandomSequence* module. Connect the output of *Wordify* to the input of *RandomSequence*. Connect the output of *RandomSequence* to the seed input of *WordRNN_Run*.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step12.png "RandomSequence")

13. Write the generated text to file. Make a *WriteTextFile* module. Connect the output of *WordRNN_Run* to the input of *WriteTextFile*.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step13.png "WriteTextFile")

14. Give the output file a name. Modify the "file" parameter of *WriteTextFile* and change the file name to "my_output_file" (without quote marks).

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step14.png "Parameterizing WriteTextFile")

Your program is now complete. You may want to additionally save the cleaned up date to file using *WriteTextFile* connected to the output of *Wordify*. You may additionally want to save the neural network model and the dictionary after training using *SaveModel* and *SaveDictionary* respectively.

15. Save your program. To save your program scroll down to the bottom left of the program editing area. Enter the program name into the text entry area under "Save Program". For this tutorial use the name "my_program" (without quote marks) and press the "save" button.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step15-1.png "Saving the program")

You can see your file if you use the tab on the left of the browser window (look for the little arrow). If you expand the panel and select "files" you should see your program in the file list.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step15-2.png "File list")

16. Run the program. Find the code cell in section 3 of the notebook. Edit the name of your program if necessary (it should have the default "my_program"). Run the cell.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step16.png "Running the program")

This will take a while. You will see a lot of text output showing you what *ReadWikipedia* is scraping. Later you will see *WordRNN_Train* reporting stats on the neural network training.

17. View your final output. When the program is done running, it should have saved a file called "my_output_file". Edit the name of your output file if neccesary (it should have the default "my_output_file"). Run the cell.

![alt text](https://raw.githubusercontent.com/markriedl/easygen/master/web/step17.png "Printing output")


# Documentation

## ReadWikipedia

This module parses content out of Wikipedia. It requires a download of a English Wikipedia dump (see installation instructions above).

**Inputs:**

None.

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| wiki_directory | directory path | This is the directory that the English Wikipedia dump was extracted into. | None |
| pattern | string | Describes what should be extracted using a special pattern language (described below). | "*" |
| categories | string | Describes what categories you are looking for (described below) | "*"

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| out_file  | text data | The text mined from Wikipedia. May have many sentences per line. May contain blank lines. Each Wikipedia entry scraped is delimited by "<EOS>". |
| titles_file | text data | A list of Wikipedia articles from which data was mined. Each title is on a separate line. |

**Patterns:**

The pattern takes a specially formatted text string as follows: `title_keywords:header1_keywords:header2_keywords:...`. The `title_keywords` is a set of keywords to look for in Wikipedia article titles. If the keyword is "*", then all titles match. Any word in title_keywords must be present in the article title. To look for different possibilities, use the "|" to indicate that any of two or more words could be matched.

For example: "cat|dog" will match any Wikipedia article with the words "cat" or "dog" in the title.

If the title matches, then the article is checked to see if it has a 1st-level header that matches the `header1_keywords`. If no header1_keywords are given or if the header1_keywords are "*", then all the text underneath that header are grabbed. As with the title, you can use "|" to indicate different possible matches to look for.

For example: "cat|dog:evolution|origin" will find any article with the words "cat" or "dog" in the title and then look for any 1st-level header inside those articles that contain the words "evolution" or "origin". Whatever text is contained beneath those headers (even if there are 2nd or 3rd level headers).
 
If the title keywords match and the article contains a 1st-level header that matches header1_keywords and `header2_keywords` is given, then the article must also have a 2nd-level header underneath a matching 1st-level that maches the header2_keywords. If all of this happens, then all text underneath the 2nd-level header is grabbed.

More levels can be provided.

(Advanced feature: If any keyword is replaced with the word "list", then the module will only retrieve the text contained in lists at the specified level. For example `*:*:taxonomy:list` would retrieve any lists beneath the 2nd-level header.)

**Categories:**

All articles have a list of categories. They can usually be found at the bottom of article pages. The categories setting takes a list of keywords separated by the "|" symbol, or "*" to indicate all categories.

This can be used in conjunction with patterns for more control over article matching. When categories are used, any Wikipedia article must first match one of the keywords within one of its categories. After the category match, then it proceeds with the pattern matches in title and headers.

For example: "domesticated|vertibrates" will only try to match the above patterns when a document has been tagged with a category that contains the word "domesticated" or "vertibrates".




## ReadTextFile

The ReadTextFile module is used for loading data into the program from a text file. ReadTextFile makes no assumptions about how the text file is formatted. It has no inputs other than the name of the file, so makes a good module to start a program with. The intended use is that modules will take the output and format it in preparation for use by a neural network, stripping out characters, deleting blank lines, breaking it up into separate datasets, etc. 

**Inputs:**

None

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| file | directory path | This is the directory path to a text file that should be loaded in. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The exact text data from the specified file. |

## WriteTextFile

The WriteTextFile module is used for saving data from the program from a text file. WriteTextFile makes no assumptions about how the text file is formatted. It has no outputs, so makes a good module to end a program with. For example, the outputs of a neural network could be saved.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data to be written to file as is. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| file | directory path | This is the directory path to a text file that should be saved to. | None |

**Outputs:**

None

## CharRNN_Train

A character-level recurrent neural network for generating text, one character at a time. This module trains the neural network on a chunk of text data. The model predicts the next character based on the previous characters it has seen. This module is paired with `CharRNN_Run` which can take the model produced by this module and use to do the actual text generation.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| data | text data | The text data to train the network with. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| history | int | This is how many characters of the input data are held in memory at any given time. The more characters in memory, the more like the neural network will recognize long-range relationships. | 35 |
| layers | int | Number of layers in the network/ how deep the network should be | 2 |
| hidden_nodes | int | How many nodes in each layer of the network? | 512 |
| epochs | int | How many times should the network look at the data/ how long the network should train for | 50 |
| learning_rate | float| A parameter for how fast neural networks are allowed to change. Smaller means slower, but possibly more accurate training. | 0.0001 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The model learned by the neural network. |
| dictionary | dictionary | A mapping from characters to numbers. |


## CharRNN_Run

A character-level recurrent neural network for generating text, one character at a time. A model trained with `CharRNN_Train` can be used to generate text. The model and dictionary from `CharRNN_Train` should be connected to the inputs of this module. 

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The model learned by the neural network. |
| dictionary | dictionary | A mapping from characters to numbers. |
| seed | text data | A string of text to get the neural network started. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| steps | int | How many characters to generate. | 600 |
| temperature | decimal between 0.0 and 0.1 | How risky the generation should be. 0.0 means try to replicate the data as well as possible. 1.0 means make a lot of risky moves. | 0.5 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text generated by running the neural network. |

The seed can come from the original data set that was fed into `CharRNN_Train`. A typical thing to do is to grab a random string from the original dataset because the neural network will be happy starting out with something it recognizes. One can use `RandomSequence` to grab a random string from the original text data. Alternatively, one could use `MakeString` or `UserInput`.  

The temperature parameters affects how random-appearing the generated text will be. A value close to 1.0 will make the output look more random because the neural network will not always take the most likely next character. A value close to 0.0 will make output that looks more like the original input data (assuming the model is properly trained) because it will always take the most likely next character given a history.

## WordRNN_Train

A word-level recurrent neural network for generating text, one word at a time. This module trains the neural network on a chunk of text data. The model predicts the next word based on the previous words it has seen. This module is paired with `WordRNN_Run` which can take the model produced by this module and use to do the actual text generation.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| data | text data | The text data to train the network with. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| history | int | This is how many words of the input data are held in memory at any given time. The more words in memory, the more like the neural network will recognize long-range relationships. | 35 |
| layers | int | Number of layers in the network/ how deep the network should be | 2 |
| hidden_nodes | int | How many nodes in each layer of the network? | 512 |
| epochs | int | How many times should the network look at the data/ how long the network should train for | 50 |
| learning_rate | float| A parameter for how fast neural networks are allowed to change. Smaller means slower, but possibly more accurate training. | 0.0001 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The model learned by the neural network. |
| dictionary | dictionary | A mapping from words to numbers. |

Typically, you will want to run `Wordify` on any input data.

## WordRNN_Run

A word-level recurrent neural network for generating text, one word at a time. A model trained with `WordRNN_Train` can be used to generate text. The model and dictionary from `WordRNN_Train` should be connected to the inputs of this module. 

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The model learned by the neural network. |
| dictionary | dictionary | A mapping from characters to numbers. |
| seed | text data | A string of text to get the neural network started. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| steps | int | How many words to generate. | 600 |
| temperature | decimal between 0.0 and 0.1 | How risky the generation should be. 0.0 means try to replicate the data as well as possible. 1.0 means make a lot of risky moves. | 0.5 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text generated by running the neural network. |

The seed can come from the original data set that was fed into `WordRNN_Train`. A typical thing to do is to grab a random string from the original dataset because the neural network will be happy starting out with something it recognizes. One can use `RandomSequence` to grab a random string from the original text data. Alternatively, one could use `MakeString` or `UserInput`.  

The temperature parameters affects how random-appearing the generated text will be. A value close to 1.0 will make the output look more random because the neural network will not always take the most likely next character. A value close to 0.0 will make output that looks more like the original input data (assuming the model is properly trained) because it will always take the most likely next character given a history.

## GPT2_Load

This module loads a pre-trained GPT-2 language model. The GPT-2 language model is trained on a very large corpus of text scraped from the internet. It can be used to generate plausible looking text from a seed. It can also be fine tuned on new data to generate text that looks more like the new data provided to it. This module loads the GPT-2 small (117M parameter) or medium (345M parameter) model and prepares it for running or fine tuning. This module is thus paired with `GPT2_Run` and `GPT2_FineTune`.

**Inputs:**

None


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| model_size | string | The size of the model: "117M" or "345M". | "117M" |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The pre-trained model that was downloaded. | 

## GPT2_Run

Generate text from the GPT-2 model. Paired with `GPT2_Load` and/or `GPT2_FineTune`.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model_in  | neural network model | The model learned by the neural network. |
| prompt | text data | A string of text to get the neural network started. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| model_size | string | The size of the model: "117M" or "345M". | "117M" |
| top_k | int | How much variance to allow during generation | 40 |
| temperature | decimal between 0.0 and 0.1 | How risky the generation should be. 0.0 means try to replicate the data as well as possible. 1.0 means make a lot of risky moves. | 1.0 |
| num_samples | int | How many times to generate? Each will be concatenated to the same output text | 1 |


**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text generated by running the neural network. |

## GPT2_FineTune

## GPT2_Run

Fine tune the GPT-2 model on new data. Paired with `GPT2_Load` and `GPT2_Run`.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model_in  | neural network model | The model learned by the neural network. |
| data | text data | The new data. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| model_size | string | The size of the model: "117M" or "345M". | "117M" |
| steps | int | How many times should the neural network look at the new data to adjust weights | 1000 |


**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model_out  | neural network model | The fine-tuned model.  |

## RandomSequence

This module takes some text data and grabs a random chunk of it, discarding the rest. The random chunk could be from anywhere in the text data, or it could always start at the start of a random line. The most typical use of this module is to create a seed from text data to be used in `CharRNN_Run` or `WordRNN_Run`.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | Some text data. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| length | int | This is how many characters to grab from the text data. | 100 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The random sequence. |


## MakeString

Hard code a text string to be sent into other modules. This is kind of like making a string variable in other programming languages.

**Inputs:**

None.

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| string | string | The characters to save in the string data. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text of the string to be passed along to other modules. |

## UserInput

Pause the program and ask the user to enter some text in the terminal.

**Inputs:**

None.

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| prompt | string | The prompt to be printed out before the user enters some text. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text of the string to be passed along to other modules. |


## SplitSentences

This model takes text data and breaks it up into individual lines when it seed end-of-sentence punctuation (e.g., ".", "!", "?", or ":").

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | This is the data that you want split up into individual lines, one per sentence. |


**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The data with each sentence on its own line. |

## RemoveEmptyLines

Takes text data and deletes any lines that are empty.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | This is the data. |


**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The data but with no empty lines. |


## StripLines

Takes text data and deletes any whitespace from the front and end of each line.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | This is the data. |


**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The data but with no whitespace characters on the front or end of any line. |

## ReplaceCharacters

This module looks for subsequences of characters in text data and replaces it with other subsequences. Like Find and Replace All in your favorite text editor.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| find | string | The string subsequence to find and replace. | None |
| replace | string | The string subsequence to replace with. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The data will all occurrences of `find` replaced with `replace`. |

You can remove characters from text data by leaving the `replace` parameter blank.

## SplitLines

For each line in text data, if it contains a character (or string subsequence), split that line into two and put the first part in one output file and the second part in another output file. This module only splits on the first instance of the character (or subsequence).

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| character | string | The character (or string subsequence) to look for on each line to make the split at. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output1  | text data | The data containing lines with text from before the split (or complete lines if the split character wasn't found). |
| output2  | text data | The data containing lines with text from after the split (or empty lines if the split character wasn't found). |

## ConcatenateTextFiles

Take two text data files and merge them together.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input_1  | text data | The text data from the first file. |
| input_2  | text data | The text data from the second file. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The data containing all lines from both inputs. |

## RandomizeLines

This module takes text data and randomizes the lines. This is useful for training sometimes.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data to be randomized. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The randomized data. |

## RemoveTags

This module removes HTML (and XML) tags from text data. This leaves the text between tags intact.

 **Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text data without HTML and XML tags. |

## CleanText

This module is a collection of text cleaning routines. (1) removes HTML (and XML) tags from text data; (2) removes special HTML characters; (3) removes periods from abbreviations such as Mr.; (4) removes periods from i.e. and e.g.; (5) Moves periods inside quote marks to the outside of quote marks; (6) removes periods from acronyms; (7) removes periods in numbers.

 **Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The cleaned text data. |

## MakeLowercase

Makes all the text in a text file lowercase.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text data with no uppercase characters. |


## Wordify

This module takes text data and inserts a blank space between all words and punctuation marks. For example, it will convert "Mark's" to "Mark" and "'s". It treats "." and other punctuation as if they were words. 

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text data. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text data with words and punctuation with blank spaces between. |



## LoadModel

Load a neural network model.

**Inputs:**

None

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| file | directory path | This is the filename of the model to load. | None |
**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The model. |

The model may be saved as a group of several files all with the same start of the filename, e.g., "mymodel", "mymodel.index", "mymodel.meta", and "mymodel.data-00000-of-00001". The file name is just the common part before the period. All files will be loaded.

Note: hasn't been tested with `DCGAN`.

## SaveModel

Save a neural network model.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The model. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| file | directory path | This is the filename to save the model to. | None |

**Outputs:**

None

The model may be saved as a group of several files all with the same start of the filename, e.g., if you give the file name as "mymodel", the following files may be save: "mymodel", "mymodel.index", "mymodel.meta", and "mymodel.data-00000-of-00001".

Note: hasn't been tested with `DCGAN`.

## LoadDictionary

Load a neural network dictionary.

**Inputs:**

None

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| file | directory path | This is the filename of the dictionary to load. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| dictionary  | neural network dictionary | The dictionary. |


## SaveDictionary

Save a neural network dictionary.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| dictionary  | neural network dictionary | The dictionary. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| file | directory path | This is the filename to save the dictionary to. | None |

**Outputs:**

None

## KeepFirstLine

Discard all text data except the first line.

## DeleteFirstLine

Discard the first line of text data and pass the rest on.

## DeleteLastLine

Discard the last line of text data and pass the rest on.

## KeepLineWhen

Delete all lines of text that do not have a given sub-sequence.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| match | string | The substring to look for in each line. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text data with only lines containing the match. |

## KeepLineUnless

Delete all lines of text that have a given sub-sequence.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| match | string | The substring to look for in each line. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text data with only lines that do not contain the match. |


## MakeCountFile

This module creates a new text file where each line contains a number, counting up. The module allows text to be placed before and after the number.


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| num | int | The number of lines. | 10 |
| prefix | string | Text to place before the number on each line. | None |
| postfix | string | Text to place after the number on each line.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text data with the given number of lines and the line number on each line. |

The default usage of this module would create a new text file with each line numbered 1-10.

```
1
2
3
...
9
10
```

This module is useful in combination with `ReadAllFromWeb` because it can be used to create URLs as such:

```
https://www.barnesandnoble.com/b/books/science-fiction-fantasy/star-trek-fiction/_/N-29Z8q8Z182c?Nrpp=20&page=1
https://www.barnesandnoble.com/b/books/science-fiction-fantasy/star-trek-fiction/_/N-29Z8q8Z182c?Nrpp=20&page=2
https://www.barnesandnoble.com/b/books/science-fiction-fantasy/star-trek-fiction/_/N-29Z8q8Z182c?Nrpp=20&page=3
...
```

## ReadFromWeb

This module grabs raw HTML from a URL.

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| url | string | The URL to grab text from. Starts with http:// | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| data  | text data | The raw HTML from the web page. |

## ReadAllFromWeb

This module reads many web pages and concatenates all the text together. The URLs are given in input text data, with one url per line.


**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text containing a list of URLs, one per line. |


**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| data  | text data | The raw HTML from the web pages concatenated together. |

## Regex_Search

This module takes text data and generates new text data that includes only what matches a regular expression pattern. It supports "groups", in which sub-sequences within the matched sequence are saved out separately. Up to two groups can be specified. The module will capture all non-overlapping matches. Matches can span across lines.

[Information on writing regular expressions](http://www.rexegg.com/regex-quickstart.html)

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| expression | string | The regular expression. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text that is matched by the regular expression. |
| group_1 | text data | If one group is captured in the regular expression, the first group of captured text will be here. |
| group_2 | text data | If two groups are captured in the regular expression, the second group of captured text will be here. |

Using groups: When there are paired parentheses `( ... )` in the regular expression, then if a match occurs, the parentheses further specify that some of those characters should be pulled aside.

For example: `<a href\=[\"]?([.]+)[\"]?>([.]+)</a>` will match against anchor tags in an html file. The primary output of the module would contain the full tags. There are two groups captured in the regular expression. The first is the URL. The `group_1` output will contain a list of URLs. The second is the content between the open of the anchor tag and the close of the anchor tag. The `group_2` output will contain the text of the links.

## Regex_Sub

This module uses a regular expression to find a sequence of text and then replace it. Groups are supported. Matches can span across lines.

[Information on writing regular expressions](http://www.rexegg.com/regex-quickstart.html)

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| expression | string | The regular expression. | None |
| replace | string | What to replace any matches with. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text with any sequences matched by the regular expression replaced. |

Using groups: When there are paired parentheses `( ... )` in the regular expression, then if a match occurs, the parentheses further specify that some of those characters should be pulled aside. Groups can then be referenced in the `replace` specification using `\1` for the first group, `\2` for the second group, and so on.

For example: `<a href\=[\"]?([.]+)[\"]?>([.]+)</a>` will match against anchor tags in an html file. The first group grabs the URL and the second group will grab the text of the link. If the replace string is `\2: \1`, it will replace all anchor tags with "the link text: URL".

## RemoveDuplicates

Remove any lines that are identical to each other.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

None

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text with out any duplicate lines. |

## TextSubtract

Remove any line from text file 1 that appears in text file 2.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| main  | text data | The text to remove lines from. |
| subtract  | text data | The text lines that should not appear in the final output. |


**Parameters:**

None

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text that contains lines from *main* and no lines from *subtract*. |

## DuplicateText

Duplicate the text in a text file *count* times. 

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| count | int | The number of times to dupliacte the input text. | 1 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text with out any duplicate lines. |

If *cout=1* then the text file will be duplicated once for a total of two copies, and so on.

## Spellcheck

Check the spelling of every word in the input text and replace it with the most likely alternative word if it is misspelled. Words for which there are no obvious replacements are left misspelled.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | text data | The text. |

**Parameters:**

None

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The text with out any duplicate lines. |

## WebCrawl

Starting at a particular web page, follow a specified link, downloading raw HTML for every file encountered.

**Inputs:**

None

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| url | string | The URL at which to start. | None |
| link_id | string | If the web page uses tag ids on anchors, use this to locate the "next page" anchor | None |
| link_text | string | Locate the "next page" anchor containing this text | None |
| max_hops | int | How many times to try to follow a "next page" link. | 10 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | text data | The raw HTML of every page encountered. |

The module prefers to use the *link_id* and will only try to use *link_text* if *link_id* is not specified. 


## ScrapePinterest

Downloads images from Pinterest boards. This module requires that you have a Pinterest account and that you enter your username and password in plaintext. I don't see this information but we advised that this data is going through Google and, if you save the program, you will be saving your password in plaintext.

**Inputs:**

None.

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| url | string | A URL a Pinterest.com. | None |
| username | string | A Pinterest.com username (email address). | None |
| password | string | Password for the Pinterest.com account. | None |
| target | int | The number of images to try to retrieve. It may not be possible to retrieve that many. | 100 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A set of image files scraped from Pinterest. |

## LoadImages

Load a directory of images (or a single image).

**Inputs:**

None.

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| directory | string | A path to a directory containing image files. Can also be the path to a single image file. | None |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| images  | images | A set of image files. |

## SaveImages

Save a set of images to a given directory.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| images  | images | The set of images. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| directory | string | A path to a directory to save the image files. | None |

**Outputs:**

None.

## ResizeImages

Resize each of the images passed in to a square with a given height/width dimension. This is useful when using StyleGAN because it assumes square images (each model is trained with a specific target image dimension).

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | images | The images to resize. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| size | int | The height/width of the images. | 256 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | The resized images. |

## RemoveGrayscale

Remove grayscale images from a set of images.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | images | The images to filter. |

**Parameters:**

None.

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | The non-grayscale images. |
| rejects | images | Grayscale images that were filtered out. |

## CropFaces

Close-crop faces in the set of images. This module also resizes images to squares with the given height/width.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | images | The images to crop. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| size | int | The height/width of the images. | 256 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | The images with faces. |
| rejects | images | The images in which no faces were found. |

## StyleGAN_FineTune

Take a pretrained StyleGAN model and continue training it on a new dataset. StyleGAN models only work on square images of a certain, specified size. Make sure you have resized your images first.

**I have only tested this module on the StyleGAN pre-trained model trained on 256x256 cat images.** 

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model_in  | neural network model | The pretrained model to fine-tune. |
| images | images| The new set of images to fine-tune the model on. | 

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| start_kimg | int | The starting iteration in terms of the number of images that have been trained upon (x1000). StyleGAN uses the iteration number to determine the level of granularity to train at and moves up and down the granularity in a prescribed schedule. The closer to 0, the more that will be retrained but also the more data and more time required. | 7000 |
| max_kimg | int | The ending iteration. The larger the number, the more the neural network will overfit to the data. A number closer to ```start_kimg``` will result in blended models. | 25000 |
| seed | int | An arbitrary value. Use the same seed value to reproduce results. |
| schedule | int or string | Determines how many iterations to run at different resolutions. The string should be formatted as follows: "{4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:30, 1024:20}". The number before the colon is a resolution. The number after the colon is the number of iteration (x1000). You must have values for 4, 8, 16, 32, 64, 128, 256, 512, and 1024. Alternatively, you can provide a single integer which will be the number of iterations (x1000) for each resolution size. |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model_out  | neural network model | The fine-tuned network. |
| animation | images | The images produced at intermediate stages. You can use these to create a movie showing how the training progressed. |

Notes:
1. Intermediate results are stored in the "results" directory. You can look at the images to to see how the training is progressing.
2. You can interrupt the training at any time (press the stop cell button once) and the rest of the program will continue. This is handy if you see that the training is done before it reaches the final tick.
3. You may need to experiment with the ```start_kimg```. A smaller number will give the neural network more freedom with regard to color and shape, but it will have to re-learn more and may need a bigger images dataset. A larger number will use a lot more of what the pretrained model has pre-learned but manipulate the finer details.
4. The default schedule is ```{4: 160, 8:140, 16:120, 32:100, 64:80, 128:60, 256:40, 512:30, 1024:20}```. I find that this is too many iteration for most small fine-tuning datasets. For < 500 images a schedule of ```{4: 2, 8:2, 16:2, 32:2, 64:2, 128:2, 256:2, 512:2, 1024:2}``` seems to work well.

## StyleGAN_Run

Run a StyleGAN model to produce images.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The pretrained or fine-tuned model. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| num | int | The number of images to create. | 1 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| images  | images | The generated images. |

## StyleGAN_Movie

Create an animation that interpolates between generated images. This creates the appearance of output images that morph into each other.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| model  | neural network model | The pretrained or fine-tuned model. |

**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| length | int | The number of generated images. | 10 |
| interp | int | The number of interpolated images in between each generated image. | 10 |
| duration | int | The duration of each frame of the movie. | 10 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| movie  | images | A single animated gif. |

## MakeMovie

Take a bunch of images and stitch them into an animated GIF.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| images  | images | A set of images. Assume that they will be assembled into a movie according to filename sort order. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| length | int | The number of generated images. | 10 |
| interp | int | The number of interpolated images in between each generated image. | 10 |
| duration | int | The duration of each frame of the movie. | 10 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| movie  | images | A single animated gif. |

## Gridify

Take a set of images and arrange them into a single ```m x n``` grid image. The number of columns must be given but the number of rows will be automatically determined by the number of images fed into this module. Images will be resized into squares of equal height and width.

 **Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | images | A set of images. Assumed to all be square and all be the be same size. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| size | int | The height and width of each image. | 256 |
| columns | int | The number of columns in the final grid. | 4 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A single image that contains a grid of input images. |

## Degridify

Take a set of images, all of which are grids made up of smaller images, and break them into individual images. The number of columns and rows of each input image must be the same and known in advance.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | images | A set of images. Assumed to all be grids of the same size and with the same number of rows and columns. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| rows | int | The number of rows in the grid images. | 4 |
| columns | int | The number of columns in the grid images. | 4 |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A set of individual images. |

## StyleTransfer

Neural style transfer takes the texture from one image (the style image) and attempts to redraw a content image to maintain the original content but also satisfy the style.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| content_image  | images | An image (or set of images) to be used as content.  |
| style_image | images | An image (or set of images) to be used for style. |


**Parameters:**

| Component | Type | Description | Default |
| --------- | ---- | ----------- | ------- |
| steps | int | Number of training steps. | 1000 |
| size | int | The height and width of each image. Images will be resized automatically. | 512 |
| content_weight | int | How much weight to place on the content image. | 1000000 |
| style_weight | int | How much weight to place on the style image. | 1 |
| content_layers | string | Which layers to collect content information from. A string containing a list of numbers from 1 to 5. If the string is "-1" then all five layers will be used. | 4 |
| style_layers | string | Which layers to collect style information from. A string containing a list of numbers from 1 to 5. If the string is "-1" then all five layers will be used. | "1, 2, 3, 4, 5" | 

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A set of images with style applied to content. |

This module can accept a single content image and single style image, in which case the style is applied to the content image. One can also specify a directory containing multiple content images and a directory containing multiple style images. When directories are given, all style images are applied to all content images in a pairwise fashion producing ```m x n``` wherem ```m``` is the number of content files and ```n``` is the number of style files.

## JoinImageDirectories

Take the image files from two separate directories and merge them into a new directory. This is useful when one wants to load multiple single images and input them all into another module that takes a set of images.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| dir1  | images | An image (or directory of images).  |
| dir2 | images | An image (or directory of images). |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A set of images. |

## SquareCrop

Most of the neural network modules in EasyGen expect square images and will resize accordingly, causing distortion. This module will crop the middle out of rectangle images so that the resulting image is a square.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| input  | images | An image (or set of images).  |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A set of images that are square. |

## UnmakeMovie

Takes an animated gif and pulls out each frame into individual images.

**Inputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| movie  | images | A directory containing movies (animated gifs).  |

**Outputs:**

| Component | Type | Description |
| --------- | ---- | ----------- |
| output  | images | A set of images. |
